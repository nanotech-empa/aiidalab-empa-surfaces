"""Surfaces adapter to the shared CubeViewer and cubehandler renderer contract.

Only GUI callbacks submit. Source/result nodes stay immutable; named views are
explicit user annotations. Callbacks retain UUIDs and JSON, never ORM instances.
"""

import copy
import fcntl
import hashlib
import json
import math
from importlib.metadata import version
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from tempfile import gettempdir
from uuid import UUID, uuid4

from aiida import orm, plugins
from aiida.engine import submit
from aiida.orm.implementation import clean_value
from cubehandler import Cube
import cubehandler
from cubehandler.cube import ANG_TO_BOHR, RenderLighting
from cubehandler.slice_plane import SlicePlane
from PIL import ImageColor
from aiidalab_widgets_empa.cube_views import render_settings_from_view
from aiida_nanotech_empa.utils.cubehandler_results import rendered_images
from aiida.common.exceptions import NotExistent
from .cube_results import CUBE_TYPES, cube_boundary, is_cube_helper

_VIEW_PREFIX = "surfaces_cube_view_"
RESOURCES = {
    "nodes": 1,
    "tasks_per_node": 1,
    "threads_per_task": 1,
    "walltime_seconds": 600,
}


def _plain(value):
    # Match AiiDA's stored numeric representation for durable view identities.
    return clean_value(json.loads(json.dumps(value, allow_nan=False)))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _renderer_identity():
    """Identify only the local renderer implementation, not the whole app."""
    root = Path(cubehandler.__file__).parent
    return {
        "version": version("cubehandler"),
        "files": {
            name: _sha(root / name)
            for name in ("cube.py", "slice_plane.py", "cli/main.py")
        },
    }


def validate_settings(values, *, periodic):
    p = _plain(values)
    required = {"orientation", "iso", "width", "height", "background", "opacity"}
    if not required <= p.keys() or p.keys() - required - {"lighting", "slice_plane"}:
        raise ValueError("Incomplete or unknown render settings.")
    Cube._decompose_ngl_orientation(p["orientation"])
    if not p["iso"]:
        raise ValueError("Add at least one signed isovalue.")
    for pair in p["iso"]:
        if len(pair) != 2 or not math.isfinite(pair[0]):
            raise ValueError("Invalid isovalue.")
        ImageColor.getrgb(pair[1])
    ImageColor.getrgb(p["background"])
    if any(
        type(p[k]) is not int or not 320 <= p[k] <= 4096 for k in ("width", "height")
    ):
        raise ValueError("Render dimensions must be 320–4096 pixels.")
    if not math.isfinite(p["opacity"]) or not 0 <= p["opacity"] <= 1:
        raise ValueError("Invalid surface opacity.")
    if "lighting" in p:
        RenderLighting(**p["lighting"])
    if "slice_plane" in p:
        plane = SlicePlane(**p["slice_plane"])
        if plane.periodic != periodic:
            raise ValueError("The slice boundary model differs from the calculation.")
    return p


def _source(identifier, cube_name):
    from surfaces_tools.widgets.cube import _load_cube_preview

    node = orm.load_node(identifier)
    if not is_cube_helper(node) or not node.is_finished_ok:
        raise ValueError("Select a completed cubehandler result.")
    if node.computer.transport_type != "core.local":
        raise ValueError("This render route is qualified only for local cube files.")
    if "remote_folder" not in node.outputs:
        raise ValueError(
            "The original full-grid render source is unavailable. Retrieved previews remain viewable."
        )
    if PurePosixPath(cube_name).name != cube_name or not cube_name.endswith(".cube"):
        raise ValueError("Select one retrieved cube filename.")
    matches = []
    for step in node.inputs.parameters.get_dict().get("steps", []):
        prefix = step.get("options", {}).get("prefix", "reduced_")
        if step.get("command") != "shrink" or not cube_name.startswith(prefix):
            continue
        full_name = cube_name[len(prefix) :]
        for arg in step.get("args", []):
            path = PurePosixPath(arg)
            if (
                not path.is_absolute()
                and ".." not in path.parts
                and fnmatch(full_name, path.name)
            ):
                matches.append(str(path.parent / full_name))
    if len(matches) != 1:
        raise ValueError("One recorded full-cube source is required for rendering.")
    relative = matches[0]
    path = Path(node.get_remote_workdir()) / relative
    if not path.is_file():
        raise ValueError(
            "The original full cube is unavailable. No render was submitted."
        )
    preview, message = _load_cube_preview(node.uuid, cube_name)
    if "unverified" in message.lower():
        raise ValueError("The preview is unverified against its full cube.")
    full = Cube.from_file(path, read_data=False)
    import numpy as np

    if not np.allclose(full.cell, preview.cell, atol=2e-4, rtol=0) or not np.allclose(
        full.origin, preview.origin, atol=1e-6, rtol=0
    ):
        raise ValueError("Full and displayed cube geometry differ.")
    model = cube_boundary(node.uuid)
    if model["periodic"] not in ("XYZ", "NONE"):
        raise ValueError(
            "The shared viewer currently supports XYZ or NONE sampling. Partial-axis rendering is unavailable."
        )
    parent_folders = getattr(node.inputs, "parent_folders", {})
    original = parent_folders.get("folder1", node.outputs.remote_folder)
    return {
        "helper_uuid": node.uuid,
        "cube_name": cube_name,
        "source_remote_uuid": node.outputs.remote_folder.uuid,
        "original_remote_uuid": original.uuid,
        "relative_path": relative,
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
        "retrieved_uuid": node.outputs.retrieved.uuid,
        "computer_uuid": node.computer.uuid,
        "cell_A": (full.cell / ANG_TO_BOHR).tolist(),
        "origin_A": (full.origin / ANG_TO_BOHR).tolist(),
        "full_shape": [int(n) for n in full.cell_n],
        "preview_shape": list(preview.data.shape),
        "model": model,
    }


def prepare_render(identifier, cube_name, code_uuid, settings, *, resources=None):
    source = _source(identifier, cube_name)
    if not code_uuid:
        raise ValueError("Select a configured local cubehandler renderer.")
    code = orm.load_code(code_uuid)
    if (
        code.default_calc_job_plugin != "nanotech_empa.cubehandler"
        or code.computer.uuid != source["computer_uuid"]
    ):
        raise ValueError(
            "Select the local cubehandler code on the cube source computer."
        )
    budget = dict(RESOURCES if resources is None else resources)
    if budget != RESOURCES:
        raise ValueError(
            "Use one local node, one CPU process, one thread and 600 seconds for this render."
        )
    values = validate_settings(settings, periodic=source["model"]["periodic"] == "XYZ")
    plan = _plain(
        {
            "source": source,
            "code_uuid": code.uuid,
            "renderer": _renderer_identity(),
            "settings": values,
            "resources": budget,
        }
    )
    plan["digest"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()
    ).hexdigest()
    return plan


def _existing_render(plan):
    q = orm.QueryBuilder().append(
        orm.Dict,
        filters={"attributes.surfaces_render.digest": plan["digest"]},
        tag="parameters",
    )
    q.append(
        orm.CalcJobNode,
        with_incoming="parameters",
        edge_filters={"label": "parameters"},
        filters={"process_type": {"in": CUBE_TYPES}},
        project="*",
    )
    for node in q.all(flat=True):
        if not node.is_terminated:
            return node.uuid
        if node.is_finished_ok:
            try:
                images = rendered_images(
                    node.inputs.parameters.get_dict(),
                    node.outputs.retrieved.base.repository.open,
                )
            except (OSError, ValueError, SyntaxError, NotExistent, AttributeError):
                continue
            if images:
                return node.uuid
    return None


def _assert_local_slot_idle(computer_uuid):
    q = orm.QueryBuilder().append(
        orm.Computer, filters={"uuid": computer_uuid}, tag="computer"
    )
    q.append(
        orm.CalcJobNode,
        with_computer="computer",
        filters={"attributes.process_state": {"in": ["created", "waiting", "running"]}},
        project="*",
    )
    active = [str(n.uuid) for n in q.all(flat=True)]
    if active:
        raise ValueError(
            "A local native job is still active. Finish it before rendering: "
            + ", ".join(active)
        )


def render_builder(plan):
    source = plan["source"]
    fresh = prepare_render(
        source["helper_uuid"],
        source["cube_name"],
        plan["code_uuid"],
        plan["settings"],
        resources=plan["resources"],
    )
    if fresh != plan:
        raise ValueError(
            "The cube source, renderer or render settings changed. Review the render again."
        )
    p = plan["settings"]
    companions = {}
    options = {
        "orientation": p["orientation"],
        "iso": [f"{v:.16g}:{c}" for v, c in p["iso"]],
        "format": "png",
        "output": "out_cubes/render.png",
        "width": p["width"],
        "height": p["height"],
        "background": p["background"],
        "opacity": p["opacity"],
    }
    for key, filename in [
        ("lighting", "render-lighting.json"),
        ("slice_plane", "render-plane.json"),
    ]:
        if key in p:
            companions[filename] = p[key]
            options[key] = filename
    builder = plugins.CalculationFactory("nanotech_empa.cubehandler").get_builder()
    builder.code = orm.load_code(plan["code_uuid"])
    builder.parameters = orm.Dict(
        {
            "steps": [
                {
                    "command": "render",
                    "args": ["cube_source/" + source["relative_path"]],
                    "options": options,
                }
            ],
            "json_files": companions,
            "surfaces_render": copy.deepcopy(plan),
        }
    )
    builder.parent_folders = {
        "cube_source": orm.load_node(source["source_remote_uuid"])
    }
    if source["original_remote_uuid"] != source["source_remote_uuid"]:
        builder.parent_folders.original_source = orm.load_node(
            source["original_remote_uuid"]
        )
    builder.metadata.label = "cube-render"
    builder.metadata.description = (
        "Full-grid render of " + source["helper_uuid"] + " / " + source["cube_name"]
    )
    builder.metadata.disable_cache = True
    builder.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
        "withmpi": False,
        "max_wallclock_seconds": 600,
        "environment_variables": {"OMP_NUM_THREADS": "1", "VTK_SMP_MAX_THREADS": "1"},
    }
    return builder


def submit_reviewed_render(plan):
    # Serialize the final local admission and submission across notebook kernels.
    lock = Path(gettempdir()) / (
        "surfaces-cube-render-" + str(UUID(plan["source"]["computer_uuid"])) + ".lock"
    )
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "Another render is being submitted. Refresh before trying again."
            ) from exc
        builder = render_builder(plan)
        existing = _existing_render(plan)
        if existing:
            return {"uuid": existing, "reused": True}
        _assert_local_slot_idle(plan["source"]["computer_uuid"])
        node = submit(builder)
        return {"uuid": node.uuid, "reused": False}


def saved_views(identifier):
    node = orm.load_node(identifier)
    return sorted(
        [v for k, v in node.base.extras.all.items() if k.startswith(_VIEW_PREFIX)],
        key=lambda v: (v["name"].casefold(), v["id"]),
    )


def saved_view(identifier, view_id):
    key = _VIEW_PREFIX + str(UUID(view_id))
    record = orm.load_node(identifier).base.extras.get(key, None)
    if record is None:
        raise ValueError("This saved visualization is no longer available.")
    return _plain(record)


def save_view(identifier, cube_name, name, state, view_id=None):
    node = orm.load_node(identifier)
    model = cube_boundary(node.uuid)
    if not name.strip():
        raise ValueError("Give this visualization a name.")
    if state["plane"]["periodic"] != (model["periodic"] == "XYZ"):
        raise ValueError("Saved-view boundaries differ from the source.")
    validate_settings(
        render_settings_from_view(state), periodic=state["plane"]["periodic"]
    )
    view_id = str(UUID(view_id)) if view_id else str(uuid4())
    if view_id and node.base.extras.get(_VIEW_PREFIX + view_id, None) is not None:
        old = saved_view(node.uuid, view_id)
        if old["helper_uuid"] != node.uuid:
            raise ValueError("The saved visualization belongs to another cube result.")
    record = _plain(
        {
            "id": view_id,
            "name": name.strip(),
            "label": name.strip() + " · " + cube_name,
            "helper_uuid": node.uuid,
            "cube_name": cube_name,
            "visualization": state,
        }
    )
    node.base.extras.set(_VIEW_PREFIX + view_id, record)
    return view_id


def remove_view(identifier, view_id):
    saved_view(identifier, view_id)
    orm.load_node(identifier).base.extras.delete(_VIEW_PREFIX + str(UUID(view_id)))
