"""Actual linked widgets, builders and provenance. No native execution/submission."""

# ruff: noqa: F811 -- imported pytest fixtures are requested by parameter name.

import ast
import json
from pathlib import Path

import ipywidgets as ipw
import pytest
import traitlets
from aiida import orm
from aiida.common import LinkType
from aiida.engine import ProcessState
from IPython.display import clear_output

from surfaces_tools.helpers import HART_2_EV
from surfaces_tools.utils import cube_render, cube_results
from surfaces_tools.widgets import cube as adapter
from test_cube_preview_geometry import retained  # noqa: F401

pytest_plugins = ["aiida.tools.pytest_fixtures"]
CAMERA = [20.0, 0, 0, 0, 0, 20.0, 0, 0, 0, 0, 20.0, 0, 0, 0, 0, 1.0]


@pytest.fixture
def widget(retained, monkeypatch):  # noqa: F811
    uuid, _ = retained
    monkeypatch.setattr(
        adapter.CubeArrayData3dViewerWidget, "update_plot", lambda self: None
    )
    value = adapter.HandleCubeFiles()
    value.node_pk = orm.load_node(uuid).pk
    assert value._shared_viewer is not None, value.error_message.value
    value.cube_selector.value = "reduced_ChargeDiff.cube"
    assert value._viewer.cube is not None, value.error_message.value
    value._shared_viewer.viewer3d.viewer._camera_orientation = CAMERA
    yield value
    value.close()


def plan(widget):
    return cube_render.prepare_render(
        widget._helper_uuid,
        widget.cube_selector.value,
        widget.cubehandler_code_widget.value,
        widget._shared_viewer.render_settings(),
    )


def test_shared_component_real_plane_png_and_native_boundary(widget):
    from aiidalab_widgets_empa.cube import CubeViewer

    assert isinstance(widget._shared_viewer, CubeViewer)
    assert widget.slice2d.periodic
    assert "3D XYZ" in widget.model_message.value
    assert widget.slice2d.slice_png.startswith(b"\x89PNG")
    widget.slice2d.offset_txt.value = "0.1"
    widget._shared_viewer._export_slice()
    assert "Download 2D slice PNG" in widget._shared_viewer.slice_download.value
    assert "data:image/png;base64," in widget._shared_viewer.slice_download.value
    settings = plan(widget)
    assert settings["source"]["full_shape"] == [30, 30, 28]
    assert settings["source"]["preview_shape"] == [5, 5, 4]
    assert "native CP2K" in settings["source"]["model"]["source"]


def test_real_builder_preserves_camera_plane_lighting_and_one_cpu(widget):
    from aiida.common.folders import SandboxFolder
    from aiida.engine.utils import instantiate_process
    from aiida.manage import get_manager

    viewer = widget._shared_viewer
    viewer.viewer3d.show_plane_checkbox.value = True
    viewer.opacity.value = 0.4
    request = plan(widget)
    builder = cube_render.render_builder(request)
    parameters = builder.parameters.get_dict()
    assert builder.metadata.options.withmpi is False
    assert builder.metadata.options.max_wallclock_seconds == 600
    assert dict(builder.metadata.options.resources) == {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "num_cores_per_mpiproc": 1,
    }
    step = parameters["steps"][0]
    assert step["options"]["orientation"] == CAMERA
    assert step["options"]["opacity"] == 0.4
    assert parameters["json_files"]["render-plane.json"]["periodic"] is True
    assert parameters["json_files"]["render-lighting.json"] == viewer.viewer3d.lighting
    process = instantiate_process(get_manager().get_runner(), builder)
    try:
        with SandboxFolder() as folder:
            process.prepare_for_submission(folder)
            assert (
                json.loads(Path(folder.get_abs_path("render-plane.json")).read_text())
                == parameters["json_files"]["render-plane.json"]
            )
            assert (
                json.loads(
                    Path(folder.get_abs_path("render-lighting.json")).read_text()
                )
                == parameters["json_files"]["render-lighting.json"]
            )
    finally:
        process.close()


@pytest.mark.parametrize(
    "change", ["opacity", "camera", "isovalue", "new-isovalue", "plane"]
)
def test_actual_view_changes_invalidate_review(widget, change):
    viewer = widget._shared_viewer
    viewer.render_button.click()
    assert not widget.render_submit.disabled, widget.render_review.value
    if change == "opacity":
        viewer.opacity.value = 0.3
    elif change == "camera":
        viewer.viewer3d.viewer._camera_orientation = [
            value * 2 if i == 0 else value for i, value in enumerate(CAMERA)
        ]
    elif change == "plane":
        viewer.viewer3d.show_plane_checkbox.value = True
    else:
        if change == "new-isovalue":
            viewer.viewer3d.isovalues.add_isovalue()
        viewer.viewer3d.isovalues.isovalues.children[-1].color_widget.value = "#123456"
    assert widget.render_submit.disabled and widget._reviewed_render is None


def test_missing_code_keeps_retrieved_preview_and_clearly_blocks_render(widget):
    widget.cubehandler_code_widget.codes = {}
    assert widget._viewer.cube is not None
    widget._shared_viewer.render_button.click()
    assert widget.render_submit.disabled
    assert "Select a configured" in widget.render_review.value


def test_source_reset_drops_view_links_and_saved_state(widget):
    widget._shared_viewer.render_button.click()
    widget.node_pk = 987654321
    assert widget._shared_viewer is None and not widget._view_observers
    assert not widget._cube_cache and widget.render_submit.disabled
    assert widget._helper_uuid is None and widget.retrieved_uuid is None
    assert "no node" in widget.error_message.value.lower()


def test_final_source_check_rejects_changed_full_cube(widget, retained):  # noqa: F811
    request = plan(widget)
    _, folder = retained
    from cubehandler import Cube

    full = Cube.from_file(folder / "ChargeDiff.cube")
    full.data += 0.1
    full.write_cube_file(folder / "ChargeDiff.cube")
    with pytest.raises(ValueError, match="values do not match"):
        cube_render.render_builder(request)


def test_missing_full_source_and_unreviewed_resource_limits_block_render(
    widget, retained
):  # noqa: F811
    with pytest.raises(ValueError, match="one local node"):
        cube_render.prepare_render(
            widget._helper_uuid,
            widget.cube_selector.value,
            widget.cubehandler_code_widget.value,
            widget._shared_viewer.render_settings(),
            resources={**cube_render.RESOURCES, "tasks_per_node": 4},
        )
    _, folder = retained
    (folder / "ChargeDiff.cube").rename(folder / "preserved_original.cube")
    with pytest.raises(ValueError, match="full cube is unavailable"):
        plan(widget)


def test_retrieved_preview_survives_absent_render_remote_folder(retained, monkeypatch):  # noqa: F811
    uuid, folder = retained
    original = orm.load_node(uuid)
    node = orm.CalcJobNode(
        computer=original.computer, process_type=cube_results.CUBE_TYPE
    )
    for name, value in [
        ("parameters", original.inputs.parameters),
        ("code", original.inputs.code),
        ("parent_folders__folder1", original.inputs.parent_folders.folder1),
    ]:
        node.base.links.add_incoming(
            value, link_type=LinkType.INPUT_CALC, link_label=name
        )
    node.base.attributes.set("remote_workdir", str(folder))
    node.store()
    retrieved = orm.FolderData()
    from cubehandler import Cube

    aligned = Cube.from_file(folder / "ChargeDiff.cube")
    aligned.reduce_data_density_slicing()
    preview_path = folder / "standalone-preview.cube"
    aligned.write_cube_file(preview_path)
    retrieved.base.repository.put_object_from_file(
        preview_path, "out_cubes/reduced_ChargeDiff.cube"
    )
    retrieved.base.links.add_incoming(
        node, link_type=LinkType.CREATE, link_label="retrieved"
    )
    retrieved.store()
    node.set_process_state(ProcessState.FINISHED)
    node.set_exit_status(0)
    node.seal()
    monkeypatch.setattr(
        adapter.CubeArrayData3dViewerWidget, "update_plot", lambda self: None
    )
    viewer = adapter.HandleCubeFiles()
    try:
        viewer.node_pk = node.pk
        viewer.cube_selector.value = "reduced_ChargeDiff.cube"
        viewer._shared_viewer.viewer3d.viewer._camera_orientation = CAMERA
        assert viewer._viewer.cube is not None, viewer.error_message.value
        viewer._shared_viewer.render_button.click()
        assert (
            viewer.render_submit.disabled
            and "previews remain viewable" in viewer.render_review.value
        )
    finally:
        viewer.close()


def test_paused_local_native_job_blocks_new_render(widget, aiida_localhost):
    node = orm.CalcJobNode(computer=aiida_localhost).store()
    node.set_process_state(ProcessState.RUNNING)
    node.pause()
    with pytest.raises(ValueError, match="still active"):
        cube_render._assert_local_slot_idle(aiida_localhost.uuid)


def test_actual_query_reuses_same_render_and_callback_consumes_review(
    widget, monkeypatch
):
    calls = []

    def capture(builder):
        node = orm.CalcJobNode(
            computer=builder.code.computer,
            process_type=builder.process_class.build_process_type(),
        )
        node.base.links.add_incoming(
            builder.parameters.store(),
            link_type=LinkType.INPUT_CALC,
            link_label="parameters",
        )
        node.store()
        calls.append(str(node.uuid))
        return node

    monkeypatch.setattr(cube_render, "submit", capture)
    request = plan(widget)
    first = cube_render.submit_reviewed_render(request)
    second = cube_render.submit_reviewed_render(request)
    assert first["uuid"] == second["uuid"] and second["reused"]
    assert len(calls) == 1
    widget._shared_viewer.render_button.click()
    widget.render_submit.click()
    widget._submit_render()
    assert widget._reviewed_render is None and widget.render_submit.disabled
    assert len(calls) == 1


@pytest.mark.parametrize(
    "output", ["absent", "empty", "corrupt", "wrong-size", "wrong-format", "valid"]
)
def test_completed_render_reuse_requires_the_requested_image(
    widget, monkeypatch, output
):
    import io

    from PIL import Image

    request = plan(widget)
    builder = cube_render.render_builder(request)
    parameters = builder.parameters.get_dict()
    previous = orm.CalcJobNode(
        computer=builder.code.computer,
        process_type=builder.process_class.build_process_type(),
    )
    previous.base.links.add_incoming(
        orm.Dict(parameters).store(), LinkType.INPUT_CALC, "parameters"
    )
    previous.store()
    retrieved = None
    if output != "absent":
        retrieved = orm.FolderData()
        if output != "empty":
            data = b"native render failed"
            if output != "corrupt":
                options = parameters["steps"][0]["options"]
                size = (options["width"], options["height"])
                if output == "wrong-size":
                    size = (size[0] - 1, size[1])
                stream = io.BytesIO()
                Image.new("RGB", size, "#336699").save(
                    stream, format="JPEG" if output == "wrong-format" else "PNG"
                )
                data = stream.getvalue()
            retrieved.base.repository.put_object_from_filelike(
                io.BytesIO(data), "out_cubes/render.png"
            )
        retrieved.store()
        retrieved.base.links.add_incoming(previous, LinkType.CREATE, "retrieved")
    previous.set_process_state(ProcessState.FINISHED)
    previous.set_exit_status(0)
    previous.seal()
    before = previous.base.attributes.all
    repository_before = retrieved.base.repository.metadata if retrieved else None
    calls = []

    def capture(new_builder):
        node = orm.CalcJobNode(
            computer=new_builder.code.computer,
            process_type=new_builder.process_class.build_process_type(),
        )
        node.base.links.add_incoming(
            new_builder.parameters.store(), LinkType.INPUT_CALC, "parameters"
        )
        node.store()
        calls.append(node.uuid)
        return node

    monkeypatch.setattr(cube_render, "submit", capture)
    first = cube_render.submit_reviewed_render(request)
    second = cube_render.submit_reviewed_render(request)
    if output == "valid":
        assert first["reused"] and first["uuid"] == previous.uuid
        assert not calls
    else:
        assert not first["reused"] and first["uuid"] != previous.uuid
        assert calls == [first["uuid"]]
    assert second["reused"] and second["uuid"] == first["uuid"]
    assert previous.base.attributes.all == before
    if retrieved:
        assert retrieved.base.repository.metadata == repository_before


def test_named_view_reopens_without_scientific_mutation(widget):
    node = orm.load_node(widget._helper_uuid)
    attributes = node.base.attributes.all
    repository = node.outputs.retrieved.get_object_content(
        "out_cubes/reduced_ChargeDiff.cube"
    )
    widget._shared_viewer.opacity.value = 0.35
    original_digest = plan(widget)["digest"]
    identifier = widget._save_view("signed interface")
    pk = node.pk
    widget.node_pk = None
    widget.node_pk = pk
    widget._load_view(identifier)
    assert widget._shared_viewer.opacity.value == 0.35
    assert plan(widget)["digest"] == original_digest
    assert node.base.attributes.all == attributes
    assert (
        node.outputs.retrieved.get_object_content("out_cubes/reduced_ChargeDiff.cube")
        == repository
    )


def adsorption_viewer_class():
    notebook = json.loads(
        (Path(__file__).parents[1] / "view_adsorption_energy.ipynb").read_text()
    )
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "class FragmentSeparationViewer" in "".join(cell.get("source", []))
    )
    namespace = {
        "ipw": ipw,
        "trt": traitlets,
        "orm": orm,
        "clear_output": clear_output,
        "HART_2_EV": HART_2_EV,
        "adsorption_cube_links_html": cube_results.adsorption_cube_links_html,
    }
    exec(compile(ast.parse(source), "view_adsorption_energy.ipynb", "exec"), namespace)
    return namespace["FragmentSeparationViewer"]


def test_actual_adsorption_page_link_reload_and_different_workflow_reject_stale(
    retained,
):  # noqa: F811
    helper_uuid, _ = retained
    helper = orm.load_node(helper_uuid)
    root = helper.caller
    viewer = adsorption_viewer_class()()
    try:
        viewer.pk_widget.value = str(root.pk)
        viewer.load()
        assert viewer._workflow_uuid == root.uuid
        assert f"handle_cubes.ipynb?pk={helper.pk}" in viewer.cube_link.value
        assert not hasattr(viewer, "node")
        viewer.load()
        assert viewer.cube_link.value.count("View cube files") == 1
        viewer.pk_widget.value = str(helper.pk)
        assert not viewer.cube_link.value and viewer._workflow_uuid is None
        viewer.load()
        assert not viewer.cube_link.value and viewer._workflow_uuid is None
        viewer.pk_widget.value = str(root.pk)
        viewer.load()
        assert f"?pk={helper.pk}" in viewer.cube_link.value
    finally:
        viewer.close()


def test_pending_and_missing_cube_outputs_do_not_borrow_another_workflow_link(retained):  # noqa: F811
    root = orm.WorkChainNode(process_type=cube_results.ADSORPTION_TYPE).store()
    assert cube_results.adsorption_cube_links(root.uuid)["status"] == "pending"
    root.set_process_state(ProcessState.FINISHED)
    root.set_exit_status(0)
    root.seal()
    assert cube_results.adsorption_cube_links(root.uuid)["status"] == "unavailable"
    assert "handle_cubes" not in cube_results.adsorption_cube_links_html(root.uuid)


@pytest.mark.parametrize("content", [None, b"", "directory"])
def test_completed_helper_without_nonempty_cube_has_no_link(
    aiida_profile_clean, aiida_localhost, content
):
    root = orm.WorkChainNode(process_type=cube_results.ADSORPTION_TYPE).store()
    helper = orm.CalcJobNode(
        computer=aiida_localhost, process_type=cube_results.CUBE_TYPE
    )
    helper.base.links.add_incoming(
        root, link_type=LinkType.CALL_CALC, link_label="cubes"
    )
    helper.store()
    if content is not None:
        import io

        retrieved = orm.FolderData()
        name = (
            "out_cubes/empty.cube/child"
            if content == "directory"
            else "out_cubes/empty.cube"
        )
        retrieved.base.repository.put_object_from_filelike(
            io.BytesIO(b"" if content == b"" else b"directory"), name
        )
        retrieved.base.links.add_incoming(
            helper, link_type=LinkType.CREATE, link_label="retrieved"
        )
        retrieved.store()
    for node in (helper, root):
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(0)
        node.seal()
    result = cube_results.adsorption_cube_links(root.uuid)
    assert result["status"] == "unavailable" and not result["helpers"]


def test_legacy_workflow_choice_is_not_substituted_for_missing_native_boundary(
    aiida_profile_clean,
):
    root = orm.WorkChainNode(process_type=cube_results.ADSORPTION_TYPE)
    root.base.links.add_incoming(
        orm.Dict({"periodic": "NONE"}).store(),
        link_type=LinkType.INPUT_WORK,
        link_label="dft_params",
    )
    root.store()
    helper = orm.CalcJobNode(process_type=cube_results.CUBE_TYPE)
    helper.base.links.add_incoming(
        root, link_type=LinkType.CALL_CALC, link_label="cube"
    )
    helper.store()
    with pytest.raises(ValueError, match="cannot assume periodic sampling"):
        cube_results.cube_boundary(helper.uuid)


def test_partial_native_boundary_is_explicitly_unavailable_in_real_viewer(retained):  # noqa: F811
    original = orm.load_node(retained[0])
    native = orm.CalcJobNode(
        process_type="aiida.calculations:cp2k", computer=original.computer
    )
    parameters = orm.Dict(
        {
            "FORCE_EVAL": {
                "SUBSYS": {"CELL": {"PERIODIC": "XY"}},
                "DFT": {"POISSON": {"PERIODIC": "XY", "POISSON_SOLVER": "MT"}},
            }
        }
    ).store()
    native.base.links.add_incoming(
        parameters, link_type=LinkType.INPUT_CALC, link_label="parameters"
    )
    native.store()
    remote = orm.RemoteData(computer=original.computer, remote_path="/synthetic/native")
    remote.base.links.add_incoming(
        native, link_type=LinkType.CREATE, link_label="remote_folder"
    )
    remote.store()
    helper = orm.CalcJobNode(
        process_type=cube_results.CUBE_TYPE, computer=original.computer
    )
    helper.base.links.add_incoming(
        remote, link_type=LinkType.INPUT_CALC, link_label="parent_folders__folder1"
    )
    helper.store()
    retrieved = orm.FolderData()
    import io

    retrieved.base.repository.put_object_from_filelike(
        io.BytesIO(b"not read for unsupported sampling"), "out_cubes/partial.cube"
    )
    retrieved.base.links.add_incoming(
        helper, link_type=LinkType.CREATE, link_label="retrieved"
    )
    retrieved.store()
    for node in (native, helper):
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(0)
        node.seal()
    viewer = adapter.HandleCubeFiles()
    try:
        viewer.node_pk = helper.pk
        assert viewer._shared_viewer is None
        assert (
            "2D XY" in viewer.error_message.value
            and "Partial-axis" in viewer.error_message.value
        )
    finally:
        viewer.close()
