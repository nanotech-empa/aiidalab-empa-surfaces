"""Short-lived provenance lookup for cube result navigation and boundary models."""

from html import escape
from aiida import orm
from aiida_nanotech_empa.utils.cp2k_results import calculation_boundary

CUBE_TYPE = "aiida.calculations:nanotech_empa.cubehandler"
ADSORPTION_TYPE = "aiida.workflows:nanotech_empa.cp2k.fragment_separation"


def adsorption_cube_links(identifier):
    """Resolve only this adsorption workflow's own successful retrieved cubes."""
    root = orm.load_node(identifier)
    if root.process_type != ADSORPTION_TYPE:
        raise ValueError("Select a Cp2kFragmentSeparationWorkChain adsorption result.")
    ready, pending = [], False
    for child in root.called_descendants:
        if not isinstance(child, orm.CalcJobNode) or child.process_type != CUBE_TYPE:
            continue
        pending |= not child.is_terminated
        if not child.is_finished_ok or "retrieved" not in child.outputs:
            continue
        retrieved = child.outputs.retrieved
        try:
            cubes = []
            for name in retrieved.base.repository.list_object_names("out_cubes"):
                if name.endswith(".cube"):
                    with retrieved.base.repository.open(
                        "out_cubes/" + name, mode="rb"
                    ) as handle:
                        if handle.read(1):
                            cubes.append(name)
            cubes.sort()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            continue
        if cubes:
            ready.append(
                {
                    "pk": child.pk,
                    "uuid": child.uuid,
                    "retrieved_uuid": retrieved.uuid,
                    "cubes": cubes,
                }
            )
    ready.sort(key=lambda item: item["pk"])
    if ready:
        return {
            "workflow_uuid": root.uuid,
            "status": "ready",
            "helpers": ready,
            "message": "Cube files are available from the completed cube helper.",
        }
    return {
        "workflow_uuid": root.uuid,
        "status": "pending" if pending or not root.is_terminated else "unavailable",
        "helpers": [],
        "message": (
            "Cube files are pending. Reload after the cube helper finishes."
            if pending or not root.is_terminated
            else "Cube files are unavailable. This workflow has no completed helper with retrieved cube outputs."
        ),
    }


def adsorption_cube_links_html(identifier):
    result = adsorption_cube_links(identifier)
    if not result["helpers"]:
        return "<p>" + escape(result["message"]) + "</p>"
    links = []
    for item in result["helpers"]:
        suffix = "" if len(result["helpers"]) == 1 else f" (helper {item['pk']})"
        links.append(
            f'<a href="handle_cubes.ipynb?pk={item["pk"]}" target="_blank" rel="noopener noreferrer">View cube files{suffix}</a>'
        )
    return "<p>" + " · ".join(links) + "</p>"


def cube_boundary(identifier):
    """Prefer native input models; never substitute legacy StructureData PBC."""
    node = orm.load_node(identifier)
    models = []
    for folder in getattr(node.inputs, "parent_folders", {}).values():
        native = folder.creator
        if (
            native is not None
            and native.process_type == "aiida.calculations:cp2k"
            and "parameters" in native.inputs
        ):
            models.append(calculation_boundary(native.inputs.parameters.get_dict()))
    if models:
        first = models[0]
        if any(model != first for model in models[1:]):
            raise ValueError(
                "The cube sources use different effective boundary models."
            )
        return {**first, "source": "native CP2K CELL and POISSON inputs"}
    raise ValueError(
        "The effective calculation boundary model is unavailable. The viewer cannot assume periodic sampling."
    )
