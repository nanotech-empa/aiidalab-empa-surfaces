"""The visible NEB layout must survive the real Show/section-rebuild path."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import aiidalab_widgets_base as awb
import ipywidgets as ipw
import numpy as np
import pytest
import traitlets as tr
from aiida import engine, orm, plugins
from aiidalab_widgets_empa import ProcessResourcesWidget
from ase import Atoms

from surfaces_tools.utils import neb_resources
from surfaces_tools.widgets import inputs

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def form_factory(aiida_profile_tmp, aiida_localhost, monkeypatch):
    assert aiida_profile_tmp.is_test_profile

    def unexpected_submit(*args, **kwargs):
        raise AssertionError("Form inspection must never submit a calculation")

    monkeypatch.setattr(engine, "submit", unexpected_submit)
    monkeypatch.setattr(inputs, "display", lambda *args: None)
    monkeypatch.setattr(inputs, "clear_output", lambda *args, **kwargs: None)
    code = orm.InstalledCode(
        label="cp2k-form-inspection",
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    ).store()

    def make(count):
        images = []
        for fraction in np.linspace(0, 1, count):
            atoms = Atoms(
                "NH3",
                positions=[
                    [6, 6, 6.4 - 0.8 * fraction],
                    [6.95, 6, 6],
                    [5.525, 6.8227241336, 6],
                    [5.525, 5.1772758664, 6],
                ],
                cell=[12, 12, 12],
                pbc=True,
            )
            images.append(orm.StructureData(ase=atoms).store())
        manager = awb.StructureManagerWidget(
            importers=[], editors=[], storable=False, node_class="StructureData"
        )
        details = inputs.InputDetails(structure_info_separate=True)
        details.structure_manager = manager
        links = [tr.dlink((manager, "structure"), (details, "structure"))]
        details.neb = True
        manager.input_structure = images[0]
        section = next(
            s for s in details.displayed_sections if isinstance(s, inputs.NebWidget)
        )
        section.initial_row.pk.value = images[0].pk
        section.last_row.pk.value = images[-1].pk
        for image in reversed(images[1:-1]):
            section.initial_row.insert.click()
            section.replica_rows[0].pk.value = image.pk

        resources = ProcessResourcesWidget()
        for name in ("neb", "n_replica_trait", "n_replica_per_group_trait"):
            links.append(tr.dlink((details, name), (resources, name)))
        links.append(
            tr.dlink(
                (resources, "nproc_replica_trait"), (details, "nproc_replica_trait")
            )
        )
        code_widget = awb.ComputationalResourcesWidget(default_calc_job_plugin="cp2k")
        code_widget.value = str(code.uuid)
        links.append(tr.dlink((code_widget, "value"), (details, "selected_code")))
        namespace = {
            "resources": resources,
            "input_details": details,
            "structure_selector": manager,
            "code_input_widget": code_widget,
            "protocol": SimpleNamespace(value="standard"),
            "workflow_description": SimpleNamespace(value="Form-only regression"),
            "output": ipw.Output(),
            "clear_output": lambda *args, **kwargs: None,
            "neb_resources": neb_resources,
            "orm": orm,
            "Cp2kNebWorkChain": plugins.WorkflowFactory("nanotech_empa.cp2k.neb"),
        }
        notebook = json.loads(
            (Path(__file__).parents[1] / "submit_neb.ipynb").read_text()
        )
        tree = ast.parse("".join(notebook["cells"][6]["source"]))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(
            compile(
                ast.Module(body=functions, type_ignores=[]),
                "submit_neb.ipynb:6",
                "exec",
            ),
            namespace,
        )
        helper = neb_resources.NebResourceHelper(namespace["_neb_resource_info"])
        helper.show.value = True
        for name in ("n_replica_trait", "n_replica_per_group_trait"):
            details.observe(helper.update, name)
        for widget in (resources.nodes_widget, resources.tasks_per_node_widget):
            widget.observe(helper.update, "value")
        return SimpleNamespace(
            images=images,
            manager=manager,
            details=details,
            resources=resources,
            helper=helper,
            namespace=namespace,
            links=links,
        )

    return make


def section(form):
    return next(
        s for s in form.details.displayed_sections if isinstance(s, inputs.NebWidget)
    )


def set_layout(form, group, nodes, tasks):
    section(form).n_replica_per_group.value = group
    form.resources.tasks_per_node_widget.value = tasks
    form.resources.nodes_widget.value = nodes
    form.resources.threads_per_task_widget.value = 1
    form.resources.walltime_widget.value = "00:10:00"
    form.helper.update()


def assert_layout(form, group, nodes, tasks, nproc):
    current = section(form)
    count = current.n_replica.value
    assert current.n_replica_per_group.value == group
    assert current.n_replica_per_group_trait == group
    assert form.details.n_replica_per_group_trait == group
    assert form.resources.n_replica_per_group_trait == group
    assert current.nproc_rep.value == str(nproc)
    assert form.details.nproc_replica_trait == nproc
    assert form.resources.nproc_replica_trait == nproc
    assert form.resources.nodes == nodes
    assert form.resources.tasks_per_node == tasks
    assert form.resources.threads_per_task == 1
    layout = form.namespace["_neb_resource_info"]()
    assert layout["n_replica_per_group"] == group
    assert layout["nproc_replica"] == nproc
    assert layout["n_groups"] == count // group
    assert f"{count} replicas at {group} per group" in form.helper.text.value
    assert f"NPROC_REP = {nproc}" in form.helper.text.value
    builder = form.namespace["prepare_inputs"]()
    assert builder is not None
    assert builder.neb_params["number_of_replica"] == count
    assert builder.neb_params["nproc_rep"] == nproc
    assert builder.options["resources"] == {
        "num_machines": nodes,
        "num_mpiprocs_per_machine": tasks,
        "num_cores_per_mpiproc": 1,
    }
    assert builder.structure.uuid == form.images[0].uuid
    assert [
        builder.replicas[f"replica_{i:03d}"].uuid for i in range(1, len(form.images))
    ] == [n.uuid for n in form.images[1:]]
    assert [r.pk.value for r in current.all_rows()] == [n.pk for n in form.images]


def test_all_five_show_actions_keep_group_model_helper_and_builder(form_factory):
    form = form_factory(5)
    set_layout(form, 5, 1, 4)
    assert_layout(form, 5, 1, 4, 4)
    for image in form.images:
        current = section(form)
        next(r for r in current.all_rows() if r.pk.value == image.pk).show.click()
        current = section(form)
        current.tabs.selected_index = 0
        current.tabs.selected_index = 1
        assert_layout(form, 5, 1, 4, 4)


@pytest.mark.parametrize(
    "count,group,nodes,tasks,nproc",
    [(5, 5, 1, p, p) for p in (1, 2, 4, 8)]
    + [(5, 1, 1, p, p // 5) for p in (5, 10, 20, 40)]
    + [(6, 3, 1, 8, 4), (6, 2, 1, 12, 4), (6, 1, 1, 24, 4), (6, 6, 1, 4, 4)]
    + [(5, 5, 2, 4, 8), (6, 3, 2, 4, 4), (5, 1, 5, 4, 4)],
)
def test_integer_layout_matrix_after_show_and_tab_changes(
    form_factory, count, group, nodes, tasks, nproc
):
    form = form_factory(count)
    set_layout(form, group, nodes, tasks)
    assert_layout(form, group, nodes, tasks, nproc)
    section(form).last_row.show.click()
    section(form).tabs.selected_index = 0
    section(form).tabs.selected_index = 1
    assert_layout(form, group, nodes, tasks, nproc)


def test_intentional_group_changes_and_invalidated_divisor(form_factory):
    form = form_factory(5)
    set_layout(form, 5, 1, 4)
    section(form).last_row.show.click()
    section(form).n_replica_per_group.value = 1
    assert_layout(form, 1, 5, 4, 4)  # Existing intentional autosizing is retained.
    section(form).n_replica_per_group.value = 5
    assert_layout(form, 5, 1, 4, 4)
    section(form).n_replica.value = 10
    assert_layout(form, 5, 2, 4, 4)
    section(form).initial_row.show.click()
    assert_layout(form, 5, 2, 4, 4)
    section(form).n_replica.value = 7
    current = section(form)
    assert list(current.n_replica_per_group.options) == [1, 7]
    assert current.n_replica_per_group.value == 1
    assert current.n_replica_per_group_trait == 1
    assert form.details.n_replica_per_group_trait == 1
    assert form.resources.n_replica_per_group_trait == 1
    form.helper.update()
    assert "7 replicas at 1 per group" in form.helper.text.value
    layout = form.namespace["_neb_resource_info"]()
    section(form).last_row.show.click()
    assert_layout(
        form, 1, layout["nodes"], layout["tasks_per_node"], layout["nproc_replica"]
    )


def test_linked_group_can_arrive_before_its_divisor_options():
    widget = inputs.NebWidget()
    widget.n_replica_per_group_trait = 5
    widget.n_replica_trait = 10
    assert widget.n_replica_per_group.value == 5
    assert widget.n_replica_per_group_trait == 5
