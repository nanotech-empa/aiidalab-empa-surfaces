"""Cube geometry and actual retained-preview widget regressions. No native jobs."""

from pathlib import Path
import hashlib
import numpy as np
import pytest
from ase import Atoms
from aiida import orm, plugins
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from cubehandler import Cube
from cubehandler.cube import ANG_TO_BOHR

pytest_plugins = ["aiida.tools.pytest_fixtures"]


class LegacyCube(Cube):
    """Construct an invented archive with the former column-scaled header."""

    def reduce_data_density_slicing(self, points_per_angstrom=2):
        step = np.round(
            1.0 / (points_per_angstrom * np.linalg.norm(self.dcell, axis=1))
        ).astype(int)
        self.data = self.data[:: step[0], :: step[1], :: step[2]]

    @property
    def dcell_au(self):
        return self.cell / np.asarray(self.cell_n)

    def write_cube_file(self, filename, low_precision=False):
        super().write_cube_file(filename, low_precision=low_precision)
        path = Path(filename)
        lines = path.read_text().splitlines(keepends=True)
        vectors = self.cell / np.asarray(self.data.shape)
        for axis in range(3):
            v = vectors[axis]
            lines[axis + 3] = (
                f"{self.data.shape[axis]:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}\n"
            )
        path.write_text("".join(lines))


def old_class():
    return LegacyCube


def viewer_module():
    from surfaces_tools.widgets import cube

    return cube


def make_cube(cls=Cube, skew=True, shape=(30, 30, 28)):
    cell = (
        np.array([[2.0000041, 0, 0], [-1.000002, 1.7320446, 0], [0, 0, 2.0]])
        if skew
        else np.diag([2.0, 2.0, 2.0])
    )
    atoms = Atoms("BN", positions=[[0.2, 0.3, 0.6], [0.5, 0.7, 0.6]], cell=cell)
    i, j, k = np.indices(shape)
    return cls(
        title="synthetic source",
        comment="density",
        ase_atoms=atoms,
        origin=np.array([0.2, 0.3, 0.4]),
        cell=cell * ANG_TO_BOHR,
        data=(i + 2 * j - 3 * k + 0.5) * 0.0001,
    )


@pytest.mark.parametrize(
    "cell",
    [
        [[2, 0, 0], [-1, 1.732, 0], [0, 0, 3]],
        [[2, 0.2, -0.1], [0.4, 3, 0.3], [-0.2, 0.1, 4]],
        [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
    ],
)
def test_row_vectors_roundtrip_and_voxel_metric(tmp_path, cell):
    c = make_cube(shape=(7, 9, 11))
    c.cell = np.array(cell) * ANG_TO_BOHR
    c.ase_atoms.set_cell(cell)
    np.testing.assert_allclose(c.dcell_au, c.cell / np.array(c.data.shape)[:, None])
    path = tmp_path / "roundtrip.cube"
    c.write_cube_file(path, low_precision=False)
    got = Cube.from_file(path)
    np.testing.assert_allclose(got.cell, c.cell, atol=6e-6, rtol=0)
    np.testing.assert_allclose(got.data, c.data, atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        got.ase_atoms.positions, c.ase_atoms.positions, atol=5e-7, rtol=0
    )
    np.testing.assert_allclose(got.origin, c.origin, atol=1e-6, rtol=0)


def test_old_writer_reproduces_skew():
    c = make_cube(old_class())
    c.reduce_data_density_slicing()
    old_header_cell = c.cell / np.array(c.data.shape) * np.array(c.data.shape)[:, None]
    assert c.data.shape == (5, 4, 4)
    assert abs(old_header_cell[1, 0] - c.cell[1, 0]) > 0.1


@pytest.mark.parametrize("skew", [True, False])
def test_aligned_slicing_preserves_original_cartesian_samples_and_cell(skew):
    c = make_cube(skew=skew)
    data = c.data.copy()
    cell = c.cell.copy()
    origin = c.origin.copy()
    n = np.array(data.shape)
    c.reduce_data_density_slicing()
    new = np.array(c.data.shape)
    steps = n // new
    assert np.all(n % new == 0)
    np.testing.assert_array_equal(c.cell_n, new)
    np.testing.assert_array_equal(c.cell, cell)
    np.testing.assert_array_equal(c.origin, origin)
    np.testing.assert_array_equal(c.data, data[:: steps[0], :: steps[1], :: steps[2]])
    new_i = np.array([new[0] - 1, new[1] - 1, new[2] - 1])
    np.testing.assert_allclose(
        origin + new_i @ c.dcell_au,
        origin + (new_i * steps) @ (cell / n[:, None]),
        atol=1e-13,
    )


def test_orthogonal_divisible_legacy_output_is_unchanged(tmp_path):
    old = make_cube(old_class(), False, shape=(32, 32, 32))
    new = make_cube(Cube, False, shape=(32, 32, 32))
    old.reduce_data_density_slicing()
    new.reduce_data_density_slicing()
    old.write_cube_file(tmp_path / "old.cube", low_precision=True)
    new.write_cube_file(tmp_path / "new.cube", low_precision=True)
    assert (tmp_path / "old.cube").read_bytes() == (tmp_path / "new.cube").read_bytes()


def test_skimage_updates_grid_count():
    c = make_cube()
    cell = c.cell.copy()
    c.reduce_data_density_skimage()
    np.testing.assert_array_equal(c.cell_n, c.data.shape)
    np.testing.assert_allclose(c.dcell_au, cell / np.array(c.data.shape)[:, None])


@pytest.fixture(params=["entry-point", "class-path"])
def retained(aiida_profile_clean, aiida_localhost, tmp_path, request):
    assert aiida_profile_clean.is_test_profile
    old = old_class()
    full = make_cube(old)
    full.write_cube_file(tmp_path / "ChargeDiff.cube", low_precision=False)
    full = old.from_file(tmp_path / "ChargeDiff.cube")
    full.reduce_data_density_slicing()
    out = tmp_path / "out_cubes"
    out.mkdir()
    full.write_cube_file(out / "reduced_ChargeDiff.cube", low_precision=True)
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/false",
        default_calc_job_plugin="nanotech_empa.cubehandler",
    ).store()
    source = orm.StructureData(ase=make_cube().ase_atoms).store()
    root_type = "aiida.workflows:nanotech_empa.cp2k.fragment_separation"
    helper_type = "aiida.calculations:nanotech_empa.cubehandler"
    if request.param == "class-path":
        # This is the serialization used by the real installed factories, not
        # a hand-picked plugin string that can conceal an incompatible guard.
        root_type = plugins.WorkflowFactory(
            "nanotech_empa.cp2k.fragment_separation"
        ).build_process_type()
        helper_type = plugins.CalculationFactory(
            "nanotech_empa.cubehandler"
        ).build_process_type()
    root = orm.WorkChainNode(process_type=root_type)
    root.base.links.add_incoming(
        source, link_type=LinkType.INPUT_WORK, link_label="structure"
    )
    for label in ("all", "sheet", "molecule"):
        root.base.links.add_incoming(
            orm.Dict({"indices": [0, 1]}).store(),
            link_type=LinkType.INPUT_WORK,
            link_label="fragments__" + label,
        )
    root.store()
    native = orm.CalcJobNode(
        computer=aiida_localhost, process_type="aiida.calculations:cp2k"
    )
    native_parameters = orm.Dict(
        {
            "FORCE_EVAL": {
                "SUBSYS": {"CELL": {"PERIODIC": "XYZ"}},
                "DFT": {"POISSON": {"PERIODIC": "XYZ", "POISSON_SOLVER": "PERIODIC"}},
            }
        }
    ).store()
    native.base.links.add_incoming(
        native_parameters, link_type=LinkType.INPUT_CALC, link_label="parameters"
    )
    native.store()
    source_remote = orm.RemoteData(
        computer=aiida_localhost, remote_path=str(tmp_path / "scf-source")
    )
    source_remote.base.links.add_incoming(
        native, link_type=LinkType.CREATE, link_label="remote_folder"
    )
    source_remote.store()
    energies = orm.Dict(
        {
            "unrelaxed_separation_energy": -0.00051,
            "separation_energy": -0.00049,
            "sheet": {"unrelaxed": -5.0, "relaxed": -5.0001},
            "molecule": {"unrelaxed": -2.0, "relaxed": -2.0001},
        }
    )
    energies.base.links.add_incoming(
        native, link_type=LinkType.CREATE, link_label="energies"
    )
    energies.store()
    energies.base.links.add_incoming(
        root, link_type=LinkType.RETURN, link_label="energies"
    )
    params = orm.Dict(
        {
            "steps": [
                {
                    "command": "shrink",
                    "args": ["ChargeDiff.cube"],
                    "options": {"output_dir": "out_cubes", "low_precision": True},
                }
            ]
        }
    ).store()
    node = orm.CalcJobNode(
        computer=aiida_localhost,
        process_type=helper_type,
    )
    node.base.attributes.set("remote_workdir", str(tmp_path))
    node.base.links.add_incoming(code, link_type=LinkType.INPUT_CALC, link_label="code")
    node.base.links.add_incoming(
        params, link_type=LinkType.INPUT_CALC, link_label="parameters"
    )
    node.base.links.add_incoming(
        source_remote,
        link_type=LinkType.INPUT_CALC,
        link_label="parent_folders__folder1",
    )
    node.base.links.add_incoming(root, link_type=LinkType.CALL_CALC, link_label="cube")
    node.store()
    folder = orm.FolderData()
    folder.base.repository.put_object_from_tree(out, path="out_cubes")
    folder.base.links.add_incoming(
        node, link_type=LinkType.CREATE, link_label="retrieved"
    )
    folder.store()
    remote = orm.RemoteData(computer=aiida_localhost, remote_path=str(tmp_path))
    remote.base.links.add_incoming(
        node, link_type=LinkType.CREATE, link_label="remote_folder"
    )
    remote.store()
    for process in (node, root, native):
        process.set_process_state(ProcessState.FINISHED)
        process.set_exit_status(0)
        process.seal()
    return node.uuid, tmp_path


def test_real_widget_load_reconstructs_qualified_legacy_preview_and_preserves_records(
    retained, monkeypatch
):
    uuid, path = retained
    m = viewer_module()
    node = orm.load_node(uuid)
    before = node.outputs.retrieved.get_object_content(
        "out_cubes/reduced_ChargeDiff.cube"
    )
    raw_hash = hashlib.sha256((path / "ChargeDiff.cube").read_bytes()).hexdigest()
    # Real widget selection/loader and 2D link. Only browser-only rendering is intercepted.
    monkeypatch.setattr(m.CubeArrayData3dViewerWidget, "update_plot", lambda self: None)
    monkeypatch.setattr(m.CubePlaneCut2D, "plot_slice", lambda self: None)
    widget = m.HandleCubeFiles()
    widget.node_pk = node.pk
    widget.cube_selector.value = "reduced_ChargeDiff.cube"
    fixed = widget._viewer.cube
    assert fixed is not None and "Derived preview" in widget.error_message.value
    assert fixed.data.shape == (5, 5, 4)
    np.testing.assert_allclose(
        fixed.ase_atoms.cell, make_cube().ase_atoms.cell, atol=2e-5, rtol=0
    )
    assert "2.00, 2.00, 2.00" in widget.slice2d.info.value
    assert widget.retrieved_uuid == node.outputs.retrieved.uuid and not hasattr(
        widget, "node"
    )
    assert (
        node.outputs.retrieved.get_object_content("out_cubes/reduced_ChargeDiff.cube")
        == before
    )
    assert (
        hashlib.sha256((path / "ChargeDiff.cube").read_bytes()).hexdigest() == raw_hash
    )
    assert (
        orm.QueryBuilder().append(orm.ProcessNode).count() == 3
    )  # synthetic fixtures only
    widget.node_pk = None
    assert widget._shared_viewer is None and not widget._cube_cache
    widget.node_pk = node.pk
    widget.cube_selector.value = "reduced_ChargeDiff.cube"
    assert "Derived preview" in widget.error_message.value


def test_missing_full_source_is_explicitly_unverified(retained):
    uuid, path = retained
    (path / "ChargeDiff.cube").rename(path / "kept_original.cube")
    c, message = viewer_module()._load_cube_preview(uuid, "reduced_ChargeDiff.cube")
    assert c.data.shape == (5, 4, 4) and "unverified" in message


def test_changed_full_values_refuse_reconstruction(retained):
    uuid, path = retained
    c = Cube.from_file(path / "ChargeDiff.cube")
    c.data += 0.1
    c.write_cube_file(path / "ChargeDiff.cube")
    with pytest.raises(ValueError, match="values do not match"):
        viewer_module()._load_cube_preview(uuid, "reduced_ChargeDiff.cube")


def test_new_aligned_preview_passes_without_reconstruction(retained):
    uuid, path = retained
    c = Cube.from_file(path / "ChargeDiff.cube")
    c.reduce_data_density_slicing()
    c.write_cube_file(path / "new.cube", low_precision=True)
    # New separate immutable result fixture; do not rewrite the retained FolderData.
    node = orm.load_node(uuid)
    folder = orm.FolderData()
    folder.base.repository.put_object_from_file(
        path / "new.cube", "out_cubes/reduced_ChargeDiff.cube"
    )
    folder.store()
    m = viewer_module()
    from types import SimpleNamespace

    proxy = SimpleNamespace(
        outputs=SimpleNamespace(retrieved=folder),
        inputs=node.inputs,
        computer=node.computer,
        get_remote_workdir=node.get_remote_workdir,
    )
    from unittest.mock import patch

    with patch.object(m.orm, "load_node", lambda identifier: proxy):
        got, message = m._load_cube_preview(uuid, "reduced_ChargeDiff.cube")
    assert "Cell and sampling match" in message and got.data.shape == (5, 5, 4)


def test_cartesian_slice_origin_and_oblique_metric(monkeypatch):
    m = viewer_module()
    monkeypatch.setattr(m.CubePlaneCut2D, "plot_slice", lambda self: None)
    c = make_cube(shape=(7, 9, 11))
    widget = m.CubePlaneCut2D()
    widget.cube = c
    indexes = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    positions = c.origin / ANG_TO_BOHR + indexes @ (
        c.cell / ANG_TO_BOHR / np.array(c.data.shape)[:, None]
    )
    np.testing.assert_allclose(
        widget._cart_to_indexcoords(positions), indexes, atol=1e-13
    )


def test_ngl_serialization_preserves_nonzero_origin(monkeypatch):
    from types import SimpleNamespace

    m = viewer_module()
    captured = []

    def add(self, source, **kwargs):
        if isinstance(source, str):
            captured.append(Cube.from_file(source))
        return SimpleNamespace(
            id="test-component",
            clear=lambda: None,
            add_surface=lambda **kw: None,
            add_ball_and_stick=lambda **kw: None,
        )

    from aiidalab_widgets_empa import cube as shared

    monkeypatch.setattr(shared.nglview.NGLWidget, "add_component", add)
    widget = m.CubeArrayData3dViewerWidget()
    c = make_cube()
    widget.cube = c
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0].origin, c.origin, atol=1e-6, rtol=0)
    np.testing.assert_allclose(captured[0].cell, c.cell, atol=2e-5, rtol=0)


def test_full_origin_mismatch_refuses_reconstruction(retained):
    uuid, path = retained
    c = Cube.from_file(path / "ChargeDiff.cube")
    c.origin += 1
    c.write_cube_file(path / "ChargeDiff.cube")
    with pytest.raises(ValueError, match="origins differ"):
        viewer_module()._load_cube_preview(uuid, "reduced_ChargeDiff.cube")
