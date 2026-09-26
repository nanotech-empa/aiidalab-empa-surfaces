"""Repeated removal must release owned widgets without closing shared ones."""

import gc
import weakref

import ipywidgets as ipw
from ipywidgets.widgets.widget import _instances

from surfaces_tools.widgets.fragments import BOX_LAYOUT, Fragment, FragmentList
from surfaces_tools.widgets.series_plotter import SeriesPlotter


def test_series_rows_are_released():
    plotter = SeriesPlotter(lambda: [], "test")
    baseline = set(_instances)
    removed = []
    for _ in range(20):
        plotter.add_selection_row()
        removed.append(weakref.ref(plotter.selections_vbox.children[-1]))
        plotter.elem_list[-1][-1].click()
    gc.collect()
    assert not plotter.elem_list
    assert not plotter.selections_vbox.children
    assert all(ref() is None for ref in removed)
    assert set(_instances) == baseline
    plotter.close()
    plotter.close()


def test_clear_releases_plot_outputs():
    plotter = SeriesPlotter(lambda: [], "test")
    baseline = set(_instances)
    removed = []
    for _ in range(20):
        output = ipw.Output()
        output.append_stdout("example payload")
        removed.append(weakref.ref(output))
        plotter.plot_output.children += (ipw.Box([output]),)
        plotter.full_clear(None)
        assert output.outputs == ()
        del output
    gc.collect()
    assert all(ref() is None for ref in removed)
    assert set(_instances) == baseline
    plotter.close()


def test_fragments_link_once_and_release_on_removal():
    fragments = FragmentList()
    baseline = set(_instances)
    baseline_callbacks = len(
        fragments._trait_notifiers.get("uks", {}).get("change", [])
    )
    removed = []
    for index in range(10):
        fragment = Fragment(name=str(index))
        removed.append(weakref.ref(fragment))
        fragments.fragments = [*fragments.fragments, fragment]
        assert fragment.master_class is fragments
    assert len(fragments._trait_notifiers["uks"]["change"]) == baseline_callbacks + 10
    fragments.fragments = list(reversed(fragments.fragments))
    assert len(fragments._trait_notifiers["uks"]["change"]) == baseline_callbacks + 10
    fragments.uks = True
    assert all(item.uks for item in fragments.fragments)
    for fragment in list(fragments.fragments):
        fragments.delete_fragment(fragment)
        assert fragment.master_class is None
    del fragment
    gc.collect()
    assert all(ref() is None for ref in removed)
    assert len(fragments._trait_notifiers["uks"]["change"]) == baseline_callbacks
    assert set(_instances) == baseline
    fragments.close()
    fragments.close()
    assert BOX_LAYOUT.comm is not None


def test_uks_reuses_multiplicity_widget():
    fragment = Fragment()
    baseline = set(_instances)
    for _ in range(20):
        fragment.uks = not fragment.uks
    assert set(_instances) == baseline
    fragment.close()
