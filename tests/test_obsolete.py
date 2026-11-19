from aiida import orm

from surfaces_tools.widgets.obsolete import ObsoleteWidget

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def test_obsolete_button_marks_selected_node(aiida_profile_clean, capsys):
    node = orm.Dict().store()
    widget = ObsoleteWidget(str(node.pk))
    assert widget.node_uuid == node.uuid
    assert not any(isinstance(value, orm.Node) for value in vars(widget).values())

    widget.on_obsolete_click()

    assert orm.load_node(node.uuid).base.extras.get("obsolete") is True
    assert f"Node {node.pk} " in capsys.readouterr().out


def test_empty_widget():
    assert ObsoleteWidget().children == ()
