import pytest
from aiida import orm

from surfaces_tools.widgets.comments import CommentsWidget

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def test_comment_roundtrip(aiida_profile_clean, capsys):
    node = orm.Dict().store()
    widget = CommentsWidget(str(node.pk))
    assert widget.node_uuid == node.uuid
    assert not any(isinstance(value, orm.Node) for value in vars(widget).values())
    widget.new_comment.value = "Test comment"
    widget.on_add_click()
    widget.on_add_click()
    assert [comment.content for comment in node.base.comments.all()] == ["Test comment"]
    capsys.readouterr()
    CommentsWidget(node.uuid)
    assert "Test comment" in capsys.readouterr().out


@pytest.mark.parametrize("identifier", [None, "unknown", "99999999"])
def test_missing_or_invalid_node_has_message(aiida_profile_clean, identifier):
    widget = CommentsWidget(identifier)
    assert widget.node_uuid is None
    assert widget.children[0].value
