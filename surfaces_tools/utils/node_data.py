"""Read mapping data from current Dict nodes and legacy generic Data nodes."""

from collections.abc import Mapping


def mapping_from_node(value):
    if isinstance(value, Mapping):
        return dict(value)
    get_dict = getattr(value, "get_dict", None)
    if callable(get_dict):
        return get_dict()
    return dict(value.base.attributes.all)
