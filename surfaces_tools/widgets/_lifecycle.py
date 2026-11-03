"""Cleanup for widget trees whose children, layouts and styles we own."""


def close_owned_widgets(*widgets, shared=()):
    """Close private widget trees, excluding explicitly borrowed objects.

    Only use this for trees constructed by the caller, not arbitrary widgets
    supplied by another view. Closing a Box alone does not close its children.
    """
    seen = {id(widget) for widget in shared}

    def close(widget):
        if widget is None or id(widget) in seen:
            return
        seen.add(id(widget))
        for child in getattr(widget, "children", ()):
            close(child)
        if hasattr(widget, "children"):
            widget.children = ()
        if hasattr(widget, "outputs"):
            widget.outputs = ()
        close(getattr(widget, "layout", None))
        close(getattr(widget, "style", None))
        widget.close()

    for widget in widgets:
        close(widget)
