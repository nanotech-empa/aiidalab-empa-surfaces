import aiidalab_widgets_base as awb
import ipywidgets as ipw
import traitlets as tl


class HorizontalItemWidget(ipw.HBox):
    stack_class = None

    def __init__(self, *args, **kwargs):
        # Delete button.
        self.delete_button = ipw.Button(
            description="x", button_style="danger", layout={"width": "30px"}
        )
        self.delete_button.on_click(self.delete_myself)

        children = kwargs.pop("children", [])
        children.append(self.delete_button)

        super().__init__(children=children, *args, **kwargs)

    def delete_myself(self, _):
        self.stack_class.delete_item(self)


class VerticalStackWidget(ipw.VBox):
    items = tl.Tuple()
    item_class = None

    def __init__(self, item_class, add_button_text="Add"):
        self.item_class = item_class

        self.add_item_button = ipw.Button(
            description=add_button_text, button_style="info"
        )
        self.add_item_button.on_click(self.add_item)

        self.items_output = ipw.VBox()
        tl.link((self, "items"), (self.items_output, "children"))

        # Outputs.
        self.add_item_message = awb.utils.StatusHTML()
        super().__init__(
            children=[
                self.items_output,
                self.add_item_button,
                self.add_item_message,
            ]
        )

    def add_item(self, _):
        self.items += (self.item_class(),)

    @tl.observe("items")
    def _observe_fragments(self, change):
        """Update the list of fragments."""
        if change["new"]:
            self.items_output.children = change["new"]
            for item in change["new"]:
                item.stack_class = self
        else:
            self.items_output.children = []

    def delete_item(self, item):
        try:
            index = self.items.index(item)
        except ValueError:
            return
        self.items = self.items[:index] + self.items[index + 1 :]
        del item

    def length(self):
        return len(self.items)
