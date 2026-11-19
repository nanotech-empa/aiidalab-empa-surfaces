import ipywidgets as ipw
from aiida.orm import load_node
from IPython.display import clear_output


class ObsoleteWidget(ipw.VBox):
    def __init__(self, workchain=None):
        if not workchain:
            super().__init__()
            return
        self.node_uuid = str(load_node(workchain).uuid)

        self.output = ipw.Output()
        btn_mark_as_obsolete = ipw.Button(
            description="Mark as obsolete", button_style="danger"
        )
        app = ipw.VBox(children=[btn_mark_as_obsolete, self.output])

        btn_mark_as_obsolete.on_click(self.on_obsolete_click)

        super().__init__([app])

    def on_obsolete_click(self, _=None):
        node = load_node(self.node_uuid)
        node.base.extras.set("obsolete", True)
        with self.output:
            clear_output()
            print(f"Node {node.pk} will not be listed in search")
