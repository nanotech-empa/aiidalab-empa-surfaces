import ipywidgets as ipw
from aiida.orm import load_node
from IPython.display import clear_output


class ObsoleteWidget(ipw.VBox):
    def __init__(self, workchain=None):

        if not workchain:
            return
        self.node = load_node(workchain)

        self.output = ipw.Output()
        btn_mark_as_obsolete = ipw.Button(
            description="Mark as obsolete", button_style="danger"
        )
        app = ipw.VBox(children=[btn_mark_as_obsolete, self.output])

        btn_mark_as_obsolete.on_click(self.on_obsolete_click)

        super().__init__([app])

    def on_obsolete_click(self, _=None):
        self.node.set_extra("obsolete", True)
        with self.output:
            clear_output()
            print("Node {self.node.pk} will not be listed in search")
