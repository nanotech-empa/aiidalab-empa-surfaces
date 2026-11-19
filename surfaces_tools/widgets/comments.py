import ipywidgets as ipw
from aiida import orm
from aiida.common.exceptions import MultipleObjectsError, NotExistent
from IPython.display import clear_output


class CommentsWidget(ipw.VBox):
    def __init__(self, workchain=None):
        self.node_uuid = None
        if workchain is None or workchain == "":
            super().__init__([ipw.Label("Open a result to view its comments.")])
            return
        try:
            node = orm.load_node(workchain)
        except (ValueError, TypeError, NotExistent, MultipleObjectsError):
            super().__init__([ipw.Label("The requested AiiDA node could not be found.")])
            return
        self.node_uuid = str(node.uuid)
        self.old_comments = ipw.Output()
        btn_add_comment = ipw.Button(description="Add comment")
        self.new_comment = ipw.Textarea(layout={"width": "60%"})
        btn_add_comment.on_click(self.on_add_click)
        super().__init__([self.old_comments, self.new_comment, btn_add_comment])
        self._show_comments(node)

    def _show_comments(self, node):
        with self.old_comments:
            clear_output()
            for comment in node.base.comments.all():
                print(comment.ctime.strftime("%Y-%m-%d %H:%M"), comment.content)

    def on_add_click(self, _=None):
        if self.new_comment.value.strip():
            node = orm.load_node(self.node_uuid)
            node.base.comments.add(self.new_comment.value)
            self.new_comment.value = ""
            self._show_comments(node)
