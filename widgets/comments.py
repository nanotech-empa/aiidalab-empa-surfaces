import ipywidgets as ipw
from IPython.display import display, clear_output

from aiida.orm import  load_node

class CommentsWidget(ipw.VBox):
    def __init__(self,workchain=None):
        
        if not workchain:
            return
        self.node=load_node(workchain)

        self.old_comments=ipw.Output()
        btn_add_comment = ipw.Button(description="Add comment")
        self.new_comment=ipw.Textarea(value='', disabled=False, layout={'width': '60%'})
        app = ipw.VBox(children=[self.old_comments, self.new_comment,btn_add_comment])
        
        with self.old_comments:
            clear_output()
            for comment in self.node.get_comments():
                print(comment.ctime.strftime("%Y-%m-%d %H:%M"))
                print(comment.content)
                
        btn_add_comment.on_click(self.on_add_click)       
        
        #self.search()
        super(CommentsWidget, self).__init__([app])
        
    def on_add_click(self,_=None):
        self.node.add_comment(self.new_comment.value)
        self.new_comment.value=''
        with self.old_comments:
            clear_output()
            for comment in self.node.get_comments():
                print(comment.ctime.strftime("%Y-%m-%d %H:%M"),comment.content)