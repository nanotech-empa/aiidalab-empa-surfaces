import ipywidgets as ipw
import traitlets as trt
from .ANALYZE_structure import mol_ids_range


class OneConstraint(ipw.HBox):
    def __init__(self):

        self.constraint_widget = ipw.Text(
            description="Constraint",
            style={"description_width": "initial"},
        )

        super().__init__([self.constraint_widget])

class ConstraintsWidget(ipw.VBox):
    details = trt.Dict()

    def __init__(self):
        self.constraints = ipw.VBox()
        
        # Add constraint button.
        self.add_constraint_button = ipw.Button(
            description="Add constraint",
            layout={"width": "initial"},
            button_style="success",
        )
        self.add_constraint_button.on_click(self.add_constraint)

        # Remove constraint button.
        self.remove_constraint_button = ipw.Button(
            description="Remove constraint",
            layout={"width": "initial"},
            button_style="danger"
        )
        self.remove_constraint_button.on_click(self.remove_constraint)

        super().__init__(
            [
                self.constraints,
                ipw.HBox([self.add_constraint_button,
                self.remove_constraint_button,])
            ]
        )
    @trt.observe("details")
    def _observe_manager(self, _=None):
        if self.details and "Slab" in self.details["system_type"]:
            self.add_constraint()
    
    def add_constraint(self, b=None):
        self.constraints.children += (OneConstraint(),)
        if len(self.constraints.children) == 1 and self.details and "Slab" in self.details["system_type"]:
            to_fix = [
                i
                for i in self.details["bottom_H"]
                + self.details["slab_layers"][0]
                + self.details["slab_layers"][1]
            ]            
            self.constraints.children[0].constraint_widget.value = 'fixed atoms ' + mol_ids_range(to_fix)

    def remove_constraint(self, b=None):
        self.constraints.children = self.constraints.children[:-1]

    def return_dict(self):
        return {"constraints": self.constraints.value}
    
    def traits_to_link(self):
        return ["details"]