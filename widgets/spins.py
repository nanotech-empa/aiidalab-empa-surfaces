import ipywidgets as ipw


class OneSpinSet(ipw.HBox):
    def __init__(self):

        self.selection = ipw.Text(
            description="Atoms",
            style={"description_width": "initial"},
        )
        self.starting_magnetization = ipw.IntText(
            description="Magnetization value",
            style={"description_width": "initial"},
        )

        super().__init__([self.selection, self.starting_magnetization])


class SpinsWidget(ipw.VBox):
    # details = trt.Dict()

    def __init__(self):
        self.spinsets = ipw.VBox()

        # Add spinset button.
        self.add_spinset_button = ipw.Button(
            description="Add spinset",
            layout={"width": "initial"},
            button_style="success",
        )
        self.add_spinset_button.on_click(self.add_spinset)

        # Remove spinset button.
        self.remove_spinset_button = ipw.Button(
            description="Remove spinset",
            layout={"width": "initial"},
            button_style="danger",
        )
        self.remove_spinset_button.on_click(self.remove_spinset)

        super().__init__(
            [
                self.spinsets,
                ipw.HBox(
                    [
                        self.add_spinset_button,
                        self.remove_spinset_button,
                    ]
                ),
            ]
        )

    def add_spinset(self, b=None):
        self.spinsets.children += (OneSpinSet(),)

    def remove_spinset(self, b=None):
        self.spinsets.children = self.spinsets.children[:-1]
