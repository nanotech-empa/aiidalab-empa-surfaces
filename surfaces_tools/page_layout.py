"""Keep notebook and widget outputs in the normal page flow."""

from IPython.display import HTML, display


def apply_page_layout():
    """Override saved/automatic output scrolling without changing input controls.

    Display this in its own uncaptured cell before the app creates widgets.
    CSS also covers outputs that grow after asynchronous widget updates.
    """
    display(
        HTML("""
<style id="surfaces-page-layout">
.output_scroll,
.widget-output,
.jp-OutputArea-output {
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
    box-shadow: none !important;
}
</style>
""")
    )
