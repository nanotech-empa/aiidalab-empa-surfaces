from html import escape


def download_link(filename):
    href = f"/files/apps/surfaces/tmp/{escape(filename)}"
    return f'<a href="{href}" download="{escape(filename)}">download zip</a>'
