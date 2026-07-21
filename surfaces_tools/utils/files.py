from html import escape


def download_link(filename):
    # Absolute path served by the Jupyter file server; assumes the app is
    # installed as "apps/surfaces". A relative href breaks inside AiiDAlab.
    href = f"/files/apps/surfaces/tmp/{escape(filename)}"
    return f'<a href="{href}" download="{escape(filename)}">download zip</a>'
