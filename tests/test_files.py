from surfaces_tools.utils.files import download_link


def test_builds_file_server_path():
    html = download_link("afm_42.zip")

    assert 'href="/files/apps/surfaces/tmp/afm_42.zip"' in html
    assert 'download="afm_42.zip"' in html


def test_escapes_special_chars():
    html = download_link('a"b&c.zip')

    # A raw " or & in the filename must not survive unescaped.
    assert "&amp;" in html
    assert "&quot;" in html
    assert '"a"b&c' not in html
