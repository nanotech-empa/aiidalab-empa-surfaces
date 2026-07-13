import unittest

from surfaces_tools.utils.files import download_link


class DownloadLinkTest(unittest.TestCase):
    def test_builds_file_server_path(self):
        html = download_link("afm_42.zip")

        self.assertIn('href="/files/apps/surfaces/tmp/afm_42.zip"', html)
        self.assertIn('download="afm_42.zip"', html)

    def test_escapes_special_chars(self):
        html = download_link('a"b&c.zip')

        # A raw " or & in the filename must not survive unescaped.
        self.assertIn("&amp;", html)
        self.assertIn("&quot;", html)
        self.assertNotIn('"a"b&c', html)
