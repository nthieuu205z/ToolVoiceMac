"""Run the real vanilla controller under a deterministic DOM fixture."""
import subprocess
from html.parser import HTMLParser
from pathlib import Path


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = {}
        self.scripts = []
        self.stack = []
        self.ancestors = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if 'id' in values:
            self.elements[values['id']] = (tag, values)
            self.ancestors[values['id']] = list(self.stack)
        if tag == 'script':
            self.scripts.append(values.get('src', '').split('?')[0])
        if tag not in {'input', 'link', 'meta', 'img', 'br', 'hr'}:
            self.stack.append(values.get('id', tag))

    def handle_endtag(self, tag):
        if self.stack:
            self.stack.pop()


def test_panel_is_collapsed_and_only_inside_text_form():
    page = Page()
    page.feed(Path('web/static/index.html').read_text())
    assert 'omnivoiceSettings' in page.elements, 'Advanced accordion missing'
    tag, attrs = page.elements['omnivoiceSettings']
    assert tag == 'details' and 'open' not in attrs and 'hidden' in attrs
    assert 'textJobForm' in page.ancestors['omnivoiceSettings']
    assert 'uploadForm' not in page.ancestors['omnivoiceSettings']
    assert page.scripts.index('omnivoice-settings.js') < page.scripts.index('app.js')


def test_controller_and_submission_behavior():
    result = subprocess.run(['node', 'tests/omnivoice_settings_runtime.cjs'], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
