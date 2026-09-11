import subprocess
from pathlib import Path
from html.parser import HTMLParser


def test_workspace_controller():
    result = subprocess.run(['node', 'tests/workspace_features_runtime.cjs'], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_settings_separate_from_dashboard_and_named_composers():
    class Page(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.parents = {}
            self.attrs = {}

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if 'id' in values:
                self.parents[values['id']] = list(self.stack)
                self.attrs[values['id']] = values
            if tag not in {'input', 'meta', 'link', 'br', 'img'}:
                self.stack.append(values.get('id', tag))

        def handle_endtag(self, tag):
            if self.stack:
                self.stack.pop()

    page = Page()
    page.feed(Path('web/static/index.html').read_text())
    assert 'dashboardView' not in page.parents['settings']
    assert 'dashboardView' in page.parents['voiceSettings']
    assert 'textJobForm' in page.parents['textJobName']
    assert 'uploadForm' in page.parents['videoJobName']
    assert page.attrs['textInput']['maxlength'] == '200000'
