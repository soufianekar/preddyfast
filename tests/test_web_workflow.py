"""The Pages workflow must stay manual: it publishes the site publicly."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "web.yml"


def test_web_workflow_is_manual_only() -> None:
    doc = yaml.safe_load(WORKFLOW.read_text())
    triggers = doc.get("on", doc.get(True))  # PyYAML parses the key `on` as True
    assert set(triggers) == {"workflow_dispatch"}
