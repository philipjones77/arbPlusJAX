from tools.core_status_report import _status_rows
from tools.custom_core_report import CUSTOM_FUNCTIONS


def test_custom_core_functions_have_specialized_rigorous_status():
    status_map = {(row.name, row.module): row for row in _status_rows()}
    missing = [
        f"{meta.module}:{meta.name}"
        for meta in CUSTOM_FUNCTIONS
        if not status_map[(meta.name, meta.module)].rigorous_specialized
    ]
    assert missing == []
