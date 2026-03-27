from __future__ import annotations

import json
from pathlib import Path

from arbplusjax import api
from arbplusjax.public_metadata import metadata_to_record


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "src" / "arbplusjax" / "public_metadata_registry.json"


def render() -> str:
    registry = api._build_live_public_metadata_registry()
    rows = [metadata_to_record(entry) for entry in sorted(registry.values(), key=lambda entry: entry.name)]
    return json.dumps(rows, indent=2, sort_keys=True) + "\n"


def main() -> None:
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"Wrote: {OUTPUT}")


if __name__ == "__main__":
    main()
