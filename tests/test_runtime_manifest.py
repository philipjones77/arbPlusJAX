from pathlib import Path

from tools.runtime_manifest import collect_runtime_manifest
from tools.runtime_manifest import write_runtime_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_collect_runtime_manifest_has_required_keys():
    manifest = collect_runtime_manifest(REPO_ROOT, jax_mode="cpu", python_path="/tmp/python")

    assert manifest["repo_root"] == str(REPO_ROOT)
    assert manifest["environment_kind"] in {"local", "wsl", "colab"}
    assert manifest["python"]["executable"] == "/tmp/python"
    assert manifest["jax"]["requested_mode"] == "cpu"
    assert "system" in manifest["os"]
    assert "commit" in manifest["git"]


def test_write_runtime_manifest_emits_json_file(tmp_path: Path):
    manifest = collect_runtime_manifest(REPO_ROOT, jax_mode="auto")
    out = write_runtime_manifest(tmp_path, manifest)

    assert out.name == "runtime_manifest.json"
    text = out.read_text(encoding="utf-8")
    assert '"repo_root"' in text
    assert '"environment_kind"' in text
