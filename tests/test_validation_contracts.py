from arbplusjax import validation


def test_parity_enabled_prefers_uppercase_env(monkeypatch):
    monkeypatch.delenv("ARBPLUSJAX_RUN_PARITY", raising=False)
    monkeypatch.delenv("arbplusjax_RUN_PARITY", raising=False)
    assert validation.parity_enabled() is False

    monkeypatch.setenv("arbplusjax_RUN_PARITY", "1")
    assert validation.parity_enabled() is True

    monkeypatch.setenv("ARBPLUSJAX_RUN_PARITY", "0")
    assert validation.parity_enabled() is False

    monkeypatch.setenv("ARBPLUSJAX_RUN_PARITY", "1")
    assert validation.parity_enabled() is True
