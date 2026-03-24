import sys


def test_package_root_import_keeps_arbplusjax_eager_module_set_small() -> None:
    before = set(sys.modules)
    import arbplusjax  # noqa: F401

    after = set(sys.modules)
    loaded = {name for name in after - before if name.startswith("arbplusjax")}
    assert loaded <= {"arbplusjax", "arbplusjax.precision"}


def test_package_root_attribute_access_loads_requested_module_lazily() -> None:
    import arbplusjax

    sys.modules.pop("arbplusjax.api", None)
    arbplusjax.__dict__.pop("api", None)

    assert "arbplusjax.api" not in sys.modules
    api_mod = arbplusjax.api
    assert api_mod is sys.modules["arbplusjax.api"]
    assert arbplusjax.api is api_mod


def test_lazy_subpackage_exports_resolve_on_demand() -> None:
    import arbplusjax.special.gamma as gamma_pkg

    sys.modules.pop("arbplusjax.special.gamma.incomplete_gamma", None)
    gamma_pkg.__dict__.pop("incomplete_gamma_lower", None)

    assert "arbplusjax.special.gamma.incomplete_gamma" not in sys.modules
    fn = gamma_pkg.incomplete_gamma_lower
    assert callable(fn)
    assert "arbplusjax.special.gamma.incomplete_gamma" in sys.modules
