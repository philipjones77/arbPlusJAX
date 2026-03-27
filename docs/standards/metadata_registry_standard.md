Last updated: 2026-03-26T00:00:00Z

# Metadata Registry Standard

## Purpose
Public metadata and function-capability reporting must not require importing
runtime implementation modules during package or API startup.

This is the canonical metadata-generation and metadata-runtime boundary
standard.

## Requirements

1. Runtime metadata must come from static descriptors.
   Acceptable sources:
   - checked-in JSON registry artifacts
   - checked-in Python descriptor tables
   - generated static data files loaded without importing implementation modules

2. Metadata lookup must not build registries from live callables during startup.
   Disallowed patterns:
   - importing broad implementation modules only to inspect signatures
   - importing provider modules only to infer metadata fields
   - calling dynamic public-registry builders as part of metadata load

3. Metadata code must remain import-light.
   Importing `arbplusjax.public_metadata` and loading the runtime metadata
   registry must stay off:
   - `arbplusjax.api`
   - point wrapper modules
   - interval/mode wrapper modules
   - provider/alternative backend modules

4. Dynamic builders are allowed only for tooling and tests.
   `build_public_metadata_registry(...)` may still exist for:
   - unit tests
   - regeneration tooling
   - schema validation
   But it must not be the default runtime path used by public metadata queries.

5. Static registry refresh must be explicit.
   Any workflow that changes public metadata shape or inventory must regenerate
   the checked-in static metadata artifact as part of the change.
   Canonical command:
   `python tools/generate_public_metadata_registry.py`

6. Metadata generation must not be a hidden startup side effect.
   Disallowed patterns:
   - generating metadata files during package import
   - rebuilding registries during `api` import
   - implicit regeneration in public runtime calls

7. Metadata descriptors must be stable inputs, not discovered startup state.
   Runtime metadata must describe the public surface from checked-in static
   descriptors, not from whatever implementation modules happen to be imported
   in the current process.

## Current Runtime Path

- Runtime loader: [`public_metadata.py`](/home/phili/projects/arbplusJAX/src/arbplusjax/public_metadata.py)
- Static artifact: [`public_metadata_registry.json`](/home/phili/projects/arbplusJAX/src/arbplusjax/public_metadata_registry.json)
- Public API consumer: [`api.py`](/home/phili/projects/arbplusJAX/src/arbplusjax/api.py)

## Enforcement

The repo must keep tests that verify:

- loading static public metadata does not import runtime implementation modules
- `api.get_public_function_metadata(...)` reads from the static registry path
- startup import boundaries are not widened by metadata/reporting code

Required tooling path:

- [generate_public_metadata_registry.py](/home/phili/projects/arbplusJAX/tools/generate_public_metadata_registry.py)

Required regression tests include:

- [test_public_metadata_contracts.py](/home/phili/projects/arbplusJAX/tests/test_public_metadata_contracts.py)
- [test_api_metadata.py](/home/phili/projects/arbplusJAX/tests/test_api_metadata.py)
