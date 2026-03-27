# Configs

This directory is the canonical repo-level home for checked-in configuration
files and configuration inventories.

Use `configs/` for:

- repo-owned runtime configuration templates
- benchmark or harness configuration profiles
- tool configuration that is specific to this repository
- documented environment/profile definitions that should be versioned and
  reviewed like code

Each committed config should document, either in the file itself or in an
adjacent note:

- purpose
- owning subsystem or tool
- expected execution surface
- required inputs or environment variables
- whether the config is CPU-only, GPU-portable, or backend-specific

Policy lives in:

- [configuration_standard.md](/docs/standards/configuration_standard.md)
- [repo_standards.md](/docs/standards/repo_standards.md)

Current examples:

- [cpu_validation_profiles.json](/configs/cpu_validation_profiles.json)
- [platform_bootstrap_profiles.json](/configs/platform_bootstrap_profiles.json)
- [optional_comparison_backends.json](/configs/optional_comparison_backends.json)

The platform bootstrap profile inventory is the checked-in companion to:

- [requirements-colab.txt](/requirements-colab.txt)
- [colab_bootstrap.sh](/tools/colab_bootstrap.sh)
