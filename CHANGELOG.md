# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-26

### Added

- Initial `pytest-paraflow` release.
- Deterministic test sharding by stable hash of test identity.
- Group-based sharding via configurable pytest markers.
- Dynamic shard count calculation from target shard size.
- Pytest plugin CLI options and environment-based defaults.
- CI matrix workflow examples for shard-based test execution.
