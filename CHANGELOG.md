# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.3.2] - 2026-03-05

### Added

- Improved GPU support for `loglikelihood`, including using robust losses with  additional  tests.

### Changed

- GPU execution path refactored to use an `oncpu` dispatch strategy for the appropriate `mapreduce` behavior.
- `Adapt` moved from weak dependency to regular dependency.
- Dependency and extension wiring around GPU `loglikelihood`/rrule support was revised.
- API documentation updated.

### Fixed

- Fixed `loglikelihood` rrule behavior for `ScaledL2Loss` on GPU.
- Fixed type issues in GPU robust `loglikelihood` rrules and cleaned unused `ChainRulesCore` import usage.

### Removed

- Removed `WeightedDataAcceleratedKernelsChainRulesCoreExt` and its dedicated test file.

## [0.3.1] - 2026-03-04

### Added

- Additional tests for `WeightedDataAdaptExt` to verify backend/storage preservation when adapting `WeightedArray` values and precisions.
- Expanded  documentation.

### Fixed

- Fixed type instability in `mean(::WeightedArray; dims=...)` caused by empty-array paths in `filterbaddata`, restoring `@inferred` stability.

## [0.3.0] - 2026-03-04

### Added

- `WeightedDataAdaptExt` support for adapting `WeightedArray` containers across backends via `Adapt.adapt`.
- `WeightedDataGPUArraysExt` support for generic `loglikelihood` evaluation on GPU-backed `WeightedArray` values (`AnyGPUArray`) and GPU-aware plain-text display.
- `WeightedDataRobustModelsGPUArraysExt` support for robust `loglikelihood` on GPU-backed `WeightedArray` values (`AnyGPUArray`) with `RobustModels.LossFunction`.

### Fixed

- `WeightedDataRobustModelsGPUArraysExt` now imports `AnyGPUArray` explicitly.
- `WeightedDataRobustModelsGPUArraysExt.loglikelihood` now uses an explicit `init` value in GPU `mapreduce` to avoid output-type inference errors.
- `WeightedDataRobustModelsGPUArraysExt.loglikelihood` now imports and calls `rho` directly from `RobustModels`, preventing `UndefVarError` during GPU execution.
- Public API and internals now consistently use `get_value` / `get_precision`; deprecated getter usage was removed from source, extensions, tests, and docs examples.
- `filterbaddata!` behavior was clarified and documented; shape validation now raises `DimensionMismatch` for invalid masks.

## [0.2.0] - 2026-02-26

### Breaking changes

- Weighted mean API migrated to `Statistics.mean` for `WeightedValue` and weighted arrays.
- Likelihood API migrated to `StatsAPI.loglikelihood` as the canonical entry point.

### Added

- `Statistics.var` support:
  - `var(x::WeightedValue) = 1 / get_precision(x)`
  - `var(x::AbstractArray{<:WeightedValue}) = 1 ./ get_precision(x)`
- `Statistics.std` support:
  - `std(x::WeightedValue) = sqrt(var(x))`
  - `std(x::AbstractArray{<:WeightedValue}) = sqrt.(var(x))`
- `StatsAPI` dependency and direct `loglikelihood` integration.
- Explicit import hygiene checks in CI via `ExplicitImports.jl`.

### Changed

- Documentation and README examples updated to `mean`/`var`/`std` and `loglikelihood`.
- `ChainRulesCore` extension updated to provide rrules for `loglikelihood`.
- Internal module imports standardized to explicit `import` style.

### Compatibility notes

- `likelihood(...)` is deprecated and forwards to `loglikelihood(...)` (with deprecation warning).
- Users are encouraged to migrate to `StatsAPI.loglikelihood`.

---

## Road to 1.0 checklist

Use this list before releasing `1.0.0`:

- [ ] No planned breaking API renames/removals in the next 1-2 minor cycles.
- [ ] Keep compatibility aliases (if any) for at least one minor release and mark deprecation timeline.
- [ ] Public API list explicitly documented (types/functions intended for external use).
- [ ] Docs include migration notes from pre-`0.2` APIs.
- [ ] CI matrix and Aqua checks stable for at least one release cycle.
- [ ] Extension behavior (`RobustModels`, `Measurements`, `Uncertain`, `OnlineSampleStatistics`, `ChainRulesCore`) validated end-to-end.
- [ ] Register and tag at least one `0.2.x` patch/minor with no emergency API rollback.
- [ ] Confirm semver policy in README (what counts as breaking after 1.0).
