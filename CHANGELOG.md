# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] - 2026-02-26

### Breaking changes
- Weighted mean API migrated to `Statistics.mean` for `WeightedValue` and weighted arrays.
- Likelihood API migrated to `StatsAPI.loglikelihood` as the canonical entry point.
- Bad-data masking API renamed from `flagbadpix`/`flagbadpix!` to `flagbaddata`/`flagbaddata!`.

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
