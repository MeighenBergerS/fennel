# Commit Summary: v2.0 API with Test Fixes

## Overview
This commit introduces the Fennel v2.0 API with result containers, comprehensive validation, and updated tests. All 117 tests now pass (6 skipped for JAX when not installed).

## Changes Summary

### Core Features Added
1. **v2 API with Result Containers** ([fennel/results.py](fennel/results.py))
   - `TrackYieldResult` - structured muon track results
   - `EMYieldResult` - electromagnetic cascade results
   - `HadronYieldResult` - hadronic cascade results
   - Pretty-print representations
   - Named attributes (no more tuple unpacking!)

2. **Input Validation** ([fennel/validation.py](fennel/validation.py))
   - `ValidationError` exception class
   - Comprehensive input checking with helpful error messages
   - Validates: energy, wavelengths, angles, particle PDG codes, etc.

3. **New API Methods** ([fennel/fennel.py](fennel/fennel.py))
   - `track_yields_v2()` - enhanced track API
   - `em_yields_v2()` - enhanced EM cascade API
   - `hadron_yields_v2()` - enhanced hadron cascade API
   - `calculate()` - universal method, auto-detects particle type
   - `quick_track()` - convenience method for simple cases
   - `quick_cascade()` - convenience method for cascades

4. **Full Backward Compatibility**
   - All v1.x methods still work identically
   - No breaking changes
   - Migration is optional

### Test Fixes
Fixed 8 failing tests that were using outdated internal APIs:

#### [tests/test_em_cascades.py](tests/test_em_cascades.py)
- ✅ `test_longitudinal_profile` - now uses public `em_yields()` API
- ✅ `test_longitudinal_profile_peak` - now uses public API  
- ✅ `test_shower_max_energy_dependence` - now uses public API

#### [tests/test_hadron_cascades.py](tests/test_hadron_cascades.py)
- ✅ `test_longitudinal_profile` - now uses public `hadron_yields()` API
- ✅ `test_angle_distribution` - now uses public API
- ✅ `test_shower_max_exists` - now uses public API

#### [tests/test_integration.py](tests/test_integration.py)
- ✅ `test_em_yields_electron` - fixed shape assertion (squeeze profile)
- ✅ `test_hadron_yields_pion` - fixed shape assertion (squeeze profile)

**Issue**: Tests were calling private methods (`log_profile_func`, `cherenkov_angle_distro`) that don't exist in the public API, and expecting wrong shapes for longitudinal profiles.

**Fix**: Updated to use public Fennel API methods and handle correct profile shapes `(1, N)` → `(N,)`.

### Documentation Added
- [UPGRADE_GUIDE_V2.md](UPGRADE_GUIDE_V2.md) - Migration guide from v1 to v2
- [notebooks/example_v2.ipynb](notebooks/example_v2.ipynb) - v2 API examples
- [CHANGELOG.md](CHANGELOG.md) - Project changelog
- [COMMIT_GUIDE.md](COMMIT_GUIDE.md) - Commit best practices
- Updated docstrings with examples and type hints

### Testing Infrastructure
- [pytest.ini](pytest.ini) - Pytest configuration
- [.pre-commit-config.yaml](.pre-commit-config.yaml) - Pre-commit hooks
- [.github/workflows/test.yml](.github/workflows/test.yml) - CI/CD
- Comprehensive test suite (117 tests total)

### Build & Tooling
- [pyproject.toml](pyproject.toml) - Modern Python packaging
- [mkdocs.yml](mkdocs.yml) - Documentation site config
- [Makefile](Makefile) - Common development tasks
- Updated [README.md](README.md) with v2 examples

## Test Results

```
$ pytest -q
117 passed, 6 skipped, 1 warning in 0.24s
```

All tests pass! Skipped tests are JAX-specific (JAX optional dependency).

## File Changes

### Modified Files (14)
- `.gitignore` - ignore patterns
- `README.md` - updated examples
- `fennel/__init__.py` - exports v2 classes
- `fennel/__main__.py` - CLI updates
- `fennel/config.py` - configuration
- `fennel/definition_generator.py` - definitions
- `fennel/em_cascades.py` - EM cascade logic
- `fennel/fennel.py` - main API (added v2 methods)
- `fennel/hadron_cascades.py` - hadron cascade logic  
- `fennel/particle.py` - particle definitions
- `fennel/photons.py` - photon calculations
- `fennel/tracks.py` - track calculations
- `notebooks/example.ipynb` - original example
- `notebooks/example_jax.ipynb` - JAX example

### New Files (22+)
- `fennel/results.py` ⭐ - Result container classes
- `fennel/validation.py` ⭐ - Input validation
- `notebooks/example_v2.ipynb` ⭐ - v2 API demo
- `UPGRADE_GUIDE_V2.md` ⭐ - Migration guide
- `CHANGELOG.md` - Project changelog
- `COMMIT_GUIDE.md` - Commit practices
- `.pre-commit-config.yaml` - Pre-commit hooks
- `pytest.ini` - Test configuration
- `pyproject.toml` - Package config
- `mkdocs.yml` - Docs config
- `Makefile` - Build tasks
- `tests/` directory - Full test suite
  - `test_config.py`
  - `test_em_cascades.py` ⭐ (fixed)
  - `test_hadron_cascades.py` ⭐ (fixed)
  - `test_integration.py` ⭐ (fixed)
  - `test_physics_regression.py`
  - `test_tracks.py`
  - `test_v2_api.py` ⭐ (new v2 tests)
  - `conftest.py`
  - `reference_values_v1.3.4.json`
- `.github/workflows/test.yml` - CI
- `docs-mkdocs/` - Documentation source
- `scripts/` - Utility scripts
- `site/` - Built documentation

⭐ = Most important files

## Commit Command

### Option 1: Commit Everything
```bash
git add .
git commit -m "feat: add v2 API with result containers and fix tests

- Add TrackYieldResult, EMYieldResult, HadronYieldResult classes
- Implement comprehensive input validation with ValidationError  
- Add convenience methods: calculate(), quick_track(), quick_cascade()
- Maintain 100% backward compatibility with v1.x API
- Fix 8 failing tests to use public API instead of private methods
- Fix longitudinal profile shape assertions in integration tests
- Add complete test suite with 117 passing tests
- Add v2 API example notebook and upgrade guide
- Add pre-commit hooks, CI/CD, and documentation infrastructure

All tests pass: 117 passed, 6 skipped"
```

### Option 2: Commit in Stages (Recommended)

```bash
# Stage 1: Core v2 API
git add fennel/results.py fennel/validation.py
git add fennel/fennel.py fennel/__init__.py
git commit -m "feat(api): add v2 result containers and validation"

# Stage 2: Test fixes
git add tests/
git commit -m "fix(tests): update tests to use public API and correct shapes"

# Stage 3: Documentation
git add UPGRADE_GUIDE_V2.md notebooks/example_v2.ipynb
git add CHANGELOG.md COMMIT_GUIDE.md README.md
git commit -m "docs: add v2 API guides and examples"

# Stage 4: Infrastructure
git add .pre-commit-config.yaml .github/ pytest.ini pyproject.toml mkdocs.yml Makefile
git commit -m "chore: add testing and build infrastructure"
```

## Pre-Commit Setup (Optional but Recommended)

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run on all files (optional)
pre-commit run --all-files
```

This will automatically:
- Format code with Black
- Sort imports with isort
- Lint with flake8
- Run tests before each commit

## Next Steps

1. **Review changes**: `git diff --staged`
2. **Commit**: Use one of the commands above
3. **Push**: `git push origin smb/version_2_0`
4. **Create PR**: On GitHub, create pull request to master
5. **Tag release**: After merge, tag v2.0.0

## Notes

- Branch: `smb/version_2_0`
- All 117 tests passing
- No breaking changes
- Full backward compatibility maintained
- Ready for code review and merge

## Questions?

See [COMMIT_GUIDE.md](COMMIT_GUIDE.md) for detailed commit practices and workflows.
