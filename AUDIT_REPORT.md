# Repository Audit Report

**Date:** 2026-03-15  
**Repo:** `/Users/yuanhaocui/awesome-integrated-sensing-and-communications/`

---

## 1. Broken Internal Links

| File | Link | Issue |
|------|------|-------|
| `code/README.md` | `[CONTRIBUTING.md](../../CONTRIBUTING.md)` | Wrong relative path — should be `../CONTRIBUTING.md` (file exists at repo root) |
| `tools/README.md` | `[🔗](...)` (PyRadar row) | Placeholder link `...` is not a valid URL |

## 2. README Consistency

### Sections Checklist

| Baseline | Badges | What It Implements | Results | Quick Start | Math Background | Project Structure | Citation |
|----------|--------|--------------------|---------|-------------|-----------------|-------------------|----------|
| csi_ratio_doppler_estimation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| isac_capacity_distortion | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ (has References section) |
| isac_energy_efficient_beamforming | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| isac_resource_allocation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| ofdm_ambiguity_function | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| ris_isac_beamforming | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ (has References section) |
| xl_mimo_beam_training | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

### Baselines Without README.md

| Baseline | Has Source Code | Status |
|----------|----------------|--------|
| `crb_analysis` | ❌ (empty src/) | Stub — all directories empty |
| `mimo_beamforming` | ❌ (empty src/) | Stub — all directories empty |
| `ofdm_isac` | ❌ (empty src/) | Stub — no source files |
| `radar_comm_tradeoff` | ❌ (empty src/) | Stub — all directories empty |

**Recommendation:** Either remove these empty stubs or add placeholder READMEs with planned scope.

## 3. Test Status

| Baseline | Test Files | Import Check |
|----------|-----------|--------------|
| csi_ratio_doppler_estimation | 3 | ⚠️ Cannot `import` directly (pytest-managed) |
| isac_capacity_distortion | 4 | ⚠️ Cannot `import` directly (pytest-managed) |
| isac_energy_efficient_beamforming | 7 | ⚠️ Cannot `import` directly (pytest-managed) |
| isac_resource_allocation | 5 | ⚠️ Cannot `import` directly (pytest-managed) |
| ris_isac_beamforming | 6 | ⚠️ Cannot `import` directly (pytest-managed) |
| xl_mimo_beam_training | 5 | ⚠️ Cannot `import` directly (pytest-managed) |

**Note:** Tests are designed to run via `pytest tests/ -v` and use relative imports/fixtures that don't work with direct `import`. The test files are present and syntactically valid — full pytest run needed to verify actual pass/fail status.

## 4. Empty Directories (Should Have Content)

| Directory | Expected Contents |
|-----------|-------------------|
| `code/baselines/crb_analysis/src/` | Python source modules |
| `code/baselines/crb_analysis/tests/` | Test files |
| `code/baselines/crb_analysis/examples/` | Demo scripts |
| `code/baselines/crb_analysis/configs/` | YAML configs |
| `code/baselines/mimo_beamforming/src/` | Python source modules |
| `code/baselines/mimo_beamforming/tests/` | Test files |
| `code/baselines/mimo_beamforming/examples/` | Demo scripts |
| `code/baselines/mimo_beamforming/configs/` | YAML configs |
| `code/baselines/radar_comm_tradeoff/src/` | Python source modules |
| `code/baselines/radar_comm_tradeoff/tests/` | Test files |
| `code/baselines/radar_comm_tradeoff/examples/` | Demo scripts |
| `code/baselines/radar_comm_tradeoff/configs/` | YAML configs |
| `code/baselines/ofdm_isac/tests/` | Test files |
| `code/baselines/ofdm_isac/examples/` | Demo scripts |
| `code/baselines/ofdm_isac/configs/` | YAML configs |
| `code/baselines/csi_ratio_doppler_estimation/configs/` | YAML configs |
| `code/baselines/csi_ratio_doppler_estimation/data/` | Dataset files |
| `assets/figures/` | Figure assets |

## 5. Figure References

### ✅ All Images Present
- `csi_ratio_doppler_estimation`: 6 images all exist
- `isac_capacity_distortion`: 1 image (`results/p0a_rate_crb_tradeoff.png`) ✅
- `isac_energy_efficient_beamforming`: 4 images all exist
- `ofdm_ambiguity_function`: 4 images all exist
- `ris_isac_beamforming`: 3 images all exist ✅ (just updated)
- `xl_mimo_beam_training`: 6 images all exist

### ❌ Broken Image References
| Baseline | Referenced Image | Actual Files in results/ |
|----------|-----------------|--------------------------|
| `isac_resource_allocation` | `results/p0d_detection_localization.png` | Missing |
| `isac_resource_allocation` | `results/p0d_resource_allocation.png` | Missing |

**Available files:** `p0d_allocation_heatmap.png`, `p0d_sensing_rate_tradeoff.png`, `p0d_tracking_pcrb.png`

## 6. GitHub Templates

| Template | Format | Status |
|----------|--------|--------|
| `.github/ISSUE_TEMPLATE/bug_report.yml` | YAML form | ✅ Valid structure |
| `.github/ISSUE_TEMPLATE/new_baseline.yml` | YAML form | ✅ Valid structure |
| `.github/ISSUE_TEMPLATE/new_paper.yml` | YAML form | ✅ Valid structure |
| `.github/PULL_REQUEST_TEMPLATE.md` | Markdown | ✅ Present |
| `.github/PAPER_REVIEWER.md` | Markdown | ✅ Present |
| `.github/workflows/test.yml` | CI workflow | ✅ Present |
| `.github/workflows/link-check.yml` | CI workflow | ✅ Present |

---

## Summary

| Category | Status |
|----------|--------|
| Broken links | 2 issues |
| Missing READMEs | 4 baselines (stubs) |
| Missing sections | Citation (6 baselines), Badges (1), What It Implements (1), Math Background (1) |
| Empty directories | 18 directories |
| Broken image refs | 2 (in isac_resource_allocation) |
| GitHub templates | All valid |
| Test infrastructure | Present (6 baselines), needs pytest run for verification |

### Priority Fixes
1. **High:** Fix broken link in `code/README.md` (`../../CONTRIBUTING.md` → `../CONTRIBUTING.md`)
2. **High:** Fix broken images in `isac_resource_allocation/README.md` (update to existing files)
3. **Medium:** Fix placeholder link in `tools/README.md` (PyRadar)
4. **Medium:** Add Citation sections to baselines missing them
5. **Low:** Add placeholder READMEs to stub baselines, or remove them
6. **Low:** Populate empty directories (configs/, data/)
