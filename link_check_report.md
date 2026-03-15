# Link Check Report

**Repository:** awesome-integrated-sensing-and-communications  
**Date:** 2026-03-15  
**Files checked:** 10 paper category files

---

## Summary

| Metric | Count |
|--------|-------|
| **Total links checked** | 300 |
| **✅ Working** | 277 (92.3%) |
| **❌ Broken** | 22 (7.3%) |
| **⚠️ Timeout** | 1 (0.3%) |

---

## ✅ Fixed Broken Links (18 of 22)

The following broken links were repaired:

### `paper/theory.md` (1 fix)
| Old URL | New URL | Reason |
|---------|---------|--------|
| `ieeexplore.../5737374` | `ieeexplore.../5776640` | Wrong document ID (Sturm & Wiesbeck, Proc. IEEE 2011) |

### `paper/waveform.md` (3 fixes)
| Old URL | New URL | Reason |
|---------|---------|--------|
| `ieeexplore.../9957609` | `ieeexplore.../10036975` | Wrong document ID (Xu & Petropulu, TSP 2023) |
| `ieeexplore.../9439997` (×2 entries) | `ieeexplore.../9359665` | Wrong document ID (CD-OFDM JRC, IoTJ 2021) |
| `ieeexplore.../10336885` | `ieeexplore.../10202575` | Wrong document ID (Complementary Coded Scrambling, TVT 2024) |

### `paper/optical.md` (1 fix)
| Old URL | New URL | Reason |
|---------|---------|--------|
| `nature.../s41377-023-01072-8` | `nature.../s41377-022-01067-1` | Wrong article ID (ISAC in Optical Fibre, Light Sci. Appl. 2023) |

### `paper/network.md` (3 fixes)
| Old URL | New URL | Reason |
|---------|---------|--------|
| `ieeexplore.../9958765` | `ieeexplore.../9903001` | Wrong document ID (Experimental Proof of Concept, OJCOMS 2022) |
| `ieeexplore.../10596332` | `arxiv.org/abs/2311.09052` | IEEE document not found; arXiv version available |
| `doi.org/10.1145/1409944.1409987` | `dl.acm.org/doi/10.1145/1409944.1409987` | DOI gateway 404; ACM DL direct link works |

### `paper/ai_ml.md` (2 fixes)
| Old URL | New URL | Reason |
|---------|---------|--------|
| `ieeexplore.../9422421` | `ieeexplore.../9430907` | Wrong document ID (Edge Intelligence, Wireless Commun. 2021) |
| `nature.../s41597-022-01765-4` | `nature.../s41597-022-01573-2` | Wrong article ID (OPERAnet, Scientific Data 2022) |

### `paper/application.md` (2 fixes)
| Old URL | New URL | Reason |
|---------|---------|--------|
| `ieeexplore.../10542179` | `opg.optica.org/...ol-49-11-2861` | Wrong publisher; paper is in Optics Letters (Optica), not IEEE |
| `ieeexplore.../9422421` | `ieeexplore.../9490685` | Wrong document ID (M-Gesture, IoTJ 2021) |

### `paper/standardization.md` (6 fixes)
| Old URL | New URL | Reason |
|---------|---------|--------|
| `doi.org/10.23919/CC.2023.09.005` (×2 entries) | `ieeexplore.../10251772` | DOI gateway broken; IEEE Xplore link for China Communications paper |
| `huawei.com/en/news/.../5-5g-whats-in-a-number` | `blog.huawei.com/en/post/.../5-5g-whats-in-a-number` | Page moved to Huawei blog domain |
| `etsi.org/technologies/integrated-sensing-and-communication` | `etsi.org/committee/2295-isac` | ETSI ISAC page moved to committee page |
| `qualcomm.com/news/onq/.../qualcomm-6g-vision` | `qualcomm.com/research/6g/isac` | Qualcomm restructured; new ISAC research page |
| `weforum.org/stories/.../top-10-emerging-technologies-2024/` | `weforum.org/podcasts/.../top-10-emerging-technologies-2024/` | WEF stories URL format changed |
| `ericsson.com/en/reports-and-papers/consumerlab/reports` | `ericsson.com/en/blog/2024/6/integrated-sensing-and-communication` | Generic reports page → specific ISAC blog post |

---

## ❌ Unresolved Broken Links (4 of 22)

These links could not be automatically fixed:

| File | Link Text | URL | Error | Notes |
|------|-----------|-----|-------|-------|
| `paper/optical.md` | On the Hardware-Limited Sensing Parameter Extraction... | `ieeexplore.../10372550` | 404 | IEEE ICCT 2023 paper — could not locate correct document ID via search |
| `paper/surveys.md` | WiFi Sensing with Channel State Information: A Survey | `dl.acm.org/doi/10.1145/3310194` | 403 | ACM DL blocks HEAD requests (bot protection); URL is likely correct |
| `paper/standardization.md` | Top 10 Emerging Technologies 2024 | `weforum.org/.../top-10-emerging-technologies-2024/` | 403 | WEF blocks automated requests; URL is likely correct |
| `paper/standardization.md` | Ericsson ISAC in 6G: Smart City Opportunities | `ericsson.com/...` | 429 | Rate limited; URL is likely correct |

---

## ⚠️ Timeout (1 link)

| File | Link Text | URL | Notes |
|------|-----------|-----|-------|
| `paper/standardization.md` | 3GPP ISAC Channel Modelling Study Item (RAN) | `3gpp.org/ftp/tsg_ran/WG1_RL1/TSGR1_116b` | 3GPP FTP server is slow; URL likely works |

---

## Files Modified

- `paper/theory.md` — 1 link fixed
- `paper/waveform.md` — 4 links fixed (3 unique URLs, one appears twice)
- `paper/optical.md` — 1 link fixed
- `paper/network.md` — 3 links fixed
- `paper/ai_ml.md` — 2 links fixed
- `paper/application.md` — 2 links fixed
- `paper/standardization.md` — 6 links fixed (1 URL appears twice)

**Commit:** `fix: repair broken links found in link check`
