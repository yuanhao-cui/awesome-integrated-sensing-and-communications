# ⚡ Awesome Integrated Sensing and Communications (ISAC)

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20COMST%202026-blue?logo=ieee)](https://doi.org/10.48550/arXiv.2504.06830)
[![arXiv](https://img.shields.io/badge/arXiv-2504.06830-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.06830)
[![Stars](https://img.shields.io/github/stars/yuanhao-cui/awesome-integrated-sensing-and-communications?style=social)](https://github.com/yuanhao-cui/awesome-integrated-sensing-and-communications/stargazers)
[![Tests](https://img.shields.io/github/actions/workflow/status/yuanhao-cui/awesome-integrated-sensing-and-communications/test.yml?label=tests)](https://github.com/yuanhao-cui/awesome-integrated-sensing-and-communications/actions)
[![Baselines](https://img.shields.io/badge/Code%20Baselines-8-green)](#reproducible-baselines)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-sa/4.0/)

> 📡 A curated list of **Integrated Sensing and Communications** resources,
> accompanying our survey: **"ISAC Over the Years: An Evolution Perspective"**
> (Di Zhang, **Yuanhao Cui**, Xiaowen Cao, Nanchi Su, Yi Gong, Fan Liu, Weijie Yuan, Xiaojun Jing, J. Andrew Zhang, Jie Xu, Christos Masouros, Dusit Niyato, Marco Di Renzo, IEEE COMST, 2026).
>
> 🌟 Including **200+ papers**, **20+ datasets**, **8+ open-source tools**,
> and **rigorously-tested Python baselines** with unit tests.

---

## 📑 Table of Contents

- [🧬 The Evolution of ISAC](#-the-evolution-of-isac)
- [📅 Timeline](#-isac-evolution-timeline)
- [Surveys & Tutorials](paper/surveys.md)
- [Papers by Topic](#-papers-by-topic)
  - [RF-based ISAC](paper/waveform.md) · [Antenna Tech](paper/antenna.md) · [Optical ISAC](paper/optical.md)
  - [Network Architecture](paper/network.md) · [AI/ML](paper/ai_ml.md) · [Security](paper/security.md)
  - [Standardization](paper/standardization.md) · [Applications](paper/application.md) · [Theory](paper/theory.md)
- [🧰 Open-Source Tools](#-open-source-tools)
- [📊 Datasets & Benchmarks](#-datasets--benchmarks)
- [💻 Reproducible Baselines](#-reproducible-baselines)
- [🏆 Leaderboard](#-leaderboard)
- [🤝 Contributing](CONTRIBUTING.md)
- [📝 Citation](#-citation)

---

## 🧬 The Evolution of ISAC

Our survey organizes ISAC research along **five evolutionary axes**:

```
                     ┌─────────────────────────────────────────────┐
                     │        🧬 EVOLUTION OF ISAC                  │
                     └─────────────────────────────────────────────┘
                                         │
           ┌─────────────────┬───────────┼───────────┬─────────────────┐
           │                 │           │           │                 │
           ▼                 ▼           ▼           ▼                 ▼
     ┌──────────┐    ┌──────────┐ ┌──────────┐ ┌──────────┐   ┌──────────┐
     │  📡 RF → │    │ 🌐Single│ │  🧠Single│ │  🔒Dual- │   │  📋      │
     │  Optical │    │ → Multi │ │ → Multi  │ │  Domain  │   │ Standard │
     │  ISAC    │    │  Cell   │ │  Modal   │ │  Sec/Priv│   │ -ization │
     └──────────┘    └──────────┘ └──────────┘ └──────────┘   └──────────┘
         │               │            │            │               │
    Section II      Section III   Section IV    Section V      Section VI
    RF & Optical    Network Arc.  Edge Intel.   Security       3GPP/IEEE/ITU
```

| Axis | Evolution | Key Topics |
|------|-----------|------------|
| 📡 **Spectrum** | RF → Optical | mmWave/THz, VLC, FSO, Photonic Sensing |
| 🌐 **Network** | Single-Cell → Multi-Cell | Cooperative Sensing, Interference Mgmt |
| 🧠 **Sensing** | Single-Modal → Multi-Modal | Edge AI, Fusion, Datasets |
| 🔒 **Security** | Separate → Dual-Domain | Comm Security, Sensing Privacy |
| 📋 **Standard** | Proprietary → 3GPP/IEEE/ITU | NR Rel-19/20, 802.11bf, 6G Vision |

---

## 📅 ISAC Evolution Timeline

```
1934  ─── First radio system (separate S&C)
            │
2014  ─── ISAC concept emerges (mid-2010s)
            │
2018  ─── 3GPP starts ISAC discussions
            │
2020  ─── IEEE 802.11bf (Wi-Fi Sensing) initiated
            │     RIS-assisted ISAC gains traction
            │
2022  ─── ITU 6G Vision includes ISAC
            │     First ISAC testbeds demonstrated
            │
2024  ─── IEEE 802.11bf frozen
            │     ISAC at Davos WEF Top 10 Emerging Tech
            │
2026  ─── 3GPP NR Rel-20 (6G) ISAC use cases
            │     This survey published (IEEE COMST)
            ▼
Future ──  LAWN, THz ISAC, Holographic MIMO
```

---

## 📚 Papers by Topic

| Category | File | Description |
|----------|------|-------------|
| 📖 Surveys & Tutorials | [paper/surveys.md](paper/surveys.md) | ISAC overview and tutorial papers |
| 📐 Theory & Bounds | [paper/theory.md](paper/theory.md) | Fundamental limits, CRB, information theory |
| 📡 Waveform Design | [paper/waveform.md](paper/waveform.md) | OFDM, OTFS, joint waveform design |
| 📡 Antenna Technology | [paper/antenna.md](paper/antenna.md) | MIMO, sparse arrays, RIS, movable antennas |
| 🔦 Optical ISAC | [paper/optical.md](paper/optical.md) | VLC, FSO, photonic sensing |
| 🌐 Network Architecture | [paper/network.md](paper/network.md) | Single/multi-cell, cooperative, interference |
| 🧠 AI/ML for ISAC | [paper/ai_ml.md](paper/ai_ml.md) | Deep learning, edge intelligence, multi-modal |
| 🔒 Security & Privacy | [paper/security.md](paper/security.md) | Dual-domain security, privacy protection |
| 📋 Standardization | [paper/standardization.md](paper/standardization.md) | 3GPP, IEEE 802.11bf, ITU |
| 🏗️ Applications | [paper/application.md](paper/application.md) | Vehicular, healthcare, smart city, IoT |

---

## 🧰 Open-Source Tools

| Tool | Type | Language | Platform | Link |
|------|------|----------|----------|------|
| PicoScenes | CSI Platform | C++ | WiFi a/g/n/ac/ax/be | [🔗](https://ps.zpj.io/) |
| Nexmon CSI | CSI Extraction | C | WiFi n/ac (BCM4339/4358) | [🔗](https://github.com/seemoo-lab/nexmon_csi) |
| Intel CSI Tool | CSI Extraction | C | WiFi n (iwlwifi) | [🔗](https://dhalperi.github.io/linux-80211n-csitool/) |
| Atheros CSI Tool | CSI Extraction | C | WiFi n (Atheros) | [🔗](https://github.com/xieyaxiongfly/Atheros-CSI-Tool) |
| ZTE WiFi Sensing | CSI Platform | C++ | WiFi n/ac/ax | [🔗](https://github.com/WiFiZTE2025/ZTE_WiFi_Sensing) |
| BFM-Tool | Beamforming Feedback | C++ | WiFi ac/ax | [🔗](https://github.com/Enze-Yi/BFM-tool) |

> See [tools/README.md](tools/README.md) for the full list with details.

---

## 📊 Datasets & Benchmarks

| Dataset | Modality | Scale | Tasks | Download |
|---------|----------|-------|-------|----------|
| XRF55 | WiFi CSI | 4 env, 39 users, 55 actions | Action Recognition | [🔗](https://aiotgroup.github.io/XRF55/) |
| Widar 3.0 | WiFi CSI | 3 env, 16 users, 16 actions | Gesture Recognition | [🔗](https://tns.thss.tsinghua.edu.cn/widar3.0/) |
| MM-Fi | WiFi CSI | 4 env, 40 users, 27 actions | Multi-modal Sensing | [🔗](https://ntu-aiot-lab.github.io/mm-fi) |
| SignFi | WiFi CSI | 2 env, 5 users, 276 gestures | Sign Language | [🔗](https://github.com/yongsen/SignFi) |
| RadarScenes | mmWave Radar | 158 scenes | Object Detection | [🔗](https://radar-scenes.com/) |
| Oxford RobotCar | Radar/Camera | 320km | SLAM | [🔗](https://robotcar-dataset.robots.ox.ac.uk/) |

> See [datasets/README.md](datasets/README.md) for the full list with details.

---

## 💻 Reproducible Baselines

We provide **rigorously-tested** Python implementations of classic ISAC methods.
Every baseline includes **unit tests** (≥80% coverage), **CI/CD**, and **Jupyter demos**.

| # | Baseline | Section | Description | Tests | Status |
|---|----------|---------|-------------|-------|--------|
| A1 | [OFDM-ISAC](code/baselines/ofdm_isac/) | II-A | OFDM waveform for joint radar-comm | - | 🔜 Phase 2 |
| A2 | [CRB Analysis](code/baselines/crb_analysis/) | I-C | Cramér-Rao bound for ISAC estimation | - | 🔜 Phase 2 |
| A3 | [Radar-Comm Trade-off](code/baselines/radar_comm_tradeoff/) | II-A | Sensing-communication Pareto frontier | - | 🔜 Phase 2 |
| A4 | [MIMO Beamforming](code/baselines/mimo_beamforming/) | II-A1 | Joint sensing-comm beamforming | - | 🔜 Phase 2 |

> All baselines require Python ≥3.10. See [CONTRIBUTING.md](CONTRIBUTING.md) for code standards.

---

## 🏆 Leaderboard

> 🚧 Coming soon. Will track reproducible benchmark results across ISAC methods.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- 📖 How to add a new paper
- 💻 How to contribute a baseline
- 🐛 How to report issues
- 📊 How to submit benchmark results

---

## 📝 Citation

If you find this repository useful, please cite our survey:

```bibtex
@article{zhang2026isac,
  title   = {Integrated Sensing and Communications Over the Years: An Evolution Perspective},
  author  = {Zhang, Di and Cui, Yuanhao and Cao, Xiaowen and Su, Nanchi and Gong, Yi
             and Liu, Fan and Yuan, Weijie and Jing, Xiaojun and Zhang, J. Andrew
             and Xu, Jie and Masouros, Christos and Niyato, Dusit and Di Renzo, Marco},
  journal = {IEEE Communications Surveys \& Tutorials},
  year    = {2026}
}
```

---

## 📜 License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

---

## 🙏 Acknowledgements

Inspired by [awesome-wireless-sensing-generalization](https://github.com/aiotgroup/awesome-wireless-sensing-generalization), [Must-Reading-on-ISAC](https://github.com/yuanhao-cui/Must-Reading-on-ISAC), and the broader awesome-lists community.
