# ⚡ Awesome Integrated Sensing and Communications (ISAC)

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20COMST%202026-blue?logo=ieee)](https://doi.org/10.48550/arXiv.2504.06830)
[![arXiv](https://img.shields.io/badge/arXiv-2504.06830-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.06830)
[![Papers](https://img.shields.io/badge/Papers-200+-orange)](#-featured-papers)
[![Datasets](https://img.shields.io/badge/Datasets-20+-purple)](#-datasets--benchmarks)
[![Tools](https://img.shields.io/badge/Tools-8+-green)](#-open-source-tools)
[![Stars](https://img.shields.io/github/stars/yuanhao-cui/awesome-integrated-sensing-and-communications?style=social)](https://github.com/yuanhao-cui/awesome-integrated-sensing-and-communications/stargazers)
[![Tests](https://img.shields.io/github/actions/workflow/status/yuanhao-cui/awesome-integrated-sensing-and-communications/test.yml?label=CI)](https://github.com/yuanhao-cui/awesome-integrated-sensing-and-communications/actions)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-sa/4.0/)

> 📡 A curated list of **Integrated Sensing and Communications** resources,
> accompanying our survey: **"ISAC Over the Years: An Evolution Perspective"**
> (IEEE COMST, 2026).

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

**Authors**: Di Zhang (BUPT) · **Yuanhao Cui** (BUPT) · Xiaowen Cao (SZU) · Nanchi Su (HIT-SZ) · Yi Gong (HIT-SZ) · Fan Liu (SEU) · Weijie Yuan (SEU) · Xiaojun Jing (BUPT) · J. Andrew Zhang (UTS) · Jie Xu (CUHK-SZ) · Christos Masouros (UCL) · Dusit Niyato (NTU) · Marco Di Renzo (Paris-Saclay / KCL)

---

## 📑 Table of Contents

- [📝 Citation](#-citation)
- [🧬 The Evolution of ISAC](#-the-evolution-of-isac)
- [📅 Evolution Timelines](#-evolution-timelines-by-subfield)
- [⭐ Featured Papers](#-featured-papers)
- [📖 Surveys & Tutorials](paper/surveys.md)
- [📚 All Papers by Topic](#-all-papers-by-topic)
- [🧰 Open-Source Tools](#-open-source-tools)
- [📊 Datasets & Benchmarks](#-datasets--benchmarks)
- [💻 Reproducible Baselines](#-reproducible-baselines)
- [🏆 Leaderboard](#-leaderboard)
- [🤝 Contributing](CONTRIBUTING.md)

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

## 📅 Evolution Timelines by Subfield

### 📡 Axis 1: RF → Optical ISAC

```
2014 ─── ISAC concept introduced (mid-2010s)
  │      First OFDM-based joint waveform designs
  │
2018 ─── Massive MIMO + mmWave ISAC emerges
  │      Sparse/nested array for ISAC proposed
  │
2019 ─── RIS-assisted ISAC gains traction
  │      W-band (96 GHz) ISAC: 48 Gbps + cm resolution
  │
2020 ─── OTFS-based ISAC for high-mobility
  │      FMCW joint radar-communication
  │
2022 ─── XL-MIMO / near-field ISAC
  │      275 GHz THz ISAC: 120 Gbps + 2.5mm resolution
  │
2023 ─── Optical ISAC framework (VLC + FSO unified)
  │      Movable/fluid antenna for ISAC
  │
2024 ─── Photonic W-band ISAC: cm-level + Gbps
  │      Space-time-coding metasurface ISAC (experimental)
  │
2025+── RF-optical hybrid architectures
       Sub-mm imaging, holographic MIMO
```

### 🌐 Axis 2: Single-Cell → Multi-Cell Network

```
2018 ─── Single-cell ISAC waveform optimization
  │      Per-resource-block OFDM radar
  │
2020 ─── Multi-BS cooperative sensing concept
  │      Macro-micro BS cooperation
  │
2021 ─── Multi-cell ISAC interference management
  │      Cell-free massive MIMO for ISAC
  │
2022 ─── Coordinated multi-cell sensing + communication
  │      Joint resource allocation across cells
  │
2023 ─── Multi-static sensing architectures
  │      Distributed data fusion strategies
  │
2024 ─── ISAC network-level optimization
  │      UAV/drone network ISAC
  │
2025+── Low-altitude network (LAWN) ISAC
       Terrestrial-aerial integrated sensing
```

### 🧠 Axis 3: Single-Modal → Multi-Modal Sensing

```
2018 ─── CSI-based WiFi sensing (single-modal)
  │      Activity recognition from channel state
  │
2019 ─── Deep learning for ISAC signal detection
  │      CNN-based target classification
  │
2020 ─── Transformer architectures for sensing
  │      First multi-modal WiFi datasets
  │
2021 ─── Multi-view cooperative sensing
  │      Edge AI for real-time processing
  │
2022 ─── GNN for ISAC resource management
  │      Federated learning for distributed ISAC
  │
2023 ─── Large-scale multi-modal ISAC datasets
  │      Radar + vision fusion for autonomous driving
  │
2024 ─── Foundation models for wireless sensing
  │      Semantic communication + ISAC
  │
2025+── ISAC-native AI/ML architectures
       Cross-modal self-supervised learning
```

### 🔒 Axis 4: Security & Privacy

```
2019 ─── Physical layer security for ISAC
  │      Sensing-aided secure communication
  │
2021 ─── ISAC-enhanced anti-eavesdropping
  │      Radar-assisted jamming detection
  │
2022 ─── Privacy risks in sensing data identified
  │      Dual-domain attack surfaces mapped
  │
2023 ─── Privacy-preserving ISAC frameworks
  │      Sensing obfuscation techniques
  │
2024 ─── Adversarial attacks on ISAC systems
  │      Dual-domain security co-design
  │
2025+── ISAC-native security standards
       Regulatory frameworks for sensing privacy
```

### 📋 Axis 5: Standardization

```
2018 ─── 3GPP initial ISAC discussions (Rel-16)
  │
2020 ─── IEEE 802.11bf (Wi-Fi Sensing) initiated
  │      3GPP Rel-17 study item on ISAC
  │
2022 ─── ITU IMT-2030: ISAC as 6G key scenario
  │      3GPP Rel-18 ISAC work item
  │
2023 ─── ETSI ISAC ISG formed (18 use cases)
  │      3GPP Rel-19: 32 ISAC use cases defined
  │
2024 ─── IEEE 802.11bf frozen (October)
  │      ISAC at Davos WEF Top 10 Emerging Tech
  │
2025 ─── IEEE 802.15.4ab (UWB sensing) in progress
  │      LAWN standardization started
  │
2026+── 3GPP Rel-20: 6G ISAC standardization
       ITU-R WP5D capability framework
```

---

## ⭐ Featured Papers

### 🔥 Landmark Surveys

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [ISAC Over the Years: An Evolution Perspective](https://arxiv.org/abs/2504.06830) | D. Zhang, **Y. Cui**, et al. | IEEE COMST | 2026 |
| [A Survey on Wi-Fi Sensing Generalizability](https://arxiv.org/abs/2503.08008) | F. Wang, ..., **Y. Cui**, F. Liu, ... | IEEE COMST | 2026 |
| [ISAC: Towards Dual-functional Wireless Networks for 6G](https://ieeexplore.ieee.org/document/9737357) | F. Liu, **Y. Cui**, et al. | IEEE JSAC | 2022 |
| [Seventy Years of Radar and Communications](https://ieeexplore.ieee.org/document/10188491) | F. Liu, ..., **Y. Cui**, et al. | IEEE SPM | 2023 |
| [An Overview of Signal Processing for JCAS](https://ieeexplore.ieee.org/document/9540344/) | J. A. Zhang, F. Liu, C. Masouros, et al. | IEEE JSTSP | 2021 |
| [Joint Radar and Communication Design: Applications & Road Ahead](https://ieeexplore.ieee.org/document/8999605) | F. Liu, C. Masouros, et al. | IEEE TCOM | 2020 |

### 📡 RF ISAC — Antenna & Waveform

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| [Sparse MIMO for ISAC](...) | IEEE TSP | 2022 | Nested/coprime arrays for hardware-efficient sensing |
| [Movable Antenna for DFRC](...) | IEEE TWC | 2024 | 59.8% joint metric improvement over fixed arrays |
| [RIS-Aided ISAC](...) | IEEE JSAC | 2022 | Joint beamforming + passive phase shift optimization |
| [STAR-RIS for ISAC](...) | IEEE TWC | 2023 | Simultaneous transmit-and-reflect ISAC |
| [OTFS for Joint Sensing and Communication](...) | IEEE TWC | 2022 | Delay-Doppler domain sensing in high-mobility |
| [XL-MIMO for Near-Field ISAC](...) | IEEE JSAC | 2024 | Near-field beamforming for high-resolution sensing |

### 🔦 Optical ISAC

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| [Unified Optical ISAC Framework for 6G](...) | IEEE JSAC | 2023 | VLC+FSO integration, optical > RF in urban |
| [Photonic W-band ISAC](...) | IEEE JLT | 2023 | cm-level localization + Gbps data rates |
| [Coherent Photonic ISAC](...) | OE | 2024 | Sub-20mm ranging at 28 GHz |

### 🌐 Network Architecture

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| [Cell-Free Massive MIMO for ISAC](...) | IEEE TWC | 2023 | Distributed cooperative sensing |
| [Multi-Cell ISAC Interference Management](...) | IEEE JSAC | 2023 | Coordinated interference mitigation |
| [UAV-Enabled ISAC](...) | IEEE JSAC | 2023 | Aerial sensing platform |

### 🧠 AI/ML for ISAC

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| [Deep Learning for ISAC Signal Detection](...) | IEEE JSAC | 2023 | Neural detection vs. CFAR comparison |
| [GNN for ISAC Resource Management](...) | IEEE TWC | 2024 | Graph neural networks for optimization |
| [Federated Learning for ISAC](...) | IEEE TWC | 2024 | Distributed training for privacy |

### 🔒 Security

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| [Physical Layer Security for ISAC](...) | IEEE TWC | 2023 | Sensing-aided secrecy |
| [Dual-Domain ISAC Security](...) | IEEE JSAC | 2024 | Joint S&C security co-design |

---

## 📚 All Papers by Topic

| Category | File | Papers | Description |
|----------|------|--------|-------------|
| 📖 Surveys & Tutorials | [paper/surveys.md](paper/surveys.md) | 20+ | ISAC overview and tutorial papers |
| 📐 Theory & Bounds | [paper/theory.md](paper/theory.md) | 10+ | Fundamental limits, CRB, information theory |
| 📡 Waveform Design | [paper/waveform.md](paper/waveform.md) | 20+ | OFDM, OTFS, FMCW, joint waveform design |
| 📡 Antenna Technology | [paper/antenna.md](paper/antenna.md) | 25+ | MIMO, sparse arrays, RIS, movable antennas, XL-MIMO |
| 🔦 Optical ISAC | [paper/optical.md](paper/optical.md) | 10+ | VLC, FSO, photonic sensing |
| 🌐 Network Architecture | [paper/network.md](paper/network.md) | 15+ | Single/multi-cell, cooperative, interference, UAV |
| 🧠 AI/ML for ISAC | [paper/ai_ml.md](paper/ai_ml.md) | 15+ | Deep learning, edge intelligence, multi-modal, FL |
| 🔒 Security & Privacy | [paper/security.md](paper/security.md) | 10+ | Dual-domain security, privacy protection |
| 📋 Standardization | [paper/standardization.md](paper/standardization.md) | 10+ | 3GPP, IEEE 802.11bf/802.15.4ab, ITU, ETSI |
| 🏗️ Applications | [paper/application.md](paper/application.md) | 15+ | Vehicular, healthcare, smart city, IIoT |

---

## 🧰 Open-Source Tools

| Tool | 802.11 | MIMO | Bandwidth | Platform | Link |
|------|--------|------|-----------|----------|------|
| PicoScenes | a/g/n/ac/ax/be | 4×4 | 320 MHz | x86 Linux | [🔗](https://ps.zpj.io/) |
| Nexmon CSI | n/ac | 4×4 | 80 MHz | BCM4339/4358 | [🔗](https://github.com/seemoo-lab/nexmon_csi) |
| Intel CSI Tool | n | 2×2 | 40 MHz | iwlwifi | [🔗](https://dhalperi.github.io/linux-80211n-csitool/) |
| Atheros CSI Tool | n | - | 40 MHz | Atheros | [🔗](https://github.com/xieyaxiongfly/Atheros-CSI-Tool) |
| ZTE WiFi Sensing | n/ac/ax | 3×2 | 160 MHz | ZTE | [🔗](https://github.com/WiFiZTE2025/ZTE_WiFi_Sensing) |
| Wi-ESP | b/g/n/ac | 2×2 | 40 MHz | Various | [🔗](https://github.com/wrlab/Wi-ESP) |
| BFM-Tool | ac/ax | 4×4 | 160 MHz | - | [🔗](https://github.com/Enze-Yi/BFM-tool) |

> See [tools/README.md](tools/README.md) for the full list.

---

## 📊 Datasets & Benchmarks

| Dataset | Modality | Scale | Tasks | Download |
|---------|----------|-------|-------|----------|
| XRF55 | WiFi CSI | 4 env, 39 users, 55 actions | Action Recognition | [🔗](https://aiotgroup.github.io/XRF55/) |
| Widar 3.0 | WiFi CSI | 3 env, 16 users, 16 actions | Gesture Recognition | [🔗](https://tns.thss.tsinghua.edu.cn/widar3.0/) |
| MM-Fi | WiFi CSI | 4 env, 40 users, 27 actions | Multi-modal Sensing | [🔗](https://ntu-aiot-lab.github.io/mm-fi) |
| SignFi | WiFi CSI | 2 env, 5 users, 276 signs | Sign Language | [🔗](https://github.com/yongsen/SignFi) |
| NTU-Fi HAR | WiFi CSI | 1 env, 20 users, 6 actions | Activity Recognition | [🔗](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) |
| OPERAnet | WiFi+Vision | 2 env, 6 users, 6 actions | Multi-modal | [🔗](https://springernature.figshare.com/collections/...) |
| RadarScenes | mmWave Radar | 158 scenes | Object Detection | [🔗](https://radar-scenes.com/) |
| Oxford RobotCar | Radar/LiDAR | 320 km | SLAM | [🔗](https://robotcar-dataset.robots.ox.ac.uk/) |
| nuScenes | Multi-sensor | 1000 scenes | Multi-task | [🔗](https://www.nuscenes.org/) |

> See [datasets/README.md](datasets/README.md) for the full list (20+ datasets).

---

## 💻 Reproducible Baselines

**Rigorously-tested** Python implementations with **unit tests** (≥80% coverage), **CI/CD**, and **Jupyter demos**.

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

## 📜 License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

---

## 🙏 Acknowledgements

Inspired by [awesome-wireless-sensing-generalization](https://github.com/aiotgroup/awesome-wireless-sensing-generalization), [Must-Reading-on-ISAC](https://github.com/yuanhao-cui/Must-Reading-on-ISAC), and the broader awesome-lists community.
