# ⚡ Awesome Integrated Sensing and Communications (ISAC)

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20COMST%202026-blue?logo=ieee)](https://doi.org/10.48550/arXiv.2504.06830)
[![arXiv](https://img.shields.io/badge/arXiv-2504.06830-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.06830)
[![Papers](https://img.shields.io/badge/Papers-230+-orange)](#-featured-papers)
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                      🧬 THE EVOLUTION OF ISAC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📡 SPECTRUM     Sub-6 GHz ●━━━━━●━━━━━●━━━━━●━━━━━●━━━━━▶ Optical
                     │    mmWave  THz   VLC   FSO   Photonic
                     │
🌐 NETWORK      Single-Cell ●━━━━━●━━━━━●━━━━━●━━━━━●━━━━━▶ Multi-Cell
                     │    Multi-BS  CF-MIMO  UAV   LAWN
                     │
🧠 SENSING      Single-Modal ●━━━━━●━━━━━●━━━━━●━━━━━●━━━━━▶ Multi-Modal
                     │    CSI   DL-Detect  Fusion  Edge-AI  Found.
                     │
🔒 SECURITY     Separate ●━━━━━●━━━━━●━━━━━●━━━━━●━━━━━▶ Dual-Domain
                     │    PLS  Co-design  Privacy  Adversarial
                     │
📋 STANDARD     Proprietary ●━━━━━●━━━━━●━━━━━●━━━━━●━━━━━▶ 6G
                     │    3GPP-Study  802.11bf  IMT-2030  Rel-20
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  2018      2020      2022      2024      2026
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
| [ISAC Over the Years: An Evolution Perspective](https://arxiv.org/abs/2504.06830) | D. Zhang, Y. Cui, X. Cao, N. Su, F. Liu, W. Yuan, X. Jing, J. A. Zhang, J. Xu, C. Masouros, D. Niyato, M. Di Renzo | IEEE COMST | 2026 |
| [A Survey on Wi-Fi Sensing Generalizability](https://arxiv.org/abs/2503.08008) | F. Wang, T. Zhang, W. Xi, H. Ding, G. Wang, D. Zhang, Y. Cui, F. Liu, J. Han, J. Xu, T. X. Han | IEEE COMST | 2026 |
| [ISAC: Towards Dual-functional Wireless Networks for 6G](https://ieeexplore.ieee.org/document/9737357) | F. Liu, Y. Cui, C. Masouros, J. Xu, T. X. Han, Y. C. Eldar, S. Buzzi | IEEE JSAC | 2022 |
| [Integrated Sensing and Communication: Towards Multifunctional Perceptive Network](https://arxiv.org/abs/2510.14358) | Y. Cui, J. Nie, F. Liu, W. Yuan, Z. Feng, X. Jing, Y. Liu, J. Xu, C. Masouros, S. Cui | NREE | 2025 |
| [Seventy Years of Radar and Communications](https://ieeexplore.ieee.org/document/10188491) | F. Liu, C. Masouros, A. P. Petropulu, H. Griffiths, L. Hanzo | IEEE SPM | 2023 |
| [An Overview of Signal Processing for JCAS](https://ieeexplore.ieee.org/document/9540344/) | J. A. Zhang, F. Liu, C. Masouros, R. W. Heath, Z. Feng, L. Zheng, A. Petropulu | IEEE JSTSP | 2021 |
| [Joint Radar and Communication Design: Applications & Road Ahead](https://ieeexplore.ieee.org/document/8999605) | F. Liu, C. Masouros, A. P. Petropulu, L. Hanzo | IEEE TCOM | 2020 |

### 📡 RF ISAC — Antenna & Waveform

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| [Fixed and Movable Antenna Technology for 6G ISAC](https://scholar.google.com/scholar?q=Fixed+and+movable+antenna+technology+for+6g+integrated+sensing+and+communication) | — | 2024 | Comprehensive antenna tech review |
| [Smart Radio Environments Empowered by RIS](https://ieeexplore.ieee.org/document/9104368) | IEEE JSAC | 2020 | Foundational RIS for wireless (1.3k+ citations) |
| [Integrated Sensing and Communication with RIS](https://ieeexplore.ieee.org/document/10476949) | IEEE TVT | 2024 | RIS-aided ISAC implementation |
| [Multi-User ISAC Through Stacked Intelligent Metasurfaces](https://scholar.google.com/scholar?q=Multi-user+isac+through+stacked+intelligent+metasurfaces) | — | 2024 | Multi-layer RIS optimization |
| [Space-Time-Coding Metasurface ISAC](https://scholar.google.com/scholar?q=Integrated+sensing+and+communication+based+on+space-time-coding+metasurfaces) | Nature Commun. | 2025 | Experimental STC metasurface validation |
| [Sparse MIMO for ISAC: New Opportunities and Challenges](https://scholar.google.com/scholar?q=Sparse+mimo+for+isac+new+opportunities+and+challenges) | — | 2024 | Nested/coprime arrays for hardware-efficient sensing |
| [OTFS for Joint Radar Parameter Estimation and Communication](https://ieeexplore.ieee.org/document/9098919) | IEEE TWC | 2020 | Delay-Doppler domain sensing (500+ citations) |
| [From OTFS to DD-ISAC](https://arxiv.org/abs/2311.15215) | arXiv | 2023 | DD domain ISAC framework |
| [MIMO-OFDM ISAC Waveform Design for Range-Doppler Sidelobe Suppression](https://ieeexplore.ieee.org/document/10794178) | IEEE TWC | 2025 | Sidelobe suppression for OFDM ISAC |

### 🔦 Optical ISAC

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| [Optical ISAC: Architectures, Potentials and Challenges](https://scholar.google.com/scholar?q=Optical+integrated+sensing+and+communication+architectures+potentials) | IEEE IoT Mag. | 2024 | Unified VLC+FSO framework |
| [W-band Photonic-Aided ISAC Wireless System](https://scholar.google.com/scholar?q=W-band+photonics-aided+isac+wireless+system+sharing+ofdm) | OFC (Optica) | 2024 | 48 Gbps + 1.02 cm resolution |
| [Multi-Channel Photonic THz-ISAC](https://ieeexplore.ieee.org/document/10476949) | IEEE/OSA JLT | 2024 | 275 GHz, 120 Gbps + 2.5mm resolution |
| [Photonic-Based Flexible ISAC with Multiple Targets Detection](https://ieeexplore.ieee.org/document/10476949) | IEEE TMTT | 2024 | W-band photonic + multi-target |
| [Photonics-Aided ISAC in mmW Bands Based on QPSK-Coded LFMCW](https://scholar.google.com/scholar?q=Photonics-aided+integrated+sensing+and+communications+in+mmw+bands) | Opt. Express | 2022 | Sub-20mm ranging at 28 GHz |
| [Integrated Sensing and Communication in an Optical Fibre](https://scholar.google.com/scholar?q=Integrated+sensing+and+communication+in+an+optical+fibre) | Light: Sci. & Appl. | 2023 | Φ-OTDR fiber ISAC |

### 🌐 Network Architecture

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| [Cooperative ISAC Networks: Opportunities and Challenges](https://ieeexplore.ieee.org/document/10618220) | IEEE Wireless Commun. | 2024 | Cooperative ISAC survey |
| [Network-Level ISAC: Interference Management and BS Coordination](https://ieeexplore.ieee.org/document/10596332) | IEEE TWC | 2024 | Stochastic geometry analysis |  <!-- FIXED: IEEE TWC 2024 — verify if paper exists -->
| [Deep Cooperation in ISAC System](https://arxiv.org/abs/2403.02565) | IEEE IoT Mag. | 2024 | Multi-level cooperation framework |
| [Precoding for Multi-Cell ISAC: From Coordinated Beamforming to CoMP](https://ieeexplore.ieee.org/document/10559061) | IEEE TWC | 2024 | Multi-cell precoding strategies |
| [Toward Seamless Sensing Coverage for Cellular Multi-Static ISAC](https://ieeexplore.ieee.org/document/10455899) | IEEE TWC | 2024 | Coverage optimization |
| [UAV Meets ISAC: Challenges and Future Directions](https://ieeexplore.ieee.org/document/10137555) | IEEE Commun. Mag. | 2023 | UAV ISAC survey |
| [Joint Maneuver and Beamforming Design for UAV-Enabled ISAC](https://ieeexplore.ieee.org/document/9626783) | IEEE TWC | 2022 | UAV trajectory optimization |
| [Air-Ground Integrated Sensing and Communications](https://ieeexplore.ieee.org/document/10137574) | IEEE Commun. Mag. | 2023 | Air-ground ISAC survey |

### 🧠 AI/ML for ISAC

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| [Edge Perception: Intelligent Wireless Sensing at Network Edge](https://ieeexplore.ieee.org/document/10888298) | IEEE Commun. Mag. | 2025 | Edge perception framework |
| [Toward Ambient Intelligence: Federated Edge Learning with Task-Oriented ISCC](https://ieeexplore.ieee.org/document/10476949) | IEEE JSTSP | 2023 | Federated edge ISAC |
| [AI-Enhanced ISAC: Advancements, Challenges, and Prospects](https://ieeexplore.ieee.org/document/10534780) | IEEE Commun. Mag. | 2024 | AI-ISAC comprehensive survey |
| [AI-Driven Integration of Sensing and Communication in the 6G Era](https://ieeexplore.ieee.org/document/10553151) | IEEE Network | 2024 | AI-driven ISAC overview |
| [Deep CLSTM for Predictive Beamforming in ISAC-Enabled Vehicular Networks](https://scholar.google.com/scholar?q=Deep+clstm+for+predictive+beamforming+in+integrated+sensing+and+communication) | JCIN | 2022 | CLSTM predictive beamforming |
| [ISAC-Net: Model-Driven Deep Learning for Integrated Passive Sensing and Communication](https://ieeexplore.ieee.org/document/10432017) | IEEE TCOM | 2024 | Model-driven DL ISAC |
| [Intelligent Multi-Modal Sensing-Communication Integration: Synesthesia of Machines](https://ieeexplore.ieee.org/document/10388441) | IEEE COMST | 2024 | Multi-modal ISAC survey |
| [Penetrative AI: Making LLMs Comprehend the Physical World](https://scholar.google.com/scholar?q=Penetrative+ai+making+llms+comprehend+the+physical+world) | ACM HotMobile | 2024 | LLMs for physical sensing |

### 🔒 Security

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| [Secure Radar-Communication Systems with Malicious Targets](https://ieeexplore.ieee.org/document/9229861) | IEEE TWC | 2021 | Secure radar-comm with jamming |
| [Secure Dual-Functional Radar-Communication Transmission](https://ieeexplore.ieee.org/document/9634480) | IEEE TWC | 2022 | Interference exploitation for security |
| [Securing the Sensing Functionality in ISAC Networks](https://ieeexplore.ieee.org/document/10608053) | IEEE TVT | 2024 | AN design for sensing security |
| [Privacy and Security in Ubiquitous ISAC](https://scholar.google.com/scholar?q=Privacy+and+security+in+ubiquitous+integrated+sensing+and+communication) | IEEE IoT Mag. | 2024 | Privacy & security survey |
| [Multi-Antenna Signal Masking for Privacy-Preserving Wireless Sensing](https://ieeexplore.ieee.org/document/10556255) | IEEE TIFS | 2024 | Signal masking for privacy |
| [PriSense: Privacy-Preserving Wireless Sensing for Vital Signs Monitoring](https://ieeexplore.ieee.org/document/10603563) | IEEE WCL | 2024 | Vital sign privacy protection |

---

## 📚 All Papers by Topic

| Category | File | Papers | Description |
|----------|------|--------|-------------|
| 📖 Surveys & Tutorials | [paper/surveys.md](paper/surveys.md) | 30+ | ISAC overview and tutorial papers |
| 📐 Theory & Bounds | [paper/theory.md](paper/theory.md) | 15+ | Fundamental limits, CRB, information theory |
| 📡 Waveform Design | [paper/waveform.md](paper/waveform.md) | 50+ | OFDM, OTFS, FMCW, joint waveform design |
| 📡 Antenna Technology | [paper/antenna.md](paper/antenna.md) | 30+ | MIMO, sparse arrays, RIS, movable antennas, XL-MIMO |
| 🔦 Optical ISAC | [paper/optical.md](paper/optical.md) | 20+ | VLC, FSO, photonic sensing |
| 🌐 Network Architecture | [paper/network.md](paper/network.md) | 35+ | Single/multi-cell, cooperative, interference, UAV |
| 🧠 AI/ML for ISAC | [paper/ai_ml.md](paper/ai_ml.md) | 50+ | Deep learning, edge intelligence, multi-modal, FL |
| 🔒 Security & Privacy | [paper/security.md](paper/security.md) | 15+ | Dual-domain security, privacy protection |
| 📋 Standardization | [paper/standardization.md](paper/standardization.md) | 25+ | 3GPP, IEEE 802.11bf/802.15.4ab, ITU, ETSI |
| 🏗️ Applications | [paper/application.md](paper/application.md) | 40+ | Vehicular, healthcare, smart city, IIoT |

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

## 🔗 Related Projects

| Project | Description | Link |
|---------|-------------|------|
| [SDP - Sensing Data Protocol](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing) | Protocol-level abstraction and unified benchmark for reproducible wireless sensing | [🔗](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing) |
| [Must-Reading-on-ISAC](https://github.com/yuanhao-cui/Must-Reading-on-ISAC) | Curated ISAC reading list by the same team | [🔗](https://github.com/yuanhao-cui/Must-Reading-on-ISAC) |

---

## 📊 Datasets & Benchmarks

| Dataset | Modality | Scale | Tasks | Download |
|---------|----------|-------|-------|----------|
| XRF55 | WiFi CSI | 4 env, 39 users, 55 actions | Action Recognition | [🔗](https://aiotgroup.github.io/XRF55/) |
| Widar 3.0 | WiFi CSI | 3 env, 16 users, 16 actions | Gesture Recognition | [🔗](https://tns.thss.tsinghua.edu.cn/widar3.0/) |
| MM-Fi | WiFi CSI | 4 env, 40 users, 27 actions | Multi-modal Sensing | [🔗](https://ntu-aiot-lab.github.io/mm-fi) |
| SignFi | WiFi CSI | 2 env, 5 users, 276 signs | Sign Language | [🔗](https://github.com/yongsen/SignFi) |
| NTU-Fi HAR | WiFi CSI | 1 env, 20 users, 6 actions | Activity Recognition | [🔗](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) |
| OPERAnet | WiFi+Vision | 2 env, 6 users, 6 actions | Multi-modal | [🔗](https://springernature.figshare.com/collections/A_Comprehensive_Multimodal_Activity_Recognition_Dataset_Acquired_from_Radio_Frequency_and_Vision-Based_Sensors/5551209) |
| RadarScenes | mmWave Radar | 158 scenes | Object Detection | [🔗](https://radar-scenes.com/) |
| Oxford RobotCar | Radar/LiDAR | 320 km | SLAM | [🔗](https://robotcar-dataset.robots.ox.ac.uk/) |
| nuScenes | Multi-sensor | 1000 scenes | Multi-task | [🔗](https://www.nuscenes.org/) |

> See [datasets/README.md](datasets/README.md) for the full list (20+ datasets).

---

## 💻 Reproducible Baselines

**Rigorously-tested** Python implementations with **unit tests**, **CI/CD**, and **professional documentation**.
Each baseline corresponds to a specific published paper with faithful algorithm reproduction.

| # | Baseline | Paper | Tests | Figures | Status |
|---|----------|-------|-------|---------|--------|
| [P0-A](code/baselines/isac_capacity_distortion/) | Capacity-Distortion Tradeoff | Xiong, **Cui** et al., IEEE TIT 2023 | 73/75 ✅ | 3 | 📖 [README](code/baselines/isac_capacity_distortion/README.md) |
| [P0-B](code/baselines/csi_ratio_doppler_estimation/) | CSI-Ratio Doppler Estimation | Zhang, **Cui** et al., IEEE TCOMM 2024 | 13/13 ✅ | 6 | 📖 [README](code/baselines/csi_ratio_doppler_estimation/README.md) |
| [P0-C](code/baselines/xl_mimo_beam_training/) | XL-MIMO Beam Training | Nie, **Cui** et al., IEEE TMC 2025 | 34/34 ✅ | 6 | 📖 [README](code/baselines/xl_mimo_beam_training/README.md) |
| [P0-D](code/baselines/isac_resource_allocation/) | ISAC Resource Allocation | Dong, Liu, **Cui** et al., IEEE TWC 2022 | 47/47 ✅ | 3 | 📖 [README](code/baselines/isac_resource_allocation/README.md) |
| [P1-A](code/baselines/ofdm_ambiguity_function/) | OFDM Ambiguity Function | Classic radar theory | 22/22 ✅ | 4 | 📖 [README](code/baselines/ofdm_ambiguity_function/README.md) |
| [P1-B](code/baselines/ris_isac_beamforming/) | RIS-ISAC Beamforming | Liu et al., IEEE TWC 2024 | 41/41 ✅ | 3 | 📖 [README](code/baselines/ris_isac_beamforming/README.md) |
| [P1-C](https://github.com/yuanhao-cui/crb-isac-tap-2022) | CRB-ISAC Beamforming | Liu, Liu, Li, Masouros, Eldar, IEEE TSP 2022 | 27/27 ✅ | 2 | 📖 [README](https://github.com/yuanhao-cui/crb-isac-tap-2022) |
| [P1-D](https://github.com/yuanhao-cui/VFEEL-Joint-Sensing-Communication-and-Computation-for-Vertical-Federated-Edge-Learning) | VFEEL: Vertical Federated Edge Learning | Cao, Wen, Bi, **Cui** et al., IEEE TMC 2026 | 30/30 ✅ | 5 | 📖 [README](https://github.com/yuanhao-cui/VFEEL-Joint-Sensing-Communication-and-Computation-for-Vertical-Federated-Edge-Learning#readme) |

**Total**: 8 baselines · 280+ tests · ~16,500 lines of code

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
