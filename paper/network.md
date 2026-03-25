# 🌐 Network Architecture

> The evolution of ISAC network architecture—from isolated single-cell systems to coordinated multi-cell, cooperative, and space-air-ground integrated networks—represents a key pillar of 6G research. This section covers waveform design for single-cell ISAC, multi-cell cooperative sensing, interference management, resource allocation, UAV/drone networks, and emerging low-altitude wireless networks (LAWN).

## Single-Cell ISAC: Waveform Design

Single-cell ISAC systems allocate resources for sensing and communication within a single base station, spanning orthogonal resource allocation (time/frequency/spatial/code division) and unified waveform designs. The fundamental ISAC tradeoff is captured by the capacity-distortion framework.

| [On the Fundamental Tradeoff of Integrated Sensing and Communications Under Gaussian Channels](https://arxiv.org/abs/2204.06938) | Y. Xiong, F. Liu, **Y. Cui** et al. | IEEE Trans. Inf. Theory | 2023 | — | Capacity-distortion tradeoff theory for ISAC |
|----------|----------|----------|----------|----------|----------|
| [Waveform Design for MIMO-OFDM Integrated Sensing and Communication System: An Information Theoretical Approach](https://arxiv.org/abs/2310.05444) | Z. Wei et al. | IEEE Trans. Commun. | 2024 | — | MI-based waveform optimization for MIMO-OFDM ISAC |
| [Time-Division ISAC Enabled Connected Automated Vehicles Cooperation](https://ieeexplore.ieee.org/document/9786023) | Q. Zhang et al. | IEEE JSAC | 2022 | — | TD-ISAC algorithm for cooperative vehicles |
| [A Dual Function Compromise for Uplink ISAC: Joint Spectrum and Power Management](https://ieeexplore.ieee.org/document/10433808) | Y. Li, Z. Wei, **Y. Cui** et al. | IEEE WCNC | 2024 | — | Joint subcarrier and power allocation for uplink ISAC |
| [Power Minimization Strategy Based Subcarrier Allocation for ISAC](https://ieeexplore.ieee.org/document/10119223) | J. Zhu, **Y. Cui**, J. Mu et al. | IEEE WCNC | 2023 | — | Power-efficient frequency-division ISAC |
| [An Integrated Sensing and Communications Waveform Design: Experimental Proof of Concept](https://ieeexplore.ieee.org/document/9903001/) | T. Xu et al. | IEEE OJCOMS | 2022 | — | Real-time PoC validation of ISAC waveform tradeoffs |
| [Mutual Information-Based Integrated Sensing and Communications: A WMMSE Framework](https://ieeexplore.ieee.org/document/10648701) | Y. Peng et al. | IEEE Wireless Commun. Lett. | 2024 | — | WMMSE optimization for MI-based ISAC |

## Multi-Cell / Cooperative ISAC

Multi-cell ISAC enables collaborative sensing across multiple base stations, improving coverage, accuracy, and interference management. Cooperative strategies include coordinated beamforming, CoMP sensing, and distributed MIMO configurations.

| [Cooperative ISAC Networks: Opportunities and Challenges](https://arxiv.org/abs/2405.06305) | K. Meng, C. Masouros et al. | IEEE Wireless Commun. | 2024 | — | Comprehensive survey on cooperative ISAC architectures |
|----------|----------|----------|----------|----------|----------|
| [Network-Level Integrated Sensing and Communication: Interference Management and BS Coordination Using Stochastic Geometry](https://arxiv.org/abs/2311.09052) | K. Meng et al. | IEEE Trans. Wireless Commun. | 2024 | — | Stochastic geometry analysis for multi-cell ISAC networks |
| [Precoding for Multi-Cell ISAC: From Coordinated Beamforming to CoMP](https://ieeexplore.ieee.org/document/10559061) | N. Babu et al. | IEEE Trans. Wireless Commun. | 2024 | — | Multi-cell precoding from coordinated beamforming to bi-static sensing |
| [Toward Seamless Sensing Coverage for Cellular Multi-Static ISAC](https://ieeexplore.ieee.org/document/10455899) | R. Li et al. | IEEE Trans. Wireless Commun. | 2024 | — | Coverage optimization for multi-static cellular ISAC |
| [ISAC Enabled Multiple Base Stations Cooperative Sensing Towards 6G](https://ieeexplore.ieee.org/document/10583163) | Z. Wei et al. | IEEE Network | 2024 | — | Multi-BS cooperative sensing architecture |
| [Collaborative Sensing in Perceptive Mobile Networks: Opportunities and Challenges](https://ieeexplore.ieee.org/document/10058017) | L. Xie et al. | IEEE Wireless Commun. | 2023 | — | Collaborative sensing survey for PMN |
| [Networked Sensing in 6G Cellular Networks: Opportunities and Challenges](https://arxiv.org/abs/2206.00493) | L. Liu et al. | arXiv | 2022 | — | Overview of networked sensing in cellular systems |

## Interference Management

Interference management in ISAC networks addresses the challenge of separating sensing echoes from communication signals across cells, users, and S&C domains.

| [Distributed Unsupervised Learning for Interference Management in Integrated Sensing and Communication Systems](https://ieeexplore.ieee.org/document/10217003) | X. Liu et al. | IEEE Trans. Wireless Commun. | 2023 | — | Unsupervised learning for ISAC interference mitigation |
|----------|----------|----------|----------|----------|----------|
| [Joint Transmit Beamforming and Receive Filters Design for Coordinated Two-Cell Interfering DFRC Networks](https://ieeexplore.ieee.org/document/9843549) | Y. Li and M. Jiang | IEEE Trans. Veh. Technol. | 2022 | — | Inter-cell interference management via coordinated beamforming |
| [Framework for a Perceptive Mobile Network Using Joint Communication and Radar Sensing](https://ieeexplore.ieee.org/document/9098919) | M. L. Rahman et al. | IEEE Trans. Aerosp. Electron. Syst. | 2021 | — | PMN framework with compressed sensing for interference extraction |
| [Opportunistic Cooperation of Integrated Sensing and Communications in Wireless Networks](https://ieeexplore.ieee.org/document/10563539) | C. Jia et al. | IEEE Wireless Commun. Lett. | 2024 | — | Opportunistic ISAC leveraging interference fluctuations |

## Resource Allocation

Resource allocation in ISAC networks optimizes power, bandwidth, and time-frequency resources to balance communication throughput and sensing quality of service across multiple users and targets.

| [Task-Oriented Sensing, Computation, and Communication Integration for Multi-Device Edge AI](https://ieeexplore.ieee.org/document/10363472) | D. Wen et al. | IEEE Trans. Wireless Commun. | 2024 | — | Joint SCC optimization for multi-device edge intelligence |
|----------|----------|----------|----------|----------|----------|
| [Sensing as a Service in 6G Perceptive Mobile Networks: Architecture, Advances, and the Road Ahead](https://arxiv.org/abs/2308.08185) | F. Dong, F. Liu, **Y. Cui** et al. | IEEE Network | 2024 | — | SaaS architecture for 6G perceptive mobile networks |
| [Deep Cooperation in ISAC System: Resource, Node and Infrastructure Perspectives](https://arxiv.org/abs/2403.02565) | Z. Wei et al. | IEEE IoT Mag. | 2024 | — | Multi-level cooperation framework for ISAC |
| [Energy-Efficient Beamforming Design for Integrated Sensing and Communications Systems](https://ieeexplore.ieee.org/document/10398869) | J. Zou et al. | IEEE Trans. Commun. | 2024 | — | Energy-efficient beamforming under QoS constraints |

## UAV / Drone Networks

UAV-enabled ISAC exploits the mobility and flexibility of aerial platforms for adaptive sensing and communication in dynamic environments. Key challenges include trajectory optimization, beamforming under high mobility, and integration with RIS.

| [UAV Meets Integrated Sensing and Communication: Challenges and Future Directions](https://ieeexplore.ieee.org/document/10004900) | J. Mu, R. Zhang, **Y. Cui** et al. | IEEE Commun. Mag. | 2023 | — | Comprehensive survey on UAV-ISAC integration |
|----------|----------|----------|----------|----------|----------|
| [Joint Maneuver and Beamforming Design for UAV-Enabled Integrated Sensing and Communication](https://arxiv.org/abs/2110.02857) | Z. Lyu et al. | IEEE Trans. Wireless Commun. | 2022 | — | UAV trajectory and beamforming optimization for ISAC |
| [Joint UAV Trajectory and Beamforming Design for RIS-Aided ISAC](https://ieeexplore.ieee.org/document/10468846) | C. Dai et al. | IEEE INFOCOM Workshops | 2024 | — | RIS-assisted UAV ISAC trajectory optimization |
| [IRS-UAV Assisted Secure Integrated Sensing and Communication](https://ieeexplore.ieee.org/document/10563387) | J. Xu et al. | IEEE Wireless Commun. | 2024 | — | IRS-assisted secure UAV ISAC |

## Space-Air-Ground Integrated Networks (SAGIN) & LAWN

Space-air-ground integration extends ISAC coverage across satellite, aerial, and terrestrial layers, enabling seamless connectivity and sensing in challenging or remote environments. Low-Altitude Wireless Networks (LAWN) represent an emerging focus area.

| [Air-Ground Integrated Sensing and Communications: Opportunities and Challenges](https://arxiv.org/abs/2302.06044) | Z. Fei et al. | IEEE Commun. Mag. | 2023 | — | Air-ground ISAC survey covering UAV, HAP, and drone platforms |
|----------|----------|----------|----------|----------|----------|
| [Beam Squint-Aware ISAC for Hybrid Massive MIMO LEO Satellite Systems](https://ieeexplore.ieee.org/document/9848832) | L. You et al. | IEEE JSAC | 2022 | — | LEO satellite ISAC with beam squint mitigation |
| [Space-Air-Ground Integrated Wireless Networks for 6G](https://ieeexplore.ieee.org/document/10514249) | Y. Xiao et al. | IEEE JSAC | 2024 | — | SAGIN architecture for 6G networks |
| [Energy-Efficient Cooperative Spectrum Sensing in Cognitive Satellite Terrestrial Networks](https://ieeexplore.ieee.org/document/9184071) | J. Hu et al. | IEEE Access | 2020 | — | Cognitive satellite-ground ISAC with distributed sensing |

## Synchronization & Timing

Distributed ISAC networks face critical clock-asynchrony challenges that degrade range-Doppler estimation and require joint synchronization-positioning solutions.

| [ACES: Adaptive Clock Estimation and Synchronization Using Kalman Filtering](https://doi.org/10.1145/1409944.1409987) | B. R. Hamilton et al. | ACM MobiCom | 2008 | — | Kalman-filter-based clock synchronization |
|----------|----------|----------|----------|----------|----------|
| [Moving Object Localization in Distributed MIMO with Clock and Frequency Offsets](https://ieeexplore.ieee.org/document/10346883) | Q. Lin et al. | IEEE Sensors J. | 2023 | — | Offset-robust localization in distributed MIMO |
| [Target Localization in Asynchronous Distributed MIMO Radar Systems](https://ieeexplore.ieee.org/document/9605807) | G. Wen et al. | IEEE Trans. Wireless Commun. | 2022 | — | Cooperative target localization under clock offsets |

---

*Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md).*
