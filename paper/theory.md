# 📐 Theory & Fundamental Performance Bounds

> Theoretical foundations of ISAC: Cramér-Rao bounds, capacity-distortion trade-offs, mutual information, information-theoretic limits, and fundamental ISAC theory.

Theoretical research in ISAC aims to establish fundamental performance limits that guide practical system design. Key theoretical frameworks include the Cramér-Rao bound (CRB) for sensing parameter estimation, capacity-distortion regions that characterize the trade-off between communication rate and sensing accuracy, and information-theoretic formulations that treat sensing and communication as dual information-extraction tasks. These foundations are critical for understanding what ISAC systems can ultimately achieve and for developing optimal waveform and resource allocation strategies.

---

## Performance Bounds & Trade-offs

This subsection covers fundamental limits on ISAC performance, including CRB-based sensing bounds, detection-rate trade-offs, and generalized Pareto optimization frameworks for multi-objective ISAC design.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [A Survey on Fundamental Limits of Integrated Sensing and Communication](https://arxiv.org/abs/2104.09954) | A. Liu, Z. Huang, M. Li, Y. Wan, W. Li, T. X. Han et al. | IEEE COMST | 2022 | — | Comprehensive review of fundamental performance bounds and metrics for ISAC systems, covering CRB, capacity, and mutual information. |
| [On the Fundamental Tradeoff of Integrated Sensing and Communication Under Gaussian Channels](https://ieeexplore.ieee.org/document/10080203) | Y. Xiong, F. Liu, K. Wan, W. Yuan, Y. Cui, C. Masouros | IEEE TIT | 2023 | — | Establishes the capacity-distortion region for ISAC under Gaussian channels, characterizing the fundamental trade-off between communication rate and sensing distortion. |
| [Fundamental Detection Probability vs. Achievable Rate Tradeoff in ISAC](https://ieeexplore.ieee.org/document/10238324) | J. An et al. | IEEE TWC | 2023 | — | Characterizes the fundamental tradeoff between target detection probability and achievable communication rate in ISAC systems. |
| [Joint Radar-Communication Transmission: A Generalized Pareto Optimization Framework](https://ieeexplore.ieee.org/document/9414894) | L. Chen et al. | IEEE TSP | 2021 | — | Proposes a Pareto optimization framework for joint radar-communication systems, enabling flexible trade-off between sensing and communication objectives. |
| [Performance Bounds and Optimization for CSI-Ratio-Based Bi-Static Doppler Sensing in ISAC](https://ieeexplore.ieee.org/document/10550409) | Y. Hu et al. | IEEE TWC | 2024 | — | Derives CRB for bi-static Doppler sensing using CSI ratios and proposes optimized sensing algorithms for ISAC networks. |

## Information Theory & Capacity

This subsection addresses information-theoretic foundations of ISAC, including mutual information-based frameworks, rate-distortion analysis, and capacity formulations that characterize the fundamental limits of joint sensing and communication.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Integrated Sensing and Communications: A Mutual Information-Based Framework](https://ieeexplore.ieee.org/document/10137511) | C. Ouyang, Y. Liu, H. Liu, J. Xu, M. Peng, Y. Cui | IEEE Commun. Mag. | 2023 | — | Proposes a unified mutual information-based framework for ISAC, treating both sensing and communication as information extraction tasks. |
| [Mutual Information-Based ISAC: A WMMSE Framework](https://ieeexplore.ieee.org/document/10602224) | Y. Peng et al. | IEEE WCL | 2024 | — | Develops a WMMSE optimization framework for mutual information-based ISAC, enabling efficient joint waveform design. |
| [On the Physical Layer of Digital Twin: An ISAC Perspective](https://ieeexplore.ieee.org/document/10253808) | Y. Cui, F. Liu, X. Jing, J. Mu, P. C. Cosman, L. B. Milstein | IEEE JSAC | 2023 | — | Explores ISAC as the physical layer enabler for digital twins, analyzing capacity limits and sensing-communication integration. |
| [Radar Mutual Information and Communication Channel Capacity of Integrated Radar-Communication System Using MIMO](https://doi.org/10.1016/j.icte.2015.09.002) | R. Xu et al. | ICT Express | 2015 | — | Early analysis of MIMO-based integrated radar-communication system capacity and mutual information for joint sensing-communication. |
| [Waveform Design and Signal Processing Aspects for Fusion of Wireless Communications and Radar Sensing](https://ieeexplore.ieee.org/document/5776640/) | C. Sturm, W. Wiesbeck | Proc. IEEE | 2011 | — | Foundational paper on waveform design theory for wireless communication and radar sensing fusion, covering basic theoretical principles. |
| [Fundamental Limits for ISAC: Information and Communication Theoretic Perspective](https://scholar.google.com/scholar?q=Fundamental+limits+for+isac+information+and+communication+theoretic) | A. Liu et al. | Springer | 2023 | — | Provides an information-theoretic perspective on fundamental limits for ISAC systems. |

## Waveform Design Theory

This subsection covers theoretical approaches to ISAC waveform design, including information-theoretic waveform optimization, MIMO-OFDM waveform theory, and multi-metric optimization frameworks.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Waveform Design for MIMO-OFDM ISAC: An Information Theoretical Approach](https://ieeexplore.ieee.org/document/10327127) | Z. Wei et al. | IEEE TCOM | 2024 | — | Develops information-theoretic waveform design for MIMO-OFDM ISAC systems, optimizing sensing and communication performance jointly. |
| [Multi-Metric Waveform Optimization for MISO Joint Communication and Radar Sensing](https://ieeexplore.ieee.org/document/9657750) | Z. Ni et al. | IEEE TCOM | 2022 | — | Proposes multi-metric optimization for MISO ISAC waveform design, balancing radar beampattern, rate, and sensing performance. |
| [Generalized Transceiver Beamforming for DFRC with MIMO Radar and MU-MIMO Communication](https://ieeexplore.ieee.org/document/9775334) | L. Chen et al. | IEEE JSAC | 2022 | — | Develops generalized transceiver beamforming theory for DFRC with MIMO radar and multi-user MIMO communication. |
| [Joint Transmit Beamforming for Multiuser MIMO Communications and MIMO Radar](https://ieeexplore.ieee.org/document/9089249) | X. Liu et al. | IEEE TSP | 2020 | — | Provides theoretical foundations for joint transmit beamforming in multiuser MIMO communication and MIMO radar co-design. |
| [Optical ISAC: Fundamental Performance Limits and Transceiver Design](https://scholar.google.com/scholar?q=Optical+isac+fundamental+performance+limits) | A. G. Khorasgani et al. | — | 2024 | — | Derives Bayesian CRB and fundamental performance limits for optical ISAC systems. |

---

*Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md).*
