# 🔒 Security & Privacy

> The integration of sensing and communication (S&C) functionalities in ISAC systems introduces unique security and privacy challenges across dual domains. While high-power sensing signals enhance target detection, they simultaneously create vulnerabilities for both communication data interception and sensing-based privacy invasion. Conversely, the shared infrastructure also enables new defense mechanisms, such as sensing-aided physical layer security and covert communication, turning the dual-use nature of ISAC from a liability into an opportunity for enhanced protection. This section surveys the evolving landscape of ISAC security, covering communication security, sensing privacy, and cross-domain co-design approaches.

## Physical Layer Security for ISAC

Physical layer security (PLS) leverages the inherent randomness of wireless channels to protect communication data without relying solely on upper-layer encryption. In ISAC systems, traditional secure beamforming must be redesigned to avoid compromising sensing performance, leading to novel joint optimization of communication secrecy and radar functionality.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Secure Radar-Communication Systems with Malicious Targets: Integrating Radar, Communications and Jamming Functionalities](https://ieeexplore.ieee.org/document/9229861) | N. Su, F. Liu, C. Masouros et al. | IEEE TWC | 2021 | — | AN-aided secure beamforming for multi-user MISO ISAC maximizing secrecy rate |
| [Secure Dual-Functional Radar-Communication Transmission: Exploiting Interference for Resilience against Target Eavesdropping](https://ieeexplore.ieee.org/document/9634480) | N. Su, F. Liu, Z. Wei et al. | IEEE TWC | 2022 | — | CI-DI approach at mmWave for directional modulation-based PLS in ISAC |
| [Securing the Sensing Functionality in ISAC Networks: An Artificial Noise Design](https://ieeexplore.ieee.org/document/10608053) | J. Zou, C. Masouros, F. Liu, S. Sun | IEEE TVT | 2024 | — | AN covariance optimization to protect sensing data from unauthorized receivers |
| [Robust Transmit Beamforming for Secure Integrated Sensing and Communication](https://ieeexplore.ieee.org/document/10217363) | Z. Ren, L. Qiu, J. Xu et al. | IEEE TC | 2023 | — | Robust beamforming under CSI uncertainty for secure ISAC |
| [On Privacy, Security, and Trustworthiness in Distributed Wireless Large AI Models](https://arxiv.org/abs/2412.02538) | Z. Yang, W. Xu, L. Liang, Y. Cui, Z. Qin, M. Debbah | Sci. China Inf. Sci. | 2025 | — | Privacy-security-trust framework for distributed wireless AI models |

## Sensing-Aided Secure Communication

ISAC's dual functionality enables a paradigm shift: the sensing capability can dynamically estimate the wiretap channel by tracking eavesdropper locations, overcoming the classical challenge of obtaining eavesdropper CSI in conventional communication-only systems. This allows real-time adaptation of secure beamforming strategies.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Sensing-Enhanced Secure Communication: Joint Time Allocation and Beamforming Design](https://arxiv.org/abs/2312.15231) | D. Xu, Y. Xu, Z. Wei, S. Song, D. W. K. Ng | WiOpt | 2023 | — | Monotonic optimization for globally optimal sensing-aided secrecy beamforming |
| [Sensing-Aided Near-Field Secure Communications with Mobile Eavesdroppers](https://arxiv.org/abs/2408.13829) | Y. Xu, M. Zheng, D. Xu, S. Song, D. B. da Costa | IEEE TWC | 2025 | — | Near-field PLS design with mobility-aware eavesdropper tracking |
| [Optimal Transmit Beamforming for Secrecy Integrated Sensing and Communication](https://arxiv.org/abs/2110.12945) | Z. Ren, L. Qiu, J. Xu | IEEE ICC | 2022 | — | Joint information/sensing beam adjustment to confuse eavesdroppers |

## Sensing Privacy Risks

Sensing signals emitted by ISAC base stations reflect off targets and may be intercepted by unauthorized receivers, exposing sensitive information such as location, movement patterns, and physiological data. Unlike communication signals that benefit from encryption, passive sensing signals cannot be effectively encrypted, presenting a unique vulnerability.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Privacy and Security in Ubiquitous Integrated Sensing and Communication: Threats, Challenges and Future Directions](https://arxiv.org/abs/2308.00253) | K. Qu, J. Ye, X. Li, S. Guo | IEEE IoT Mag. | 2024 | — | Comprehensive survey on ISAC privacy threats and attack surfaces |
| [Multi-Antenna Signal Masking and Round-Trip Transmission for Privacy-Preserving Wireless Sensing](https://ieeexplore.ieee.org/document/10556255) | Y. Wang, L. Sun, Q. Du | IEEE TIFS | 2024 | — | CSI time-correlation exploitation to thwart location eavesdropping |
| [PriSense: Privacy-Preserving Wireless Sensing for Vital Signs Monitoring](https://ieeexplore.ieee.org/document/10603563) | Y. Wang, L. Sun, Q. Du, M. Elkashlan | IEEE WCL | 2024 | — | Pilot-shielding method with spectral masking for vital sign privacy |

## Privacy-Preserving ISAC

Privacy-preserving ISAC aims to prevent unauthorized sensing and data extraction while maintaining legitimate S&C functionality. Key approaches include AI-assisted adaptive beamforming to obfuscate signals, generative AI for secure sensing graph construction, and signal-masking techniques that preserve operational accuracy.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Generative AI Based Secure Wireless Sensing for ISAC Networks](https://arxiv.org/abs/2408.11398) | J. Wang, H. Du, Y. Liu, G. Sun, D. Niyato et al. | IEEE TIFS | 2025 | — | Diffusion model-based secure sensing with 70% inference risk reduction |
| [Privacy and Security in Ubiquitous ISAC: Threats, Challenges and Future Directions](https://arxiv.org/abs/2308.00253) | K. Qu, J. Ye, X. Li, S. Guo | IEEE IoT Mag. | 2024 | — | AI-assisted adaptive beamforming to block unauthorized sensing |

## Dual-Domain Security Co-Design

Dual-domain security co-design recognizes that communication security and sensing privacy are deeply intertwined in ISAC systems. Integrated optimization frameworks must balance secrecy rate, sensing accuracy, and privacy protection simultaneously, accounting for the unique vulnerabilities and opportunities that arise from shared S&C infrastructure.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [IRS-UAV Assisted Secure Integrated Sensing and Communication](https://ieeexplore.ieee.org/document/10563387) | J. Xu, X. Yu, L. Xu, C. Xing, N. Zhao, X. Wang, D. Niyato | IEEE Wireless Commun. | 2024 | — | IRS-enhanced dual-domain security with UAV trajectory optimization |
| [Secure ISAC (S-ISAC) Network](https://ieeexplore.ieee.org/document/10685324) | T. Guo, H. Li | IEEE VTC-Fall | 2024 | — | Cross-domain secure ISAC framework addressing hybrid threats |
| [Blockchain-Based Credential Management for Anonymous Authentication in SAGVN](https://ieeexplore.ieee.org/document/9786329) | D. Liu et al. | IEEE JSAC | 2022 | — | Blockchain authentication for space-air-ground-vehicle ISAC networks |

## Adversarial Attacks on ISAC

Adversarial attacks on ISAC systems can exploit both sensing and communication domains simultaneously. Jammer-assisted covert communication and adversarial sensing deception represent emerging threat models that require hybrid defense strategies accounting for coordinated cross-domain manipulation.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Jammer-Assisted Joint Power Allocation and Beamforming for Covert Wireless Transmission in ISAC](https://ieeexplore.ieee.org/document/10468926) | Y. Wang, X. Ni, W. Yang, H. Fan, Z. Yuan, H. Shi | ICCC | 2023 | — | SDR-based covert throughput maximization with radar detection constraints |
| [Sensing-Assisted Covert Communication: Exploiting ISAC Duality](https://ieeexplore.ieee.org/document/10685324) | Y. Wang et al. | IEEE Trans. | 2024 | — | Sensing-assisted transmission undetectability via beamforming optimization |

---

*Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md).*
