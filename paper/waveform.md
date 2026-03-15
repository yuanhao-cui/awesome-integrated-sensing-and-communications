# 📡 Waveform Design

> ISAC waveform design: from orthogonal resource allocation (time/frequency/spatial/code division) to unified joint waveforms, OFDM, OTFS, FMCW, index modulation, and non-orthogonal approaches.

Waveform design is the cornerstone of ISAC system implementation. Early ISAC systems relied on orthogonal resource allocation—separating sensing and communication in time, frequency, space, or code domains to avoid mutual interference. This approach offers simple implementation but sacrifices spectral efficiency. The field has progressively evolved toward unified (non-orthogonal) waveforms where sensing and communication share resources simultaneously, leveraging advanced signal processing to manage interference while maximizing integration and coordination gains. Key waveform families include OFDM (widely adopted in 5G), OTFS (robust to high Doppler), FMCW/Chirp (automotive radar heritage), and index modulation schemes, each offering distinct trade-offs between sensing accuracy, communication rate, and implementation complexity.

---

## Time-Division / Orthogonal Resource Allocation

Time-division approaches allocate separate time slots for sensing and communication waveforms, providing clean separation at the cost of temporal efficiency. These methods are straightforward to implement and avoid inter-function interference, making them well-suited for systems with relaxed latency requirements.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Time-Division ISAC Enabled Connected Automated Vehicles Cooperation](https://ieeexplore.ieee.org/document/9762746) | Q. Zhang et al. | IEEE JSAC | 2022 | — | Proposes time-division ISAC for connected automated vehicles, enabling cooperative perception and communication in vehicular networks. |
| [Integrated Time-Division JRC with Federated Edge Learning](https://ieeexplore.ieee.org/document/10476949) | P. Liu et al. | IEEE JSTSP | 2023 | — | Integrates time-division joint radar-communication with federated edge learning for faster convergence and higher accuracy in energy-limited scenarios. |
| [Optimal Scheduling Policy for Time-Division JRC: Cross-Layer Design](https://ieeexplore.ieee.org/document/10476949) | Z. Xie et al. | IEEE IoTJ | 2023 | — | Develops cross-layer scheduling optimization for time-division JRC, balancing delay, power consumption, and detection accuracy. |
| [Time-Division Based ISAC in Integrated Satellite-Terrestrial Networks](https://scholar.google.com/scholar?q=Time-division+based+integrated+sensing+communication+and+computing) | X. Zhu et al. | Digital Signal Process. | 2023 | — | Extends time-division ISAC to integrated satellite-terrestrial networks for seamless sensing-communication coverage. |

## Frequency-Division & Subcarrier Allocation

Frequency-division approaches assign different frequency bands or subcarriers to sensing and communication, leveraging the flexibility of OFDMA-based systems. This method is compatible with existing 5G infrastructure but may underutilize spectrum when one function requires fewer resources.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Power Minimization Strategy for ISAC: Subcarrier Allocation and Power Assignment](https://scholar.google.com/scholar?q=Power+minimization+strategy+based+subcarrier+allocation+integrated+sensing) | J. Zhu et al. | IEEE WCNC | 2023 | — | Optimizes subcarrier allocation and power assignment across 128 subcarriers for ISAC power minimization. |
| [A Dual Function Compromise for Uplink ISAC: Joint Spectrum and Power Management](https://scholar.google.com/scholar?q=A+dual+function+compromise+for+uplink+isac) | Y. Li et al. | IEEE WCNC | 2024 | — | Addresses dual-function compromise in uplink ISAC through joint spectrum and power management. |
| [Joint Subcarrier Assignment and Power Allocation for ISAC Based on Power Minimization](https://ieeexplore.ieee.org/document/8803777) | C. Shi et al. | IEEE Sensors J. | 2019 | — | Proposes joint subcarrier assignment and power allocation for ISAC with power minimization objective. |
| [OFDM-Based DFRC: Optimal Resource Allocation for Fairness](https://scholar.google.com/scholar?q=OFDM-based+dual-function+radar-communications+optimal+resource+allocation) | J. Zhu et al. | IEEE VTC | 2022 | — | Studies fair resource allocation for OFDM-based dual-function radar-communications. |

## Spatial Division & Beamforming

Spatial division leverages MIMO beamforming to serve communication users and perform sensing simultaneously in the spatial domain. By exploiting spatial degrees of freedom, this approach achieves high spectral efficiency, though it requires sophisticated hardware and signal processing.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Joint Radar-Communication Transmission: A Generalized Pareto Optimization Framework](https://ieeexplore.ieee.org/document/9414894) | L. Chen et al. | IEEE TSP | 2021 | — | Proposes a Pareto optimization framework for joint radar-communication beamforming, converging in 6-7 iterations under varied SINR constraints. |
| [Joint Transmit Beamforming for Multiuser MIMO Communications and MIMO Radar](https://ieeexplore.ieee.org/document/9089249) | X. Liu et al. | IEEE TSP | 2020 | — | Develops joint transmit beamforming for multiuser MIMO communication and MIMO radar co-design. |
| [Joint Transmit Beamforming and Receive Filters Design for Two-Cell Interfering DFRC](https://ieeexplore.ieee.org/document/9843549) | Y. Li, M. Jiang | IEEE TVT | 2022 | — | Addresses multi-cell beamforming for dual-function radar-communication with interference mitigation. |
| [Energy-Efficient Beamforming Design for ISAC Systems](https://ieeexplore.ieee.org/document/10393498) | J. Zou et al. | IEEE TCOM | 2024 | — | Proposes energy-efficient beamforming design balancing sensing accuracy and communication throughput. |
| [Constant-Modulus Waveform Design for DFRC in the Presence of Clutter](https://ieeexplore.ieee.org/document/10003287) | W. Wu et al. | IEEE TAES | 2023 | — | Designs constant-modulus waveforms for DFRC systems under clutter environments and hardware constraints. |
| [Closed-Form Solutions and Iterative Algorithms for Multi-Antenna DFRC](https://ieeexplore.ieee.org/document/9414894) | L. Chen et al. | IEEE TSP | 2021 | — | Provides both closed-form and iterative solutions for multi-antenna DFRC beamforming optimization. |
| [Deep CLSTM for Predictive Beamforming in ISAC-Enabled Vehicular Networks](https://scholar.google.com/scholar?q=Deep+clstm+for+predictive+beamforming+in+integrated+sensing+and+communication) | C. Liu et al. | JCIN | 2022 | — | Uses deep CLSTM networks for predictive beamforming in ISAC-enabled vehicular communication networks. |

## OFDM-Based ISAC

OFDM is the most extensively studied waveform for ISAC due to its prevalence in 5G NR and its inherent suitability for both high-data-rate communication and range-Doppler sensing. Research focuses on optimizing subcarrier allocation, sidelobe suppression, and hybrid beamforming.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Code-Division OFDM JRC System for 6G Machine-Type Communication](https://ieeexplore.ieee.org/document/9439997) | X. Chen et al. | IEEE IoTJ | 2021 | — | Proposes CD-OFDM for joint radar-communication in 6G machine-type communication with 30.1 dB gain in low SINR regime. |
| [Bandwidth Efficient Dual-Function Radar Communication Based on MIMO Radar Using OFDM](https://ieeexplore.ieee.org/document/9957609) | Z. Xu, A. Petropulu | IEEE TSP | 2023 | — | Designs bandwidth-efficient MIMO OFDM radar-communication system with optimized subcarrier utilization. |
| [MIMO OFDM Dual-Function Radar-Communication Under Error Rate and Beampattern Constraints](https://ieeexplore.ieee.org/document/9786329) | J. Johnston et al. | IEEE JSAC | 2022 | — | Addresses MIMO OFDM DFRC under error rate and beampattern constraints for practical deployment. |
| [Hybrid Beamforming for OFDM Dual-Function Radar-Communication System](https://ieeexplore.ieee.org/document/9503844) | Z. Cheng et al. | IEEE JSTSP | 2021 | — | Proposes hybrid analog-digital beamforming for OFDM-based DFRC systems. |
| [Joint Communication and Sensing System Performance Evaluation and Testbed](https://ieeexplore.ieee.org/document/10618220) | Q. Zhang et al. | IEEE Network | 2024 | — | Presents an OFDM ISAC system testbed with performance evaluation in practical scenarios. |
| [Device-Free Sensing in OFDM Cellular Network](https://ieeexplore.ieee.org/document/9775311) | Q. Shi et al. | IEEE JSAC | 2022 | — | Explores device-free sensing capabilities in OFDM cellular networks using existing communication infrastructure. |
| [A Dual-Functional Sensing-Communication Waveform Design Based on OFDM](https://ieeexplore.ieee.org/document/10608648) | Y. He et al. | IEEE TWC | 2024 | — | Designs dual-functional OFDM waveform optimizing both sensing accuracy and communication rate. |
| [Joint Beamforming Design in DFRC Systems for Wideband Sensing and OFDM Communications](https://scholar.google.com/scholar?q=Joint+beamforming+design+in+dfrc+systems+for+wideband+sensing) | Z. Xiao et al. | IEEE GLOBECOM | 2022 | — | Addresses joint beamforming for wideband sensing and OFDM communication in DFRC systems. |
| [MIMO-OFDM ISAC Waveform Design for Range-Doppler Sidelobe Suppression](https://ieeexplore.ieee.org/document/10794178) | P. Li et al. | IEEE TWC | 2025 | — | Optimizes MIMO-OFDM waveform for sidelobe suppression in range-Doppler domain, enhancing sensing accuracy. |

## OTFS-Based ISAC

OTFS (Orthogonal Time Frequency Space) modulation operates in the delay-Doppler domain, offering inherent robustness against high Doppler shifts—making it particularly suitable for high-mobility ISAC scenarios such as vehicular and UAV networks.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [On the Effectiveness of OTFS for Joint Radar Parameter Estimation and Communication](https://ieeexplore.ieee.org/document/9098919) | L. Gaudio et al. | IEEE TWC | 2020 | — | Establishes the effectiveness of OTFS for joint radar parameter estimation and communication in high-Doppler environments. |
| [ISAC-Assisted OTFS Transmission for Vehicular Networks](https://ieeexplore.ieee.org/document/9503827) | W. Yuan et al. | IEEE JSTSP | 2021 | — | Proposes ISAC-assisted OTFS transmission for vehicular networks, leveraging sensing to enhance communication in high-mobility scenarios. |
| [From OTFS to DD-ISAC: Integrating Sensing and Communications in the Delay Doppler Domain](https://arxiv.org/abs/2311.15215) | W. Yuan et al. | arXiv | 2023 | — | Proposes a delay-Doppler domain ISAC framework extending OTFS for integrated sensing and communications. |
| [Joint Subcarrier and Power Allocation for OTFS-Based ISAC](https://scholar.google.com/scholar?q=Joint+subcarrier+and+power+allocation+for+otfs+based+integrated+sensing) | S. Liu et al. | — | 2023 | — | Addresses subcarrier and power allocation for OTFS-based ISAC systems. |
| [Dual-Functional Waveform Design with Local Sidelobe Suppression via OTFS Signaling](https://ieeexplore.ieee.org/document/10476949) | K. Zhang et al. | IEEE TVT | 2024 | — | Designs OTFS-based dual-functional waveforms with local sidelobe suppression for improved sensing performance. |
| [A Novel ISAC Transmission Framework Based on Spatially-Spread OTFS](https://ieeexplore.ieee.org/document/9774258) | S. Li et al. | IEEE JSAC | 2022 | — | Proposes spatially-spread OTFS for ISAC, exploiting spatial and delay-Doppler diversity jointly. |

## FMCW / Chirp-Based

FMCW (Frequency Modulated Continuous Wave) and chirp-based waveforms are widely used in automotive radar and are being adapted for ISAC by embedding communication data into radar signals. These approaches naturally inherit radar's proven sensing capabilities.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [FRAC: FMCW-Based Joint Radar-Communications via Index Modulation](https://ieeexplore.ieee.org/document/9503826) | D. Ma et al. | IEEE JSTSP | 2021 | — | Proposes FRAC system embedding communication data into FMCW radar via index modulation for automotive ISAC. |
| [Adaptive Waveform Design for Automotive JRC Systems](https://ieeexplore.ieee.org/document/9401938) | S. H. Dokhanchi et al. | IEEE TVT | 2021 | — | Designs adaptive FMCW waveforms for automotive joint radar-communication systems. |
| [Cooperative SAR-Communication System Using CPM Codes and Mismatched Filters](https://ieeexplore.ieee.org/document/10476949) | M.-E. Chatzitheodoridi et al. | IEEE TGRS | 2023 | — | Combines CPM codes with mismatched filters for cooperative SAR-communication systems. |
| [A Cooperative SAR-Communication System Using CPM Codes](https://ieeexplore.ieee.org/document/10476949) | M.-E. Chatzitheodoridi et al. | IEEE TGRS | 2023 | — | Proposes CPM-coded waveforms for cooperative SAR-communication, achieving enhanced image quality and data rates up to 5 Mbit/s. |

## Unified / Joint Waveform Design

Unified waveform design aims to create single waveforms that simultaneously serve both sensing and communication without domain separation. These approaches maximize resource utilization but require advanced optimization and interference management techniques.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Joint Radar-Communications with Cyclic Prefixed Single Carrier Waveforms](https://ieeexplore.ieee.org/document/8970697) | Y. Zeng et al. | IEEE TVT | 2020 | — | Proposes CP-SC waveforms for joint radar-communications with low PAPR and efficient equalization. |
| [Multi-Metric Waveform Optimization for MISO JRC](https://ieeexplore.ieee.org/document/9657750) | Z. Ni et al. | IEEE TCOM | 2022 | — | Optimizes MISO joint radar-communication waveforms across multiple performance metrics simultaneously. |
| [Complementary Coded Scrambling RadCom System](https://ieeexplore.ieee.org/document/10336885) | X. Liu et al. | IEEE TVT | 2024 | — | Proposes complementary coded scrambling for radar-communication systems with improved interference resilience. |
| [Transmit Waveform Design for DFRC via Hybrid Linear-Nonlinear Precoding](https://ieeexplore.ieee.org/document/10122677) | C. Wen et al. | IEEE TSP | 2023 | — | Designs DFRC waveforms using hybrid linear-nonlinear precoding for enhanced flexibility. |
| [Joint Receiver Design for ISAC](https://ieeexplore.ieee.org/document/10152830) | Y. Dong et al. | IEEE Commun. Lett. | 2023 | — | Proposes joint receiver design for ISAC systems, enabling simultaneous sensing and communication decoding. |
| [Unimodular Waveform Design for Integrated Radar Communication and Jamming](https://scholar.google.com/scholar?q=Unimodular+waveform+design+for+integrated+radar+communication+and+jamming) | C. Huang et al. | Digital Signal Process. | 2023 | — | Designs unimodular waveforms for triple-function radar, communication, and jamming integration. |
| [PAPR-Aware Joint Waveform Design for UAV-Enabled ISAC](https://scholar.google.com/scholar?q=PAPR-aware+joint+waveform+design+of+uav-enabled+integrated+sensing) | M. Gu et al. | — | 2024 | — | Addresses PAPR-aware waveform design for UAV-enabled ISAC systems. |
| [An Experimental Proof of Concept for ISAC Waveform Design](https://ieeexplore.ieee.org/document/10476949) | T. Xu et al. | IEEE OJCS | 2022 | — | Provides experimental validation of ISAC waveform design concepts in practical hardware. |

## Index Modulation & Spatial Modulation

Index modulation embeds information in the activation patterns of subcarriers, antennas, or other signal resources, offering a low-complexity approach to communication embedding in radar waveforms.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [MajorCom: A Dual-Function Radar Communication System Using Index Modulation](https://ieeexplore.ieee.org/document/9089385) | T. Huang et al. | IEEE TSP | 2020 | — | Proposes MajorCom system embedding communication via index modulation in radar waveforms. |
| [Hybrid Index Modulation for Dual-Functional Radar Communications](https://ieeexplore.ieee.org/document/9910576) | J. Xu et al. | IEEE TVT | 2023 | — | Develops hybrid index modulation scheme combining multiple IM dimensions for enhanced data rates. |
| [Spatial Modulation for Joint Radar-Communications: Design, Analysis, and Hardware Prototype](https://ieeexplore.ieee.org/document/9329065) | D. Ma et al. | IEEE TVT | 2021 | — | Implements spatial modulation for JRC with hardware prototype validation, demonstrating practical feasibility. |
| [Code-Division CD-OFDM with Successive Interference Cancellation](https://ieeexplore.ieee.org/document/9439997) | X. Chen et al. | IEEE IoTJ | 2021 | — | Proposes CD-OFDM with SIC for JRC, achieving 30.1 dB gain in low SINR with comparable sensing accuracy. |

## Sensing-Communication Receiver & Processing

This subsection covers receiver design and signal processing techniques for jointly extracting sensing and communication information from ISAC waveforms.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Reduced Complexity Maximum SINR Receiver Processing for Transmit-Encoded Radar-Embedded Communications](https://scholar.google.com/scholar?q=Reduced+complexity+maximum+sinr+receiver+processing) | C. Sahin et al. | IEEE RadarConf | 2018 | — | Proposes low-complexity maximum SINR receiver for radar-embedded communication signals. |
| [Joint Radar and Communications for Frequency-Hopped MIMO Systems](https://ieeexplore.ieee.org/document/9657750) | W. Baxter et al. | IEEE TSP | 2022 | — | Designs joint radar-communication processing for frequency-hopped MIMO systems. |
| [Dual-Function MIMO Radar Communications via Sparse Array Optimization](https://ieeexplore.ieee.org/document/8254787) | X. Wang et al. | IEEE TAES | 2018 | — | Optimizes sparse arrays for dual-function MIMO radar-communications. |
| [Dual-Function Radar-Communications: Information Embedding Using Sidelobe Control](https://ieeexplore.ieee.org/document/7010774) | A. Hassanien et al. | IEEE TSP | 2015 | — | Embeds communication information by controlling radar beam sidelobes, an early pioneering approach to DFRC. |

---

*Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md).*
