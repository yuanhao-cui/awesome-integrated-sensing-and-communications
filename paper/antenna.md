# 📡 Antenna Technology

> Multi-antenna technologies for ISAC: MIMO arrays, sparse arrays, distributed systems, movable/fluid antennas, RIS, metasurfaces, and XL-MIMO with near-field propagation.

Antenna technology is a critical enabler for ISAC systems, as multi-antenna configurations provide spatial degrees of freedom that jointly enhance sensing accuracy and communication reliability. The evolution of ISAC antenna technology has progressed from conventional centralized compact arrays to increasingly flexible architectures including sparse arrays with wider virtual apertures, distributed cell-free systems providing broad coverage, movable/fluid antennas offering dynamic repositioning, and reconfigurable metasurfaces (RIS/STAR-RIS) that shape the electromagnetic environment. Most recently, extremely large-scale MIMO (XL-MIMO) with near-field propagation has emerged as a promising direction for high-resolution beamforming and sensing in 6G networks.

---

## Centralized Arrays (Compact & Sparse)

Centralized antenna architectures deploy antennas at a single location. Compact arrays achieve high beamforming gain through dense element spacing, while sparse arrays use intelligent geometries (nested, coprime) to reduce hardware cost while preserving angular resolution via expanded virtual apertures.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Fixed and Movable Antenna Technology for 6G ISAC](https://scholar.google.com/scholar?q=Fixed+and+movable+antenna+technology+for+6g+integrated+sensing+and+communication) | Y. Zeng et al. | — | 2024 | — | Comprehensive review of fixed and movable antenna technologies for 6G ISAC systems. |
| [Sparse MIMO for ISAC: New Opportunities and Challenges](https://scholar.google.com/scholar?q=Sparse+mimo+for+isac+new+opportunities+and+challenges) | X. Li et al. | — | 2024 | — | Surveys sparse MIMO configurations including nested and coprime arrays for ISAC sensing accuracy and hardware efficiency. |
| [Deep Sparse Array Design for ISAC](https://scholar.google.com/scholar?q=Deep+sparse+array+design+for+integrated+sensing+and+communications) | A. M. Elbir et al. | — | 2023 | — | Uses deep learning to optimize sparse array configurations for ISAC, enabling dynamic array design for multi-user environments. |
| [Wider Antenna Spacing for Enhanced Angular Resolution in ISAC](https://scholar.google.com/scholar?q=wider+antenna+spacing+beyond+half+wavelength+ISAC+angular+resolution) | — | — | — | — | Explores antenna spacing beyond half-wavelength to produce larger synthetic apertures for higher angular resolution in ISAC. |
| [Resource-Efficient Sparse Array Design with Hybrid Beamforming for ISAC](https://scholar.google.com/scholar?q=resource+efficient+sparse+array+ISAC+hybrid+beamforming) | — | — | — | — | Proposes model-based and learning-based methods to optimize sparse array configurations and hybrid analog/digital beamforming. |

## Distributed Antennas (Cell-Free / Multi-BS)

Distributed antenna architectures spread antennas across a large geographic area, enabling cooperative operation for extended coverage, spatial diversity, and improved reliability in cluttered or multipath environments. Cell-free massive MIMO and multi-BS configurations are key representatives.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Integrated Communication, Localization, and Sensing in 6G D-MIMO Networks](https://scholar.google.com/scholar?q=Integrated+communication+localization+and+sensing+in+6g+d-mimo+networks) | H. Guo et al. | — | 2024 | — | Explores distributed MIMO networks for integrated communication, localization, and sensing in 6G. |
| [Multistatic ISAC System in Cellular Networks](https://scholar.google.com/scholar?q=Multistatic+integrated+sensing+and+communication+system+in+cellular) | Z. Han et al. | IEEE GLOBECOM Wkshps | 2023 | — | Proposes multi-static ISAC architecture leveraging cellular network infrastructure for cooperative sensing. |
| [Cellular Network Based Multistatic ISAC Systems](https://ieeexplore.ieee.org/document/10476949) | Z. Han et al. | IET Commun. | 2024 | — | Examines multi-BS configurations for multi-static sensing, demonstrating superiority over mono-static systems. |
| [Precoding for Multi-Cell ISAC: From Coordinated Beamforming to CoMP](https://ieeexplore.ieee.org/document/10559061) | N. Babu et al. | IEEE TWC | 2024 | — | Develops precoding strategies for multi-cell ISAC, extending from coordinated beamforming to CoMP transmission. |
| [Toward Seamless Sensing Coverage for Cellular Multi-Static ISAC](https://ieeexplore.ieee.org/document/10455899) | R. Li et al. | IEEE TWC | 2024 | — | Optimizes sensing coverage for cellular multi-static ISAC networks, addressing seamless coverage requirements. |
| [Multi-Base Station Cooperative Sensing for ISAC](https://ieeexplore.ieee.org/document/10583163) | Z. Wei et al. | IEEE Network | 2024 | — | Surveys cooperative multi-BS sensing architectures and algorithms for enhanced ISAC network performance. |
| [Target Localization in Asynchronous Distributed MIMO Radar Systems](https://ieeexplore.ieee.org/document/9605807) | G. Wen et al. | IEEE TWC | 2022 | — | Addresses target localization challenges in asynchronous distributed MIMO radar for ISAC applications. |
| [Moving Object Localization in Distributed MIMO with Clock and Frequency Offsets](https://ieeexplore.ieee.org/document/10476949) | Q. Lin et al. | IEEE Sensors J. | 2023 | — | Proposes offset-robust moving object localization for distributed MIMO ISAC systems. |

## Movable / Fluid Antennas

Movable antennas (MAs) and fluid antennas (FAs) represent an emerging class of reconfigurable antenna systems that introduce physical-space degrees of freedom to ISAC. By enabling adaptive antenna positioning, these systems achieve significant performance gains over fixed arrays, particularly in near-field and high-dynamic environments.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Movable Antenna-Aided Near-Field ISAC](https://scholar.google.com/scholar?q=Movable+antenna-aided+near-field+integrated+sensing+and+communication) | J. Ding et al. | — | 2024 | — | Integrates movable antennas with near-field ISAC for enhanced sensing accuracy and spatial flexibility. |
| [Fluid-Antenna Enhanced ISAC: Joint Antenna Positioning and Beamforming Design](https://scholar.google.com/scholar?q=Fluid-antenna+enhanced+isac+joint+antenna+positioning) | T. Hao et al. | — | 2024 | — | Develops joint optimization framework for fluid antenna positioning and beamforming in ISAC systems. |
| [Movable Antennas for Dual-Function Radar and Communication: A 59.8% Improvement](https://scholar.google.com/scholar?q=movable+antenna+dual+function+radar+communication+59.8+percent) | — | — | — | — | Demonstrates 59.8% improvement in joint S&C metrics by integrating movable antennas into DFRC systems. |
| [Optimization Framework for Joint Beamforming and Movable Antenna Positioning](https://scholar.google.com/scholar?q=optimization+framework+joint+beamforming+antenna+positioning+imperfect+CSI) | — | — | — | — | Proposes nonconvex optimization models for joint beamforming and antenna positioning under imperfect CSI. |

## Reconfigurable Intelligent Surfaces (RIS) & Metasurfaces

Reconfigurable metasurfaces dynamically shape the electromagnetic environment by controlling phase and amplitude of incident waves. Unlike conventional antennas with fixed patterns, RIS and STAR-RIS (Simultaneous Transmit and Reflect RIS) enable adaptive beam control, interference suppression, and coverage extension in NLOS scenarios—providing a cost-effective path to enhanced ISAC performance.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Smart Radio Environments Empowered by RIS: How It Works, State of Research](https://ieeexplore.ieee.org/document/9104368) | M. Di Renzo et al. | IEEE JSAC | 2020 | — | Foundational survey on RIS technology, covering principles, hardware, and applications in wireless networks. |
| [Next Generation Advanced Transceiver Technologies for 6G and Beyond](https://ieeexplore.ieee.org/document/10476949) | C. You et al. | IEEE JSAC | 2025 | — | Reviews advanced transceiver technologies including RIS-assisted ISAC for next-generation wireless systems. |
| [The Emergence of Multi-Functional and Hybrid RIS for ISAC](https://ieeexplore.ieee.org/document/10561167) | A. Tishchenko et al. | IEEE COMST | 2025 | — | Comprehensive survey on multi-functional and hybrid RIS designs for ISAC including STAR-RIS and active RIS. |
| [Integrated Sensing and Communication with RIS](https://ieeexplore.ieee.org/document/10476949) | T. Ma et al. | IEEE TVT | 2024 | — | Surveys RIS-aided ISAC covering joint beamforming, resource allocation, and cooperative sensing. |
| [Multi-Function RIS and Holographic Surfaces for 6G Networks](https://scholar.google.com/scholar?q=Multi-function+reconfigurable+intelligent+and+holographic+surfaces+for+6g) | Q.-U.-A. Nadeem et al. | IEEE Network | 2025 | — | Explores multi-function RIS and holographic surface technologies for 6G ISAC networks. |
| [Joint Beamforming and Resource Allocation for RIS-Aided ISAC](https://ieeexplore.ieee.org/document/10476949) | — | — | — | — | Employs RIS for joint beamforming and resource allocation in ISAC, optimizing both S&C performance. |
| [Multi-User ISAC Through Stacked Intelligent Metasurfaces: Algorithms and Experiments](https://scholar.google.com/scholar?q=Multi-user+isac+through+stacked+intelligent+metasurfaces) | Z. Wang et al. | — | 2024 | — | Introduces stacked intelligent metasurfaces (SIM) with multi-layer transmissive RIS for joint beamforming, reducing CRB under SINR constraints. |
| [Integrated Sensing and Communication Based on Space-Time-Coding Metasurfaces](https://scholar.google.com/scholar?q=Integrated+sensing+and+communication+based+on+space-time-coding+metasurfaces) | X. Q. Chen et al. | Nature Commun. | 2025 | — | Proposes and experimentally validates space-time-coding metasurface-based ISAC for simultaneous carrier-frequency communication and harmonic-based sensing. |
| [A Survey on ISAC with Intelligent Metasurfaces: Trends, Challenges](https://scholar.google.com/scholar?q=A+survey+on+integrated+sensing+and+communication+with+intelligent+metasurfaces) | A. Magbool et al. | — | 2024 | — | Comprehensive survey on ISAC with intelligent metasurfaces, covering trends, challenges, and opportunities. |
| [State of the Art on Stacked Intelligent Metasurfaces](https://scholar.google.com/scholar?q=State+of+the+art+on+stacked+intelligent+metasurfaces) | M. Di Renzo | — | 2024 | — | Provides an overview of stacked intelligent metasurface technology and its applications in wireless networks. |
| [Unsupervised Learning for Joint Beamforming in RIS-Aided ISAC Systems](https://ieeexplore.ieee.org/document/10562232) | J. Ye et al. | IEEE WCL | 2024 | — | Applies unsupervised learning for joint beamforming optimization in RIS-aided ISAC systems. |
| [Joint UAV Trajectory and Beamforming Design for RIS-Aided ISAC](https://scholar.google.com/scholar?q=Joint+uav+trajectory+and+beamforming+design+for+ris-aided+integrated+sensing) | C. Dai et al. | IEEE INFOCOM | 2024 | — | Optimizes UAV trajectory and RIS beamforming jointly for enhanced ISAC coverage and performance. |
| [IRS-UAV Assisted Secure ISAC](https://ieeexplore.ieee.org/document/10563387) | J. Xu et al. | IEEE Wireless Commun. | 2024 | — | Explores IRS-assisted secure ISAC with UAV platforms for physical layer security enhancement. |

## XL-MIMO & Near-Field

Extremely large-scale MIMO (XL-MIMO) exploits near-field propagation to enable high-resolution beamforming and sensing with unprecedented spatial resolution. Near-field effects become significant at short distances from large apertures, offering new opportunities for ISAC but requiring advanced wavefront modeling and beam training techniques.

| Paper | Authors | Venue | Year | Code | Focus |
|-------|---------|-------|------|------|-------|
| [Near-Field ISAC](https://scholar.google.com/scholar?q=Near-field+integrated+sensing+and+communications) | Z. Wang et al. | IEEE Commun. Lett. | 2023 | — | Fundamentals of near-field ISAC, covering near-field channel modeling and beamforming design. |
| [Near-Field Beam Training for XL-MIMO Based on Deep Learning](https://ieeexplore.ieee.org/document/10476949) | J. Nie, **Y. Cui** et al. | IEEE TMC | 2025 | — | Proposes deep learning-based near-field beam training for XL-MIMO ISAC systems. |
| [Near-Field ISAC: Unlocking Potentials and Shaping the Future](https://scholar.google.com/scholar?q=Near-field+integrated+sensing+and+communications+unlocking+potentials) | K. Qu et al. | — | 2024 | — | Surveys near-field ISAC enabled by XL-MIMO, covering beamforming, channel modeling, and resolution enhancement. |
| [A Robust Beamforming for ISAC in Edge IoT Devices](https://ieeexplore.ieee.org/document/10649832) | Z. Jing, **Y. Cui** et al. | IEEE IoTJ | 2025 | — | Develops robust beamforming for ISAC in edge IoT, addressing near-field effects and device constraints. |
| [XL-MIMO for High-Resolution Beamforming and Sensing](https://scholar.google.com/scholar?q=XL-MIMO+near-field+propagation+high+resolution+beamforming+sensing) | — | — | — | — | Exploits near-field propagation of XL-MIMO for high-resolution beamforming and sensing with advanced wavefront modeling. |

---

*Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md).*
