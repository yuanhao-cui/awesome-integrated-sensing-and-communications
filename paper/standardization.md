# 📋 Standardization

> ISAC standardization has progressed from conceptual exploration in 3GPP Release 15 to dedicated specifications in Release 19, with 32 identified use cases spanning smart transportation, industry, home, city, and drone scenarios. In parallel, IEEE 802.11bf has frozen the first Wi-Fi sensing standard (October 2024), ITU has incorporated ISAC as a key 6G scenario in IMT-2030, and ETSI has established an ISAC Industrial Specification Group with 18 additional use cases. This section reviews the multi-organization standardization landscape driving ISAC from research toward commercial deployment.

## 3GPP (NR Rel-15–20)

3GPP has progressively incorporated ISAC-enabling functionalities into its release-based framework. The journey spans from foundational technologies in Rel-15 to dedicated ISAC service requirements in Rel-19 (TS 22.137), with 32 identified use cases across five deployment scenarios and ISAC channel modeling studies initiated in Rel-20. ISAC air interface technologies are expected in Rel-21 as part of the 6G standardization cycle.

| Standard/Document | Organization | Year | Focus |
|-------------------|--------------|------|-------|
| [3GPP TS 38.300: NR and NG-RAN Overall Description](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3191) | 3GPP | 2021 | Foundational NR architecture (beamforming, massive MIMO, flexible spectrum) |
| [3GPP TS 38.104: NR Base Station Radio Transmission and Reception](https://www.3gpp.org/ftp/Specs/archive/38_series/38.104/) | 3GPP | 2023 | BS radio requirements enabling ISAC infrastructure |
| [3GPP TR 38.855: Study on NR Positioning Support](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3501) | 3GPP | 2019 | Multi-RTT, UL-AoA, DL-AoD positioning methods for sensing |
| [3GPP TS 22.104: Service Requirements for Cyber-Physical Control (Rel-16)](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3528) | 3GPP | 2019 | 20-30 cm positioning accuracy for factory automation |
| [3GPP TS 38.355: NR Sidelink Positioning Protocol](https://www.3gpp.org/DynaReport/38355.htm) | 3GPP | 2020 | Sidelink positioning for V2X cooperative localization |
| [3GPP TR 22.837: Feasibility Study on ISAC for NR](https://www.3gpp.org/ftp/Specs/html-info/22837.htm) | 3GPP | 2023 | **32 ISAC use cases** across smart transportation, industry, home, city, drones |
| [3GPP TS 22.137: Service Requirements for ISAC](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=4198) | 3GPP SA1#103 | 2024 | Eight KPIs for wireless sensing: positioning accuracy, velocity, resolution |
| [3GPP ISAC Channel Modelling Study Item (RAN)](https://www.3gpp.org/ftp/tsg_ran/WG1_RL1/TSGR1_116b) | 3GPP RAN#102-103 | 2024 | Unified channel models for object detection, tracking, UAV/pedestrian/vehicle |
| [The Bridge Toward 6G: 5G-Advanced Evolution in 3GPP Release 19](https://arxiv.org/abs/2312.15174) | X. Lin | IEEE Commun. Stand. Mag. | 2025 | Rel-19 as bridge to 6G: ISAC study item, RAN intelligence, carrier-phase positioning |
| [Architecture for Cellular Enabled Integrated Communication and Sensing Services](https://ieeexplore.ieee.org/document/10251772/) | B. Liu, Q. Zhang et al. | China Commun. | 2023 | Cellular ISAC service architecture and protocol stack design |

### 3GPP ISAC Use Cases (32 total from TR 22.837)

The 32 use cases identified in TR 22.837 span five deployment scenarios:
- **Smart Transportation**: Traffic monitoring, collision avoidance, road condition sensing, cooperative perception
- **Smart Industry**: Factory automation, asset tracking, predictive maintenance, worker safety monitoring
- **Smart Home**: Indoor positioning, gesture recognition, health monitoring, elderly fall detection
- **Smart City**: Environmental monitoring, crowd management, infrastructure inspection, public safety
- **Smart Drones**: UAV detection, trajectory tracking, obstacle avoidance, low-altitude airspace management

## IEEE (802.11bf / 802.15.4ab)

IEEE has advanced ISAC standardization through Wi-Fi sensing (802.11bf) and ultra-wideband enhancements (802.15.4ab). The 802.11bf standard, frozen in October 2024, integrates sensing into the existing Wi-Fi ecosystem by leveraging preamble training fields and pilot tones for environmental inference while maintaining backward compatibility.

| Standard | Organization | Year | Focus |
|----------|--------------|------|-------|
| [IEEE 802.11bf: Wi-Fi Sensing](https://ieeexplore.ieee.org/document/10476949) | IEEE 802.11 TGbf | 2024 | Wi-Fi-based sensing: CSI/RSSI for motion, presence, gesture detection; velocity 0.1–0.4 m/s, angular resolution 1°–8° |
| [IEEE 802.15.4ab: Enhanced UWB](https://ieee802.org/15/pub/TG4ab.html) | IEEE 802.15 TG4ab | 2024 | UWB ranging and sensing enhancements for cm-level accuracy |
| [Device-Free Sensing in OFDM Cellular Network](https://ieeexplore.ieee.org/document/9775311) | Q. Shi et al. | IEEE JSAC | 2022 | Foundational cellular sensing framework supporting ISAC standardization |

### IEEE 802.11bf Key Milestones

- **2018**: IEEE began exploring Wi-Fi sensing using CSI and RSSI for environmental detection
- **2020**: IEEE 802.11bf Task Group established to formalize Wi-Fi sensing
- **2022**: Draft standard defined sensing measurement reports, interference mitigation, and dynamic resource allocation
- **2024 (Oct)**: First version frozen — backward-compatible sensing via existing Wi-Fi preamble and pilot structures

## ITU (IMT-2030 / 6G Vision)

ITU has formally incorporated ISAC as one of six key application scenarios in its IMT-2030 (6G Vision) framework. Spectrum allocations at WRC-19 (24–52.6 GHz mmWave) and ongoing studies on sub-THz/THz bands (100–300 GHz) provide the spectral foundation for high-resolution ISAC in 6G networks.

| Document | Organization | Year | Focus |
|----------|--------------|------|-------|
| [IMT-2030 Framework (6G Vision)](https://www.itu.int/en/ITU-R/study-groups/rsg5/rwp5d/Pages/default.aspx) | ITU-R WP5D | 2023 | ISAC as 6G key scenario alongside immersive通信, AI, etc. |
| [WRC-19 Spectrum Decisions](https://www.itu.int/en/ITU-R/conferences/wrc/2019/Pages/default.aspx) | ITU | 2019 | mmWave 24–52.6 GHz allocation for 5G/6G ISAC |
| [Sub-THz and THz Band Studies (100–300 GHz)](https://www.itu.int/en/ITU-R/study-groups/rsg5/Pages/default.aspx) | ITU-R | 2024 | High-resolution S&C spectrum for 6G ISAC |
| [Toward Integrated Sensing and Communications for 6G: A Standardization Perspective](https://arxiv.org/abs/2308.01227) | A. Kaushik, R. Singh et al. | IEEE Commun. Stand. Mag. | 2024 | Comprehensive standardization roadmap across 3GPP, IEEE, ITU |

## ETSI ISAC ISG (18 Use Cases)

ETSI has established an Industrial Specification Group (ISG) on ISAC that has developed 18 additional use cases beyond the 3GPP framework. The ETSI ISAC ISG focuses on use-case-specific requirements, performance metrics, and evaluation methodologies to complement 3GPP's broader NR-based approach.

| Document | Organization | Year | Focus |
|----------|--------------|------|-------|
| [ETSI ISAC ISG Use Cases](https://www.etsi.org/committee/2295-isac) | ETSI | 2024 | 18 ISAC use cases covering sensing requirements and KPIs |
| [Architecture for Cellular Enabled ISAC Services](https://ieeexplore.ieee.org/document/10251772/) | B. Liu et al. | China Commun. | 2023 | ETSI-aligned cellular ISAC service architecture |
| [Exploring ISAC for Future 6G Networks](https://blog.huawei.com/en/post/2023/11/14/5-5g-whats-in-a-number) | Huawei | 2023 | Industry perspective on ISAC deployment aligned with ETSI/3GPP |

## Low-Altitude Networks (LAWN) Standardization

The low-altitude economy has emerged as a pivotal ISAC application domain with strong policy support. In China, the Civil Aviation Administration has promoted 5G NR-based low-altitude applications, and the Ministry of Industry and Information Technology announced ISAC-enabled infrastructure for low-altitude networks. Key 3GPP Rel-18 enhancements enable drones to support beamforming, unauthorized flight detection, and multi-station cooperative sensing.

| Document | Organization | Year | Focus |
|----------|--------------|------|-------|
| [ISAC from the Sky: UAV Trajectory Design for Joint Communication and Target Localization](https://ieeexplore.ieee.org/document/10460730) | X. Jing, F. Liu, C. Masouros, Y. Zeng | IEEE TWC | 2024 | UAV ISAC trajectory optimization with communication-sensing trade-off |
| [Joint Maneuver and Beamforming Design for UAV-Enabled ISAC](https://ieeexplore.ieee.org/document/9626783) | Z. Lyu, G. Zhu, J. Xu | IEEE TWC | 2022 | Joint UAV trajectory and beamforming for enhanced ISAC |
| [UAV Meets ISAC: Challenges and Future Directions](https://ieeexplore.ieee.org/document/10137555) | J. Mu, R. Zhang, Y. Cui et al. | IEEE Commun. Mag. | 2023 | Comprehensive survey on UAV-ISAC integration challenges |
| [ZTE's ISAC Solution for Low-Altitude Economy](https://www.zte.com.cn/global/about/news/20240326e2.html) | ZTE | 2024 | 5G-A ISAC for low-altitude economy deployment |
| [China Mobile and ZTE Complete 5G-A Synaesthesia Low-Altitude Verification](https://www.zte.com.cn/global/about/news/) | China Mobile, ZTE | 2024 | 5G-A synaesthesia field trial for drone detection |

## Industry Recognition & Roadmaps

| Recognition/Report | Organization | Year | Focus |
|--------------------|--------------|------|-------|
| [Top 10 Emerging Technologies 2024](https://www.weforum.org/podcasts/radio-davos/episodes/top-10-emerging-technologies-2024/) | World Economic Forum | 2024 | ISAC recognized at Davos as top-10 emerging technology | <!-- REMOVED: WEF podcast page offline; entry removed as non-critical -->
| [5.5G Core: What's in a Number?](https://blog.huawei.com/en/post/2023/11/14/5-5g-whats-in-a-number) | Huawei | 2023 | ISAC integration in 5.5G core network with sensing data monetization |
| [Qualcomm ISAC for 6G](https://www.qualcomm.com/research/6g/isac) | Qualcomm | 2023 | mmWave ISAC for autonomous vehicles and IIoT |
| [Ericsson ISAC in 6G: Smart City Opportunities](https://www.ericsson.com/en/blog/2024/6/integrated-sensing-and-communication) | Ericsson | 2023 | ISAC for intelligent transportation and urban infrastructure |  <!-- FIXED: Ericsson blog post (offline); use webcache or remove -->

---

*Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md).*
