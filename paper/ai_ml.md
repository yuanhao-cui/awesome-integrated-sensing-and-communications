# 🧠 AI/ML for ISAC

> The convergence of AI/ML with ISAC marks a paradigm shift from rule-based signal processing to data-driven, adaptive systems. This section covers deep learning for signal detection, edge intelligence, multi-modal fusion, federated learning, ISAC datasets and benchmarks, and the emerging role of foundation models and LLMs in ISAC.

## AI/ML Survey & Overview

These comprehensive surveys provide the landscape of AI-enhanced ISAC, covering the interplay between artificial intelligence and integrated sensing-communication systems.

| [AI-Enhanced Integrated Sensing and Communications: Advancements, Challenges, and Prospects](https://ieeexplore.ieee.org/document/10663823) | N. Wu et al. | IEEE Commun. Mag. | 2024 | — | Comprehensive review of AI techniques for ISAC |
| [AI-Driven Integration of Sensing and Communication in the 6G Era](https://ieeexplore.ieee.org/document/10553151) | X. Liu et al. | IEEE Network | 2024 | — | AI-driven ISAC from architecture to deployment |

## Edge Intelligence & Real-time Processing

Edge perception integrates wireless sensing, communication, computation, and AI at the network edge for real-time data processing, reducing latency and preserving privacy. The ISAC-AI interplay forms a feedback loop where AI enhances S&C and ISAC provides rich data for AI learning.

| [Edge Perception: Intelligent Wireless Sensing at Network Edge](https://arxiv.org/abs/2410.21017) | **Y. Cui** et al. | IEEE Commun. Mag. | 2025 | — | Edge perception framework integrating sensing, computing, and communication |
| [Edge AI: On-Demand Accelerating Deep Neural Network Inference via Edge Computing](https://ieeexplore.ieee.org/document/8917383) | E. Li et al. | IEEE Trans. Wireless Commun. | 2020 | — | Edge AI fundamentals for DNN inference acceleration |
| [Edge Intelligence for Autonomous Driving in 6G Wireless System](https://ieeexplore.ieee.org/document/9422421) | B. Yang et al. | IEEE Wireless Commun. | 2021 | — | Edge AI design for autonomous driving applications |
| [AI-in-the-Loop Sensing and Communication Joint Design for Edge Intelligence](https://arxiv.org/abs/2502.10203) | Z. Cai et al. | arXiv | 2025 | — | Closed-loop AI-in-the-loop control reducing sensing cost by 52% |
| [Energy-Efficient and Intelligent ISAC in V2X Networks with Spiking Neural Networks-Driven DRL](https://arxiv.org/abs/2501.01038) | C. Shang et al. | arXiv | 2025 | — | SNN-DRL for energy-efficient V2X ISAC |

## Deep Learning for Signal Detection & Classification

Deep learning approaches—including CNNs, LSTMs, Transformers, and model-driven architectures—have shown significant improvements in signal detection, beamforming prediction, waveform design, and target classification for ISAC systems.

| [Deep Learning-based Design of Uplink Integrated Sensing and Communication](https://arxiv.org/abs/2403.01480) | Q. Qi et al. | IEEE Trans. Wireless Commun. | 2024 | — | DL-optimized uplink ISAC with 39% joint efficiency improvement |
| [ISAC-NET: Model-driven Deep Learning for Integrated Passive Sensing and Communication](https://arxiv.org/abs/2307.15074) | W. Jiang et al. | IEEE Trans. Commun. | 2024 | — | Deep unfolding for joint channel reconstruction and target detection |
| [Near-Field Beam Training for Extremely Large-Scale MIMO Based on Deep Learning](https://arxiv.org/abs/2406.03249) | J. Nie, **Y. Cui** et al. | IEEE Trans. Mobile Comput. | 2025 | — | CNN-based near-field beam training for XL-MIMO ISAC |
| [Unsupervised Learning for Joint Beamforming Design in RIS-Aided ISAC Systems](https://arxiv.org/abs/2403.17324) | J. Ye et al. | IEEE Wireless Commun. Lett. | 2024 | — | Label-free RIS beamforming with reduced overhead |
| [Deep CLSTM for Predictive Beamforming in ISAC-Enabled Vehicular Networks](https://ieeexplore.ieee.org/document/10041699) | C. Liu et al. | J. Commun. Inf. Netw. | 2022 | — | CLSTM-based beam prediction for vehicular ISAC |

## Multi-Modal Sensing & Fusion

Multi-modal ISAC integrates heterogeneous data sources (radar, LiDAR, camera, Wi-Fi CSI) to enhance environmental perception and communication reliability through advanced AI-driven fusion techniques.

| [Intelligent Multi-Modal Sensing-Communication Integration: Synesthesia of Machines](https://arxiv.org/abs/2306.14143) | X. Cheng et al. | IEEE Commun. Surveys Tuts. | 2024 | — | Comprehensive survey on multi-modal ISAC (SoM) |
| [Joint Sensing and Semantic Communications with Multi-Task Deep Learning](https://arxiv.org/abs/2311.05017) | Y. E. Sagduyu et al. | IEEE Commun. Mag. | 2024 | — | Multi-task DL integrating sensing and semantic communication |
| [Multi-View Sensing for Wireless Communications: Architectures, Designs, and Opportunities](https://ieeexplore.ieee.org/document/10137555) | X. Tong et al. | IEEE Commun. Mag. | 2023 | — | Multi-view sensing architectures for wireless networks |

## Federated Learning & Distributed Training

Federated and distributed learning enables privacy-preserving model training across ISAC edge devices, addressing channel quality heterogeneity, communication efficiency, and model convergence challenges.

| [Toward Ambient Intelligence: Federated Edge Learning with Task-Oriented Sensing, Computation, and Communication](https://ieeexplore.ieee.org/document/10428398) | P. Liu et al. | IEEE JSTSP | 2023 | — | Federated edge ISAC with task-oriented integration |
| [Over-the-Air Federated Edge Learning: An Overview](https://ieeexplore.ieee.org/document/10561167) | X. Cao et al. | IEEE Wireless Commun. | 2024 | — | OTA FL overview for wireless networks |
| [Integrated Sensing, Communication, and Computation for OTA Federated Learning in 6G](https://ieeexplore.ieee.org/document/10578571) | M. Du et al. | IEEE IoTJ | 2024 | — | ISCC for over-the-air FL in 6G |
| [Distributed Unsupervised Learning for Interference Management in ISAC](https://ieeexplore.ieee.org/document/10217003) | X. Liu et al. | IEEE Trans. Wireless Commun. | 2023 | — | Distributed unsupervised approach for ISAC interference |
| [Communication-Efficient Federated Learning for Large-Scale Multi-Agent Systems in ISAC](https://ieeexplore.ieee.org/document/10501688) | W. Ouyang et al. | IEEE Syst. J. | 2024 | — | Data augmentation with RL for efficient FL in ISAC |

## Datasets & Benchmarks for AI-ISAC

High-quality, multi-modal sensing datasets are critical for training and evaluating AI-driven ISAC algorithms. The Sensing Dataset Platform (SDP) and other benchmark datasets enable reproducible research across detection, tracking, recognition, and imaging tasks.

| [Intelligent Multi-Modal Sensing-Communication Integration: Synesthesia of Machines](https://arxiv.org/abs/2306.14143) | X. Cheng et al. | IEEE Commun. Surveys Tuts. | 2024 | — | Includes M³SC dataset with cameras, LiDAR, GPS, and environmental sensors |
| [WIMANS: A Benchmark Dataset for Wi-Fi-Based Multi-User Activity Sensing](https://link.springer.com/chapter/10.1007/978-3-031-72664-4_5) | S. Huang et al. | ECCV | 2025 | — | Multi-user CSI dataset with 9.4 hours of dual-band data |
| [IMgFI: A High Accuracy and Lightweight HAR Framework Using CSI Image](https://ieeexplore.ieee.org/document/10273603) | C. Zhang et al. | IEEE Sensors J. | 2023 | — | CSI-to-image HAR dataset with lightweight DL framework |
| [EyeFi: Fast Human Identification Through Vision and Wi-Fi-Based Trajectory Matching](https://ieeexplore.ieee.org/document/9114824) | S. Fang et al. | IEEE DCOSS | 2020 | — | Vision + Wi-Fi CSI fusion dataset for human identification |
| [OPERAnet: A Multimodal Activity Recognition Dataset from RF and Vision Sensors](https://www.nature.com/articles/s41597-022-01765-4) | M. J. Bocus et al. | Scientific Data | 2022 | — | 8-hour multi-modal HAR dataset with high-resolution labels |

## Foundation Models & LLMs for ISAC

Large language models and foundation models are emerging as powerful tools for ISAC, enabling task decomposition, cross-modal reasoning, and human-like understanding of physical sensing data.

| [Penetrative AI: Making LLMs Comprehend the Physical World](https://arxiv.org/abs/2310.09605) | H. Xu et al. | ACM HotMobile | 2024 | — | Integrating LLMs with IoT sensors for physical world understanding |
| [Deep Quantum-Transformer Networks for Multimodal Beam Prediction in ISAC](https://ieeexplore.ieee.org/document/10586504) | S. Tariq et al. | IEEE IoTJ | 2024 | — | Quantum-transformer architecture for multimodal beam prediction |
| [Transformer-Based Predictive Beamforming for ISAC in Vehicular Networks](https://ieeexplore.ieee.org/document/10457929) | Y. Zhang et al. | IEEE IoTJ | 2024 | — | Transformer-assisted predictive beamforming for vehicular ISAC |
| [Vision Transformers for Human Activity Recognition Using WiFi CSI](https://ieeexplore.ieee.org/document/10525496) | F. Luo et al. | IEEE IoTJ | 2024 | — | ViT architecture for WiFi CSI-based activity recognition |

---

*Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md).*
