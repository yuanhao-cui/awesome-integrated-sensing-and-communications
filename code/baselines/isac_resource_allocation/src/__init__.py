"""
ISAC Resource Allocation Framework.

Unified framework for ISAC resource allocation with three sensing QoS metrics:
- Detection QoS (Eq. 18-21)
- Localization QoS (Eq. 22-31)
- Tracking QoS (Eq. 44-47)

Reference: "Sensing as a Service in 6G Perceptive Networks: A Unified Framework for ISAC Resource Allocation"
Authors: Fuwang Dong, Fan Liu, Yuanhao Cui, Wei Wang, Kaifeng Han, Zhiqin Wang
IEEE Transactions on Wireless Communications, 2022
"""

from .system_model import ISACSystem
from .detection_qos import DetectionQoS
from .localization_qos import LocalizationQoS
from .tracking_qos import TrackingQoS
from .comm_rate import CommunicationRate
from .ao_solver import AOSolver, AOResult
from .fairness import FairnessMetrics, FairnessType

__version__ = "1.0.0"
__author__ = "Fuwang Dong, Fan Liu, Yuanhao Cui, Wei Wang, Kaifeng Han, Zhiqin Wang"

__all__ = [
    "ISACSystem",
    "DetectionQoS",
    "LocalizationQoS",
    "TrackingQoS",
    "CommunicationRate",
    "AOSolver",
    "AOResult",
    "FairnessMetrics",
    "FairnessType"
]
