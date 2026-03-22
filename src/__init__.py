"""mipt_master.src — Semantic ID recommendation system.

Modules:
- rqvae: RQ-VAE for hierarchical semantic ID generation
- device: Device management (CUDA/MPS/CPU)
- logger: Logging setup
"""

from .device import DeviceInfo, get_device_info
from .logger import setup_logger
