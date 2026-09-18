"""Historical validation for cross-system forecast adaptation."""
from .method import GaugeFormerConfig, GaugeFormerMemory, GaugeFormerOutput
GaugeFormer = GaugeFormerMemory
__all__ = ['GaugeFormer', 'GaugeFormerConfig', 'GaugeFormerMemory', 'GaugeFormerOutput']
