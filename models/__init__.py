"""
GDyNet Model Architectures

This module provides graph neural network models for molecular dynamics analysis.
"""

from models.gdynet_vanilla import CrystalGraphConvNet as GDyNetVanilla
from models.gdynet_ferro import CrystalGraphConvNet as GDyNetFerro

__all__ = [
    'GDyNetVanilla',
    'GDyNetFerro',
]
