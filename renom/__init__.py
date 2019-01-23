# -*- coding: utf-8 -*-
"""
ReNom


╔══════╗ ╔══════╗ ╔════╗══╗ ╔══════╗ ╔════╦════╗
║  ╔══╗║ ║  ╔═══╝ ║    ║  ║ ║ ╔══╗ ║ ║    ║    ║
║  ╚══╝║ ║  ╚═══╗ ║  ║ ║  ║ ║ ║  ║ ║ ║  ║ ║ ║  ║
║   ═══╣ ║  ╔═══╝ ║  ║ ║  ║ ║ ║  ║ ║ ║  ║ ║ ║  ║
║  ╔═╗ ║ ║  ╚═══╗ ║  ║ ║  ║ ║ ╚══╝ ║ ║  ║   ║  ║
╚══╝ ╚═╝ ╚══════╝ ╚══╝════╝ ╚══════╝ ╚══╝═══╚══╝

All ReNom classes and methods necessary for graph creation are visible within this module.

ReNom follows the following structure:
Intermediates (  ── Name ──  ) indicate packages.
Leafs (  ── Name  ) indicate functionalities. (This will probably be what you are looking for)

ReNom (Top Level)
  │
  ├── Core ──┬──────┬───────┐
  │          │      │       │
  │         Node  Grads  BasicOps
  │
  │
  ├── Cuda ──┬───────┬────────┬──────┬────────┬────────┐
  │          │       │        │      │        │        │
  │         Base   cuBLAS   cuDNN  cuRAND  GPUValue  Thrust
  │
  │
  ├── Layers ────┬─────────────────────┬─────────────┐
  │              │                     │             │
  │         Activation               Loss         Function
  │              │                     │             │
  │     < Activation Methods >  < Loss Methods >     ├── Model
  │                                                  │
  │                                                  └── < Pre-implemented network models >
  └── Operations



"""
from __future__ import absolute_import
from renom.config import precision
from renom import cuda
from renom import core
from renom.core import Pos
from renom.core import Variable
from renom import operation
from renom.operation import *
from renom.utility import *
from renom.utility.distributor import *
from renom.layers.activation import *
from renom.layers.function import *
from renom.layers.loss import *
from renom.optimizer import *
from renom.debug_graph import *
from renom import graph
import numpy as np


def set_renom_seed(seed=30, all_devices=False):
    """This function sets given seed to both numpy and curand random number generator.

    Args:
        seed(int): Seed.
        all_devices(bool): If True is given, the seed will be set to each device's curand generator.
    """
    if is_cuda_active():
        curand_set_seed(seed, all_devices=all_devices)
    np.random.seed(seed)


__version__ = "2.6.2"
