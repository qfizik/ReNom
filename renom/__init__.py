#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

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

__version__ = "3.0b1"

import argparse
import numpy as np

from renom.config import precision
from renom import cuda
from renom.cuda import has_cuda, set_cuda_active, is_cuda_active, get_device_count
if has_cuda():
    from renom.cuda.gpuvalue import GPUValue
from renom.auxiliary import populate_value
from renom import graph

logging_level = 0


def set_renom_seed(seed=30, all_devices=False):
    """This function sets given seed to both numpy and curand random number generator.

    Args:
        seed(int): Seed.
        all_devices(bool): If True is given, the seed will be set to each device's curand generator.
    """
    if is_cuda_active():
        cuda.curand_set_seed(seed, all_devices=all_devices)
    np.random.seed(seed)


def show_config(args):
    import os
    import platform
    os_name = platform.system()
    os_platform = platform.platform()
    os_version = platform.version()  # NOQA
    python_version = platform.python_version()
    installed_location = os.path.join('/', *list(__file__.split("/")[:-2]))
    if cuda.has_cuda():
        cuda_driver_version = "None"  # NOQA
        cuda_toolkit_version = "None"  # NOQA
        cuda_cudnn_version = "None"  # NOQA
        connected_gpu_list = []  # NOQA
    else:
        cuda_driver_version = "None"  # NOQA
        cuda_toolkit_version = "None"  # NOQA
        cuda_cudnn_version = "None"  # NOQA
        connected_gpu_list = []  # NOQA

    print()
    # 1. OS information.
    print("       OS : {}({})".format(os_name, os_platform))
    # 2. Python information.
    print("   Python : {}".format(python_version))
    # 3. Cuda information.
    # 4. CuDNN information.
    # 5. ReNom version.
    print(" Location : {}".format(installed_location))
    print()


def console_scripts():
    parser = argparse.ArgumentParser(description='ReNomDL support scripts.')
    subparsers = parser.add_subparsers()
    parser_add = subparsers.add_parser('show',
                                       help='Show configuration of current environment.')
    parser_add.set_defaults(handler=show_config)

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()
