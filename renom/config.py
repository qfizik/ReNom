#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import os
import numpy as np

p = os.environ.get("RENOM_PRECISION", 32)
if p == "64":
    precision = np.float64
else:
    precision = np.float32
