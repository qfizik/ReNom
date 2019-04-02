#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)

import numpy as np

import renom as rm
from renom.graph.train import initializer as init
from renom.graph.utils import roi_pooling_slice, region_cordinates, roi_pooling_slice_decode
from renom.graph.core import UserGraph, operational_element, operation, \
    GraphMultiStorage, GraphFactory, graph_variable
from renom.graph import populate_graph


class roi_pooling_forward(operation):
    name = "Roi Pool (F)"

    def __init__(self, outw=7, outh=7, spatial_scale=1 / 16.):
        self.outw = outw
        self.outh = outh
        self.spatial_scale = spatial_scale

    def setup(self, inputs):
        rois = inputs[1]['y']
        inputs = inputs[0]['y']
        output_shape = (rois.shape[0], inputs.shape[1], self.outh, self.outw)

        self.gpus = inputs.gpus
        self.inputs = inputs
        self.rois = rois

        self.outputs = GraphMultiStorage(
            shape=output_shape, gpus=self.gpus, initializer=init.Constant(0))
        self.argmax = GraphMultiStorage(
            shape=output_shape, gpus=self.gpus, initializer=init.Constant(0))
        self._vars = {'y': self.outputs}

    def perform(self):
        _, c, h, w = self.inputs.shape
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):

            rm.cuda.curoi_pool2d_forward(self.rois[gpu], self.inputs[gpu], self.spatial_scale, c,
                                         h, w, self.outh, self.outw, self.outputs[gpu], self.argmax[gpu])


class roi_pooling_forward_cpu(roi_pooling_forward):

    def perform(self):
        n_rois = self.rois.shape[0]
        n, c, h, w = self.inputs.shape
        z = self.outputs['cpu']
        index = self.argmax['cpu']
        rois = self.rois['cpu']
        x = self.inputs['cpu']

        for i_roi in range(n_rois):
            idx, xmin, ymin, xmax, ymax = region_cordinates(rois[i_roi], self.spatial_scale)
            roi_height = max(ymax - ymin + 1, 1)
            roi_width = max(xmax - xmin + 1, 1)
            strideh = float(roi_height) / float(self.outh)
            stridew = float(roi_width) / float(self.outw)

            for idx_h in range(self.outh):
                sliceh, lenh = roi_pooling_slice(idx_h, strideh, h, ymin)
                if lenh <= 0:
                    continue
                for idx_w in range(self.outw):
                    slicew, lenw = roi_pooling_slice(idx_w, stridew, w, xmin)
                    if lenw <= 0:
                        continue
                    roi_data = x[int(idx), :, sliceh, slicew].reshape(c, -1)
                    z[i_roi, :, idx_h, idx_w] = np.max(roi_data, axis=1)
                    max_idx_slice = np.unravel_index(np.argmax(roi_data, axis=1), (lenh, lenw))
                    max_idx_slice_h = max_idx_slice[0] + sliceh.start
                    max_idx_slice_w = max_idx_slice[1] + slicew.start
                    max_idx_slice = max_idx_slice_h * w + max_idx_slice_w
                    index[i_roi, :, idx_h, idx_w] = max_idx_slice


class roi_pooling_backward(operation):
    name = "Roi Pool (B)"

    def __init__(self, associated_forward):
        self._fwd_op = associated_forward

    def setup(self, inputs):
        inputs = inputs[0]['y']
        self.spatial_scale = self._fwd_op.spatial_scale
        self.outw = self._fwd_op.outw
        self.outh = self._fwd_op.outh
        self.fwd_inputs = self._fwd_op.inputs
        self.fwd_argmax = self._fwd_op.argmax
        self.fwd_rois = self._fwd_op.rois
        self.inputs = inputs
        self.gpus = inputs.gpus

        self.outputs = GraphMultiStorage(
            shape=self.fwd_inputs.shape, gpus=self.gpus, initializer=init.Constant(0))
        self._vars = {'y': self.outputs, id(self.fwd_inputs): self.outputs}

    def perform(self):
        n, c, h, w = self.fwd_inputs.shape
        for gpu, handle in rm.cuda.RenomHandlers(self.gpus):
            dy = self.inputs[gpu]
            rm.cuda.curoi_pool2d_backward(dy, self.fwd_argmax[gpu], self.fwd_rois[gpu],
                                          self.spatial_scale, c, h, w, self.outh, self.outw, self.outputs[gpu])


class roi_pooling_backward_cpu(roi_pooling_backward):

    def perform(self):
        dy = self.inputs['cpu']
        n, ch, h, w = self.fwd_inputs.shape
        n_rois = self.fwd_rois.shape[0]

        for i_roi in range(n_rois):
            idx, xmin, ymin, xmax, ymax = region_cordinates(self.fwd_rois['cpu'][i_roi],
                                                            self.spatial_scale)
            roi_height = max(ymax - ymin + 1, 1)
            roi_width = max(xmax - xmin + 1, 1)

            stride_h = float(roi_height) / float(self.outh)
            stride_w = float(roi_width) / float(self.outw)
            for idx_h in range(ymin, ymax + 1):
                for idx_w in range(xmin, xmax + 1):
                    start_w, end_w = roi_pooling_slice_decode(
                        idx_w, stride_w, self.outw, xmin)
                    start_h, end_h = roi_pooling_slice_decode(
                        idx_h, stride_h, self.outh, ymin)

                    for ph in range(start_h, end_h):
                        for pw in range(start_w, end_w):
                            max_idx_tmp = self.fwd_argmax['cpu'][i_roi, :, ph, pw].astype(np.int)
                            for c in range(ch):
                                if max_idx_tmp[c] == (idx_h * w + idx_w):
                                    self.outputs['cpu'][idx, c, idx_h,
                                                        idx_w] += dy[i_roi, c, ph, pw]


class RoiPoolElement(UserGraph):

    def __init__(self, outw=7, outh=7, spatial_scale=1 / 16., previous_element=None):
        self.outh = outh
        self.outw = outw
        self.spatial_scale = spatial_scale

        args = (self.outw, self.outh, self.spatial_scale)
        fwd_op = roi_pooling_forward(
            *args) if rm.is_cuda_active() else roi_pooling_forward_cpu(*args)
        bwd_ops = [roi_pooling_backward(fwd_op) if rm.is_cuda_active(
        ) else roi_pooling_backward_cpu(fwd_op)]

        super().__init__(forward_operation=fwd_op, backward_operations=bwd_ops, previous_elements=previous_element)


@populate_graph
class RoiPool(GraphFactory):
    '''ROI pooling function.

    Args:
        outh (tuple,int): Filter size of the convolution filter.
        outw (tuple,int): Size of the zero-padding around the image.
        spatial_scale (tuple,int): Stride-size of the convolution.

    '''

    def prepare(self, outw=7, outh=7, spatial_scale=1 / 16.):
        self.outw = outw
        self.outh = outh
        self.spatial_scale = spatial_scale

    def connect(self, x, roi):
        return RoiPoolElement(self.outw, self.outh, self.spatial_scale, previous_element=[x, roi])
