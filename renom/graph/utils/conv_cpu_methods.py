import numpy as np
from renom.utils import im2col, col2im, imncol, colnim, colnw

def _get_expanded_value(value, dims):
    if isinstance(value, int):
        ret = np.array(list(value for i in range(dims))).astype(np.int32)
    elif isinstance(value, tuple):
        assert len(value) == dims, 'tuple and input shape mismatch'
        ret = np.array(value).astype(np.int32)
    else:
        raise ValueError('Expected int or tuple, but got {}'.format(type(value)))
    return ret



def grouped_conv_forward(x, w, b, col, groups, kernel, stride, padding, dilation):
    out_h, out_w = col.shape[-2:]

    out_channels = w.shape[0]
    in_channels = x.shape[1]

    iCg = in_channels // groups
    oCg = out_channels // groups
    k_h, k_w = kernel
    N = x.shape[0]

    col = col.transpose(1, 2, 3, 0, 4, 5)
    col = col.reshape(groups, iCg * k_h * k_w, N * out_h * out_w)
    w_new = w.reshape(groups, oCg, iCg * k_h * k_w)

    value = np.matmul(w_new, col)
    value = value.reshape(groups * oCg, N, out_h, out_w)
    value = value.transpose(1, 0, 2, 3)

    value += b.reshape(1, b.size, 1, 1)
    return value, col


def grouped_conv_back(x, w, b, dy, col, groups, kernel, stride, padding, dilation):
    out_h, out_w = dy.shape[-2:]
    out_channels = w.shape[0]
    in_channels = x.shape[1]
    iCg = in_channels // groups
    oCg = out_channels // groups
    k_h, k_w = kernel[0], kernel[1]
    N = x.shape[0]
    dy_temp = dy.transpose(1, 0, 2, 3)
    dy_temp = dy_temp.reshape(groups, oCg, N * out_h * out_w)
    w_temp = w.reshape(groups, oCg, iCg * k_h * k_w)
    w_temp = w_temp.transpose(0, 2, 1)
    dx = np.matmul(w_temp, dy_temp)
    dx = dx.reshape(groups * iCg, k_h, k_w, N, out_h, out_w)
    dx = np.rollaxis(dx, 3)
    dx = col2im(dx, x.shape[2:], stride,
                padding, dilation)

    col_temp = col
    col_temp = col_temp.transpose(0, 2, 1)

    dy_temp = dy.transpose(1, 0, 2, 3)
    dy_temp = dy_temp.reshape(groups, oCg, N * out_h * out_w)

    dw = np.matmul(dy_temp, col_temp)
    dw = dw.reshape(groups * oCg, iCg, k_h, k_w)

    db = np.sum(dy, (0, 2, 3), keepdims=True)

    return dx, dw, db
