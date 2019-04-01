import urllib.request as request
import gzip
import os
from os import path
import sys
import numpy as np
from tqdm import tqdm

_mnist_urls = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
]


def _maybe_print(string, should_print):
    if should_print:
        print(string)


def _check_files_exist(folder, filenames, verbose=True):
    _maybe_print('Checking for contents in {}'.format(folder), verbose)
    if not path.isdir(folder):
        return False
    for name in filenames:
        if not path.isfile(name):
            return False
    return True


_bar = None


def _my_hook(num, read, total):
    global _bar
    if _bar is None:
        _bar = tqdm(total=total)
    if _bar.n + read > total:
        read = total - _bar.n
    _bar.update(read)


def _download_files(folder, filenames, urls, verbose=True):
    global _bar
    getter = request.URLopener()
    if not path.isdir(folder):
        _maybe_print('Creating new data folder.', verbose)
        os.mkdir(folder)
    for name, link in zip(filenames, urls):
        _maybe_print('Checking if {} exists'.format(name), verbose)
        if not path.isfile(name):
            _maybe_print('Downloading {} into {}.'.format(link, name), verbose)
            getter.retrieve(link, name, _my_hook)
            _bar.close()
            _bar = None

        else:
            _maybe_print('File exists, skipping.', verbose)


def get_mnist(onehot=True, verbose=False, force_download=False):
    '''
    Download mnist data from [mnist]_ Data base.

    Args:
        onehot (bool): If True is given, target will be transformed to onehot vector.
        verbose (bool): If True is given, detail of download progress will be shown.
        force_download (bool): If True is given, data will be downloaded
            even if the data already exist.

    Return:
        (tuple of ndarray): The tuple contains train_image, train_target, test_image and test_target.

    Example:
        >>> from renom.auxiliary.mnist import get_mnist
        >>>
        >>> train_x, train_y, test_x, test_y = get_mnist(onehot=True, verbose=False)
        >>> print(train_x.shape, train_y.shape)
        (60000, 1, 28, 28) (60000, 10)
        >>> print(test_x.shape, test_y.shape)
        (10000, 1, 28, 28) (10000, 10)


    .. [mnist] THE MNIST DATABASE. (http://yann.lecun.com/exdb/mnist/)

    '''
    cur_folder = path.realpath(path.dirname(__file__))
    data_folder = path.join(cur_folder, 'data/')
    names = [link.split('/')[-1] for link in _mnist_urls]
    filenames = [path.join(data_folder, name) for name in names]

    files_exist = _check_files_exist(data_folder, filenames, verbose)
    if not files_exist:
        _download_files(data_folder, filenames, _mnist_urls, verbose)
    else:
        _maybe_print('Data exists.', verbose)
        if force_download:
            _maybe_print('Forcing redownload', verbose)
            for filename in filenames:
                os.remove(filename)
            _download_files(data_folder, filenames, _mnist_urls, verbose)

    # Read in bytes
    mode = 'rb'

    ret_arrays = []
    for filename, name in zip(filenames, names):
        _maybe_print('Decrypting {} to byte contents'.format(name), verbose)
        with gzip.open(filename, mode) as f:
            # Return the byte values of the decompressed file
            file_contents = f.read()
            type_byte = file_contents[2]
            metadata_type = np.dtype('>i4')
            type_char = {8: 'B',
                         9: 'b',
                         11: '>i2',
                         12: '>i4',
                         13: '>f4',
                         14: '>f8'}[type_byte]
            dims = int(file_contents[3])
            shape = []
            for dim in range(1, dims + 1):
                dim_bytes = file_contents[4 * dim: 4 * (dim + 1)]
                shape.append(int(np.frombuffer(dim_bytes, metadata_type)))
            offset = 4 + 4 * dims
            array_type = np.dtype(type_char)
            array_contents = np.frombuffer(file_contents, array_type,
                                           -1, offset)
        if onehot and name in [names[1], names[3]]:
            _maybe_print('Converting labels to one-hot arrays', verbose)
            classes = 10
            size = array_contents.size
            array = np.zeros((size, classes), dtype=np.float32)
            array[np.arange(size), array_contents] = 1
        else:
            array = array_contents.astype(np.float32).reshape(*shape)
            array = np.expand_dims(array, 1)
        final_array = array
        ret_arrays.append(final_array)
        _maybe_print('Finished decrypting {}'.format(name), verbose)

    _maybe_print('Finished building arrays', verbose)
    return ret_arrays[0], ret_arrays[1], ret_arrays[2], ret_arrays[3]
