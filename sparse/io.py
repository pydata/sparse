import numpy as np

from sparse.coo.core import COO

def save_npz(filename, matrix, compressed=True):
    nodes = {'data':   matrix.data,
             'coords': matrix.coords}

    if compressed:
        np.savez_compressed(filename, **nodes)
    else:
        np.savez(filename, **nodes)

def load_npz(filename):
    with np.load(filename) as fp:
        try:
            coords = fp['coords']
            data   = fp['data']
        except KeyError:
            raise RuntimeError('The file {} does not contain a valid sparse matrix'.format(filename))

    return COO(coords=coords, data=data)
