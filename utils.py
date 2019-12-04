import h5py


def as_hdf5(f, mode):
    if isinstance(f, h5py.File) or isinstance(f, h5py.Group):
        return f
    else:
        return h5py.File(f, mode)