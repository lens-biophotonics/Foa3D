import psutil
import shutil
import tempfile

from datetime import datetime
from os import makedirs, path

import nibabel as nib
import numpy as np
import tifffile as tiff

from foa3d.utils import get_item_size


def create_save_dirs(img_path, img_name, cli_args, is_fovec=False):
    """
    Create saving directory.

    Parameters
    ----------
    img_path: str
        path to input microscopy volume image

    img_name: str
        name of the input volume image

    cli_args: see ArgumentParser.parse_args
        updated namespace of command line arguments

    is_fovec: bool
        True when fiber orientation vectors are provided as input
        to the pipeline

    Returns
    -------
    save_subdirs: list (dtype=str)
        saving subdirectory string paths

    tmp_dir: str
        temporary directory (for memory-map objects)
    """

    # get current time
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # get output path
    out_path = cli_args.out
    if out_path is None:
        out_path = path.dirname(img_path)

    # create saving directory
    base_out_dir = path.join(out_path, f"Foa3D_{time_stamp}_{img_name}")
    save_dir_lst = list()
    if not path.isdir(base_out_dir):
        makedirs(base_out_dir)

    # create Frangi filter output subdirectory
    if not is_fovec:
        frangi_dir = path.join(base_out_dir, 'frangi')
        makedirs(frangi_dir)
        save_dir_lst.append(frangi_dir)
    else:
        save_dir_lst.append(None)

    # create ODF analysis output subdirectory
    if cli_args.odf_res is not None:
        odf_dir = path.join(base_out_dir, 'odf')
        makedirs(odf_dir)
        save_dir_lst.append(odf_dir)
    else:
        save_dir_lst.append(None)

    # create temporary directory
    tmp_dir = tempfile.mkdtemp(dir=out_path)

    return save_dir_lst, tmp_dir


def save_array(fname, save_dir, nd_array, px_sz=None, fmt='tiff'):
    """
    Save array to file.

    Parameters
    ----------
    fname: string
        output filename

    save_dir: string
        saving directory string path

    nd_array: NumPy memory-map object or HDF5 dataset
        data

    px_sz: tuple
        pixel size (Z,Y,X) [um]

    fmt: str
        output format

    Returns
    -------
    None
    """

    # get maximum RAM and initialized array memory size
    ram = psutil.virtual_memory()[1]
    item_sz = get_item_size(nd_array.dtype)
    vol_sz = item_sz * np.prod(nd_array.shape).astype(np.float64)

    # copy memory-mapped array (too large for RAM) to output directory
    if vol_sz >= ram:
        shutil.copy(nd_array.filename, path.join(save_dir, f'{fname}.npy'))

    # load array to RAM and export to file with requested format
    else:
        # check output format
        fmt = fmt.lower()
        if fmt == 'tif' or fmt == 'tiff':

            # retrieve image pixel size
            px_sz_z, px_sz_y, px_sz_x = px_sz

            # adjust bigtiff optional argument
            bigtiff = True if nd_array.itemsize * nd_array.size >= 4294967296 else False

            # save array to TIFF file
            out_name = f'{fname}.{fmt}'
            if nd_array.ndim == 3:
                tiff.imwrite(path.join(save_dir, out_name), nd_array, imagej=True, bigtiff=bigtiff,
                             resolution=(1 / px_sz_x, 1 / px_sz_y),
                             metadata={'axes': 'ZYX', 'spacing': px_sz_z, 'unit': 'um'}, compression='zlib')
            elif nd_array.ndim == 4:
                tiff.imwrite(path.join(save_dir, out_name), nd_array, imagej=True, bigtiff=bigtiff,
                             resolution=(1 / px_sz_x, 1 / px_sz_y),
                             metadata={'spacing': px_sz_z, 'unit': 'um'}, compression='zlib')

        # save array to NumPy file
        elif fmt == 'npy':
            np.save(path.join(save_dir, fname + '.npy'), nd_array)

        # save array to NIfTI file
        elif fmt == 'nii':
            nd_array = nib.Nifti1Image(nd_array, np.eye(4))
            nd_array.to_filename(path.join(save_dir, fname + '.nii'))

        # raise error
        else:
            raise ValueError("Unsupported data format!!!")
