import psutil
import tempfile

from datetime import datetime
from os import makedirs, mkdir, path

import nibabel as nib
import numpy as np
from tifffile import TiffWriter

from foa3d.printing import print_flsh
from foa3d.utils import get_item_size


def create_save_dirs(img_path, img_name, cli_args, is_vec=False):
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
    save_dirs: dict
        saving directories
        ('frangi': Frangi filter, 'odf': ODF analysis, 'tmp': temporary files)
    """

    # get current time
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # get output path
    out_path = cli_args.out
    if out_path is None:
        out_path = path.dirname(img_path)

    # create saving directory
    base_out_dir = path.join(out_path, f"Foa3D_{time_stamp}_{img_name}")
    if not path.isdir(base_out_dir):
        makedirs(base_out_dir)

    # initialize empty dictionary
    save_dirs = dict()

    # create Frangi filter output subdirectory
    if not is_vec:
        frangi_dir = path.join(base_out_dir, 'frangi')
        mkdir(frangi_dir)
        save_dirs['frangi'] = frangi_dir
    else:
        save_dirs['frangi'] = None

    # create ODF analysis output subdirectory
    if cli_args.odf_res is not None:
        odf_dir = path.join(base_out_dir, 'odf')
        mkdir(odf_dir)
        save_dirs['odf'] = odf_dir
    else:
        save_dirs['odf'] = None

    # create temporary directory
    tmp_dir = tempfile.mkdtemp(dir=base_out_dir)
    save_dirs['tmp'] = tmp_dir

    return save_dirs


def save_array(fname, save_dir, nd_array, px_sz=None, fmt='tiff', ram=None):
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

    ram: float
        maximum RAM available

    Returns
    -------
    None
    """

    # get maximum RAM and initialized array memory size
    if ram is None:
        ram = psutil.virtual_memory()[1]
    itm_sz = get_item_size(nd_array.dtype)
    dz = np.floor(ram / (itm_sz * np.prod(nd_array.shape[1:]))).astype(int)
    nz = np.ceil(nd_array.shape[0] / dz).astype(int)

    # check output format
    fmt = fmt.lower()
    if fmt == 'tif' or fmt == 'tiff':

        # retrieve image pixel size
        px_sz_z, px_sz_y, px_sz_x = px_sz

        # adjust bigtiff optional argument
        bigtiff = True if nd_array.itemsize * nd_array.size >= 4294967296 else False
        out_name = f'{fname}.{fmt}'
        with TiffWriter(path.join(save_dir, out_name), bigtiff=bigtiff, append=True) as tif:
            for z in range(nz):
                zs = z * dz
                tif.write(nd_array[zs:zs + dz, ...], contiguous=True, resolution=(1 / px_sz_x, 1 / px_sz_y),
                          metadata={'spacing': px_sz_z, 'unit': 'um'})

    # save array to NIfTI file
    elif fmt == 'nii':
        nd_array = nib.Nifti1Image(nd_array, np.eye(4))
        nd_array.to_filename(path.join(save_dir, fname + '.nii'))

    # raise error
    else:
        raise ValueError("Unsupported data format!!!")


def save_frangi_arrays(save_dir, img_name, out_img, px_sz, ram=None):
    """
    Save the output arrays of the Frangi filter stage to TIF files.

    Parameters
    ----------
    save_dir: str
        saving directory string path
    
    img_name: str
        name of the input microscopy image

    out_img: dict
        fbr_vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
            fiber orientation vector field

        fbr_vec_clr: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=uint8)
            orientation colormap image

        fa_img: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            fractional anisotropy image

        frangi_img: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            Frangi-enhanced image (fiber probability)

        iso_fbr: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            isotropic fiber image

        fbr_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            fiber mask image

        bc_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            neuron mask image

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size (Z,Y,X) [μm]

    ram: float
        maximum RAM available

    Returns
    -------
    None
    """
    # loop over output image dictionary fields and save to TIFF
    for img_key in out_img.keys():
        if out_img[img_key] is not None:
            save_array(f'{img_key}_{img_name}', save_dir, out_img[img_key], px_sz, ram=ram)

    # print output directory
    print_flsh(f"\nFrangi filter arrays saved to: {save_dir}\n")


def save_odf_arrays(odf, bg, odi_pri, odi_sec, odi_tot, odi_anis, disarray, px_sz, save_dir, img_name, odf_scale_um):
    """
    Save the output arrays of the ODF analysis stage to TIF and Nifti files.
    Arrays tagged with 'mrtrixview' are preliminarily transformed
    so that ODF maps viewed in MRtrix3 are spatially consistent
    with the analyzed microscopy volume, and the output TIF files.

    Parameters
    ----------
    odf: NumPy memory-map object (axis order=(X,Y,Z,C), dtype=float32)
        ODF spherical harmonics coefficients

    bg: NumPy memory-map object (axis order=(X,Y,Z), dtype=uint8)
        background for ODF visualization in MRtrix3

    odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        primary orientation dispersion parameter

    odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        secondary orientation dispersion parameter

    odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        total orientation dispersion parameter

    odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        orientation dispersion anisotropy parameter

    disarray: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size (Z,Y,X) [μm]

    save_dir: str
        saving directory string path

    img_name: str
        name of the input volume image

    odf_scale_um: float
        fiber ODF resolution (super-voxel side [μm])

    Returns
    -------
    None
    """

    # save ODF image with background to NIfTI files (adjusted view for MRtrix3)
    sbfx = f'{odf_scale_um}_{img_name}'
    save_array(f'bg_mrtrixview_sv{sbfx}', save_dir, bg, fmt='nii')
    save_array(f'odf_mrtrixview_sv{sbfx}', save_dir, odf, fmt='nii')

    # save total orientation dispersion
    save_array(f'odi_tot_sv{sbfx}', save_dir, odi_tot, px_sz)

    # save primary orientation dispersion
    if odi_pri is not None:
        save_array(f'odi_pri_sv{sbfx}', save_dir, odi_pri, px_sz)

    # save secondary orientation dispersion
    if odi_sec is not None:
        save_array(f'odi_sec_sv{sbfx}', save_dir, odi_sec, px_sz)

    # save orientation dispersion anisotropy
    if odi_anis is not None:
        save_array(f'odi_anis_sv{sbfx}', save_dir, odi_anis, px_sz)

    # save angular disarray
    if disarray is not None:
        save_array(f'disarray_sv{sbfx}', save_dir, disarray, px_sz)
