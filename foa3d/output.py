import psutil
import tempfile

from datetime import datetime
from os import makedirs, mkdir, path

import nibabel as nib
import numpy as np
from tifffile import TiffWriter

from foa3d.printing import print_flsh
from foa3d.utils import get_item_size


def create_save_dirs(cli_args, in_img):
    """
    Create saving directory.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        updated namespace of command line arguments

    in_img: dict
        input image dictionary
            fb_ch: int
                neuronal fibers channel

            bc_ch: int
                brain cell soma channel

            msk_bc: bool
                if True, mask neuronal bodies within
                the optionally provided channel

            psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
                3D FWHM of the PSF [μm]

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                pixel size [μm]

            path: str
                path to the 3D microscopy image

            name: str
                name of the 3D microscopy image

            fmt: str
                format of the 3D microscopy image

            is_tiled: bool
                True for tiled reconstructions aligned using ZetaStitcher

            is_vec: bool
                vector field flag

    Returns
    -------
    save_dirs: dict
        saving directories

            frangi: Frangi filter

            odf: ODF analysis

            tmp: temporary data
    """
    # get output path
    out_path = cli_args.out
    if out_path is None:
        out_path = path.dirname(in_img['path'])

    # create saving directory
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out_dir = path.join(out_path, f"Foa3D_{time_stamp}_{in_img['name']}")
    if not path.isdir(base_out_dir):
        makedirs(base_out_dir)

    # create Frangi filter output subdirectory
    save_dirs = {}
    frangi_dir = path.join(base_out_dir, 'frangi')
    mkdir(frangi_dir)
    save_dirs['frangi'] = frangi_dir

    # create ODF analysis output subdirectory
    if cli_args.odf_res is not None:
        odf_dir = path.join(base_out_dir, 'odf')
        mkdir(odf_dir)
        save_dirs['odf'] = odf_dir
    else:
        save_dirs['odf'] = None

    # create temporary directory
    save_dirs['tmp'] = tempfile.mkdtemp(dir=base_out_dir)

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
    if fmt in ('tif', 'tiff'):

        # retrieve image pixel size
        px_sz_z, px_sz_y, px_sz_x = px_sz

        # adjust axes (for correct visualization in Fiji)
        if nd_array.ndim == 3:
            nd_array = np.expand_dims(nd_array, 1)

        # adjust bigtiff optional argument
        bigtiff = nd_array.itemsize * nd_array.size >= 4294967296
        metadata = {'axes': 'ZCYX', 'spacing': px_sz_z, 'unit': 'um'}
        out_name = f'{fname}.{fmt}'
        with TiffWriter(path.join(save_dir, out_name), bigtiff=bigtiff, append=True) as tif:
            for z in range(nz):
                zs = z * dz
                tif.write(nd_array[zs:zs + dz, ...],
                          resolution=(1 / px_sz_x, 1 / px_sz_y),
                          compression='zlib',
                          metadata=metadata)

    # save array to NIfTI file
    elif fmt == 'nii':
        nd_array = nib.Nifti1Image(nd_array, np.eye(4))
        nd_array.to_filename(path.join(save_dir, fname + '.nii'))

    # raise error
    else:
        raise ValueError("Unsupported data format!!!")


def save_frangi_arrays(save_dir, img_name, out_img, ram=None):
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
    # loop over output image dictionary fields and save to TIFF files
    for img_key in out_img.keys():
        if isinstance(out_img[img_key], np.ndarray) and img_key not in (None, 'iso'):
            save_array(f'{img_key}_{img_name}', save_dir, out_img[img_key], out_img['px_sz'], ram=ram)

    print_flsh(f"\nFrangi filter arrays saved to: {save_dir}\n")


def save_odf_arrays(save_dir, img_name, odf_scale_um, px_sz, odf, bg, fbr_dnst, odi_pri, odi_sec, odi_tot, odi_anis):
    """
    Save the output arrays of the ODF analysis stage to TIF and Nifti files.
    Arrays tagged with 'mrtrixview' are preliminarily transformed
    so that ODF maps viewed in MRtrix3 are spatially consistent
    with the analyzed microscopy volume, and the output TIF files.

    Parameters
    ----------
    save_dir: str
        saving directory string path

    img_name: str
        name of the 3D microscopy image

    odf_scale_um: float
        fiber ODF resolution (super-voxel side [μm])

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size (Z,Y,X) [μm]

    odf: NumPy memory-map object (axis order=(X,Y,Z,C), dtype=float32)
        ODF spherical harmonics coefficients

    bg: NumPy memory-map object (axis order=(X,Y,Z), dtype=uint8)
        background for ODF visualization in MRtrix3

    fbr_dnst: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        fiber orientation density [1/μm³]

    odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        primary orientation dispersion parameter

    odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        secondary orientation dispersion parameter

    odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        total orientation dispersion parameter

    odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        orientation dispersion anisotropy parameter

    Returns
    -------
    None
    """
    # save ODF image with background to NIfTI files (adjusted view for MRtrix3)
    sbfx = f'{odf_scale_um}_{img_name}'
    save_array(f'bg_mrtrixview_sv{sbfx}', save_dir, bg, fmt='nii')
    save_array(f'odf_mrtrixview_sv{sbfx}', save_dir, odf, fmt='nii')
    del bg
    del odf

    # save fiber density
    save_array(f'fbr_dnst_sv{sbfx}', save_dir, fbr_dnst, px_sz)
    del fbr_dnst

    # save total orientation dispersion
    save_array(f'odi_tot_sv{sbfx}', save_dir, odi_tot, px_sz)
    del odi_tot

    # save primary orientation dispersion
    if odi_pri is not None:
        save_array(f'odi_pri_sv{sbfx}', save_dir, odi_pri, px_sz)
        del odi_pri

    # save secondary orientation dispersion
    if odi_sec is not None:
        save_array(f'odi_sec_sv{sbfx}', save_dir, odi_sec, px_sz)
        del odi_sec

    # save orientation dispersion anisotropy
    if odi_anis is not None:
        save_array(f'odi_anis_sv{sbfx}', save_dir, odi_anis, px_sz)
        del odi_anis
