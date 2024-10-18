import os
from platform import system

import numpy as np

from foa3d.utils import elapsed_time

# adjust ANSI escape sequence
# decoding to Windows OS
if system == 'Windows':
    os.system("color")


def color_text(r, g, b, text):
    """
    Get colored text string.

    Parameters
    ----------
    r: int
        red channel value

    g: int
        green channel value

    b: int
        blue channel value

    text: str
        text string

    Returns
    -------
    clr_text: str
        colored text
    """
    clr_text = f"\033[38;2;{r};{g};{b}m{text} \033[38;2;255;255;255m"

    return clr_text


def print_flsh(string_to_print=""):
    """
    Print string and flush output data buffer.

    Parameters
    ----------
    string_to_print: str
        string to be printed

    Returns
    -------
    None
    """
    print(string_to_print, flush=True)


def print_frangi_info(alpha, beta, gamma, scales_um, img_shp_um, in_slc_shp_um,
                      tot_slc_num, px_sz, img_item_sz, msk_bc):
    """
    Print Frangi filter stage heading.

    Parameters
    ----------
    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    scales_um: list (dtype=float)
        analyzed spatial scales [μm]

    img_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    in_slc_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the analyzed image slices [μm]

    tot_slc_num: int
        total number of analyzed image slices

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    img_item_sz: int
        image item size (in bytes)

    msk_bc: bool
        if True, mask neuronal bodies within
        the optionally provided channel

    Returns
    -------
    None
    """

    scales_um = np.asarray(scales_um)
    if gamma is None:
        gamma = 'auto'

    print_flsh(color_text(0, 191, 255, "\n3D Frangi Filter\n") + "\nSensitivity\n" + \
        f"• plate-like \u03B1: {alpha:.1e}\n• blob-like  \u03B2: {beta:.1e}\n• background \u03B3: {gamma}\n\n" + \
        f"Enhanced scales      [μm]: {scales_um}\nEnhanced diameters   [μm]: {4 * scales_um}\n")

    # print iterative analysis information
    print_slicing_info(img_shp_um, in_slc_shp_um, tot_slc_num, px_sz, img_item_sz)

    # print neuron masking info
    print_soma_masking(msk_bc)


def print_blur(sigma_um, psf_fwhm):
    """
    Print the standard deviation of the smoothing Gaussian filter.

    Parameters
    ----------
    sigma_um: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the smoothing Gaussian filter [μm]
        (resolution anisotropy correction)

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D FWHM of the PSF [μm]

    Returns
    -------
    None
    """
    psf_sz = np.max(psf_fwhm)
    print_flsh(f"Gaussian blur σ      [μm]: ({sigma_um[0]:.3f}, {sigma_um[1]:.3f}, {sigma_um[2]:.3f})\n" + \
        f"Adjusted PSF FWHM    [μm]: ({psf_sz:.3f}, {psf_sz:.3f}, {psf_sz:.3f})")


def print_import_time(start_time):
    """
    Print image import time.

    Parameters
    ----------
    start_time: float
        import start time

    Returns
    -------
    None
    """
    _, _, mins, secs = elapsed_time(start_time)
    print_flsh(f"Image loaded in: {mins} min {secs:3.1f} s")


def print_odf_info(odf_scales_um, odf_degrees):
    """
    Print ODF analysis heading.

    Parameters
    ----------
    odf_scales_um: list (dtype=float)
        fiber ODF resolutions (super-voxel sides [μm])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    Returns
    -------
    None
    """
    print_flsh(color_text(0, 191, 255, "\n3D ODF Analysis\n") + \
        f"\nResolution   [μm]: {odf_scales_um}\n" + \
        f"Expansion degrees: {odf_degrees}\n")


def print_pipeline_heading():
    """
    Print Foa3D tool heading.

    Returns
    -------
    None
    """
    print_flsh(color_text(0, 250, 154, "\n3D Fiber Orientation Analysis"))


def print_prepro_heading():
    """
    Print preprocessing heading.

    Returns
    -------
    None
    """
    print_flsh(color_text(0, 191, 255, "\n\nMicroscopy Image Preprocessing\n") + \
        "\n                              Z      Y      X")


def print_native_res(px_sz, psf_fwhm):
    """
    Print the native pixel size and optical resolution of the microscopy system.

    Parameters
    ----------
    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        PSF 3D FWHM [μm]

    Returns
    -------
    None
    """
    print_flsh(f"Pixel size           [μm]: ({px_sz[0]:.3f}, {px_sz[1]:.3f}, {px_sz[2]:.3f})\n" + \
        f"PSF FWHM             [μm]: ({psf_fwhm[0]:.3f}, {psf_fwhm[1]:.3f}, {psf_fwhm[2]:.3f})")


def print_new_res(px_sz_iso):
    """
    Print the adjusted isotropic spatial resolution.

    Parameters
    ----------
    px_sz_iso: numpy.ndarray (shape=(3,), dtype=float)
        new isotropic pixel size [μm]

    Returns
    -------
    None
    """
    print_flsh(f"Adjusted pixel size  [μm]: ({px_sz_iso[0]:.3f}, {px_sz_iso[1]:.3f}, {px_sz_iso[2]:.3f})\n")


def print_slicing_info(img_shp_um, slc_shp_um, tot_slc_num, px_sz, img_item_sz):
    """
    Print information on the slicing of the basic image sub-volumes processed by the Foa3D tool.

    Parameters
    ----------
    img_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        3D microscopy image [μm]

    slc_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the analyzed image slices [μm]

    tot_slc_num: int
        total number of analyzed image slices

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    img_item_sz: int
        image item size (in bytes)

    Returns
    -------
    None
    """

    # adjust slice shape
    if np.any(img_shp_um < slc_shp_um):
        slc_shp_um = img_shp_um

    # get image memory size
    img_sz = img_item_sz * np.prod(np.divide(img_shp_um, px_sz))

    # get slice memory size
    max_slc_sz = img_item_sz * np.prod(np.divide(slc_shp_um, px_sz))

    # print info
    print_flsh("\n                              Z      Y      X")
    print_flsh(f"Total image shape    [μm]: ({img_shp_um[0]:.1f}, {img_shp_um[1]:.1f}, {img_shp_um[2]:.1f})\n" + \
        f"Total image size     [MB]: {np.ceil(img_sz / 1024**2).astype(int)}\n\n" + \
        f"Image slice shape    [μm]: ({slc_shp_um[0]:.1f}, {slc_shp_um[1]:.1f}, {slc_shp_um[2]:.1f})\n" + \
        f"Image slice size     [MB]: {np.ceil(max_slc_sz / 1024**2).astype(int)}\n" + \
        f"Image slice number:        {tot_slc_num}\n")


def print_soma_masking(msk_bc):
    """
    Print information on the optional masking of neuronal bodies.

    Parameters
    ----------
    msk_bc: bool
        if True, mask neuronal bodies within
        the optionally provided channel

    Returns
    -------
    None
    """
    prt = 'Soma mask: '
    if msk_bc:
        print_flsh(f'{prt}active\n')
    else:
        print(f'{prt}not active\n')


def print_image_shape(cli_args, in_img):
    """
    Print 3D microscopy image shape.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    in_img: dict
        input image dictionary
        ('img_data': image data, 'ts_msk': tissue sample mask, 'ch_ax': channel axis)

    Returns
    -------
    None
    """

    # get pixel size
    px_sz_z = cli_args.px_size_z
    px_sz_xy = cli_args.px_size_xy

    # get channel axis (RGB image only)
    ch_ax = in_img['ch_ax']

    # get image shape (ignore channel axis)
    img_shp = in_img['img_data'].shape
    if ch_ax is not None:
        img_shp = np.delete(img_shp, ch_ax)

    print_flsh("\n                              Z      Y      X\nImage shape          [μm]: " + \
        f"({img_shp[0] * px_sz_z:.1f}, {img_shp[1] * px_sz_xy:.1f}, {img_shp[2] * px_sz_xy:.1f})")
