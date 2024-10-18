import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from foa3d.printing import (print_blur, print_flsh,
                            print_new_res, print_prepro_heading)
from foa3d.utils import fwhm_to_sigma


def config_anisotropy_correction(px_sz, psf_fwhm):
    """
    Scanning and light-sheet fluorescence microscopes provide 3D data
    characterized by a lower resolution along the optical axis
    (i.e. the z-axis). However, the longitudinal anisotropy of the PSF
    introduces a strong bias in the estimated 3D orientations
    as discussed by Morawski et al. (NeuroImage, 2018).
    Thus, for obtaining a uniform 3D resolution, the X and Y axes of the input
    microscopy volume images need in general to be blurred.

    Parameters
    ----------
    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D FWHM of the PSF [μm]

    Returns
    -------
    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (resolution anisotropy correction)

    px_sz_iso: numpy.ndarray (shape=(3,), dtype=float)
        new isotropic pixel size [μm]
    """

    # set the isotropic pixel resolution equal to the z-sampling step
    px_sz_iso = np.max(px_sz) * np.ones(shape=(3,))

    # detect preprocessing requirement
    cndt_1 = not np.all(psf_fwhm == psf_fwhm[0])
    cndt_2 = not np.all(px_sz == px_sz[0])
    smooth_sigma = None
    if cndt_1 or cndt_2:

        # print preprocessing heading
        print_prepro_heading()

        # adjust PSF anisotropy via lateral Gaussian blurring
        if cndt_1:

            # estimate the PSF variance from input FWHM values [μm²]
            psf_var = np.square(fwhm_to_sigma(psf_fwhm))

            # estimate the in-plane filter variance [μm²]
            gauss_var = np.max(psf_var) - psf_var

            # ...and the corresponding standard deviation [px]
            smooth_sigma_um = np.sqrt(gauss_var)
            smooth_sigma = np.divide(smooth_sigma_um, px_sz)

            # print preprocessing info
            print_blur(smooth_sigma_um, psf_fwhm)

        # print pixel resize info
        if cndt_2:
            print_new_res(px_sz_iso)
        else:
            print_flsh()

    # skip line
    else:
        print_flsh()

    return smooth_sigma, px_sz_iso


def correct_image_anisotropy(img, rsz_ratio, sigma=None, pad=None, pad_mode='reflect',
                             anti_alias=True, trunc=4, ts_msk=None):
    """
    Smooth and downsample the raw microscopy image so as to uniform the lateral sizes
    of the optical system's PSF and the original voxel size.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D resize ratio

    sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the smoothing Gaussian filter [px]
        (resolution anisotropy correction)

    pad: numpy.ndarray (shape=(3,2), dtype=int)
        image padding array to be resized

    smooth_pad_mode: str
        image padding mode adopted for the smoothing Gaussian filter

    anti_alias: bool
        if True, apply an antialiasing filter when downsampling

    trunc: int
        truncate the Gaussian kernel at this many standard deviations

    tissue_msk: numpy.ndarray (dtype=bool)
        tissue binary mask

    Returns
    -------
    iso_img: numpy.ndarray (axis order=(Z,Y,X))
        isotropic microscopy image

    pad_rsz: numpy.ndarray (shape=(3,2), dtype=int)
        resized image padding array

    ts_msk_rsz: numpy.ndarray (dtype=bool)
        resized tissue binary mask
    """
    # no resizing
    if np.all(rsz_ratio == 1):

        # resize tissue mask, when available
        if ts_msk is not None:
            ts_msk = ts_msk[np.newaxis, ...]
            ts_msk = np.repeat(ts_msk, img.shape[0], axis=0)

        return img, pad, ts_msk

    # lateral blurring
    else:
        if sigma is not None:
            img = gaussian_filter(img, sigma=sigma, mode=pad_mode, truncate=trunc, output=np.float32)

        # downsampling
        iso_shp = np.ceil(np.multiply(np.asarray(img.shape), rsz_ratio)).astype(int)
        iso_img = np.zeros(shape=iso_shp, dtype=img.dtype)
        for z in range(iso_shp[0]):
            iso_img[z, ...] = \
                resize(img[z, ...], output_shape=tuple(iso_shp[1:]), anti_aliasing=anti_alias, preserve_range=True)

        # resize padding array accordingly
        pad_rsz = np.floor(np.multiply(np.array([rsz_ratio, rsz_ratio]).transpose(), pad)).astype(int) \
            if pad is not None else None

        # resize tissue mask, when available
        if ts_msk is not None:
            ts_msk_rsz = resize(ts_msk, output_shape=tuple(iso_shp[1:]), preserve_range=True)
            ts_msk_rsz = ts_msk_rsz[np.newaxis, ...]
            ts_msk_rsz = np.repeat(ts_msk_rsz, iso_shp[0], axis=0)
        else:
            ts_msk_rsz = None

        return iso_img, pad_rsz, ts_msk_rsz
