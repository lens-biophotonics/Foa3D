from itertools import product

import numpy as np
import psutil

from numba import njit
from foa3d.utils import get_available_cores


@njit(cache=True)
def adjust_axis_range(ax_iter, img_shp, slc_per_ax, start, stop, ovlp=0):
    """
    Trim slice axis range at image boundaries
    and adjust padded ranges accordingly.

    Parameters
    ----------
    ax_iter: int
        iteration counter along axis

    img_shp: int
        total image shape along axis [px]

    slc_per_ax: int
        slices along axis

    start: int
        start coordinate [px]

    stop: int
        stop coordinate [px]

    ovlp: int
        overlapping range between slices along each axis [px]

    Returns
    -------
    start: int
        adjusted start coordinate [px]

    stop: int
        adjusted stop coordinate [px]

    pad: numpy.ndarray (shape=(2,), dtype=int)
        lower and upper padding ranges [px]
    """

    # initialize slice padding array
    pad = np.zeros((2,), dtype=np.int64)

    # adjust start coordinate
    start -= ovlp
    if start < 0:
        pad[0] = -start
        start = 0

    # adjust stop coordinate
    stop += ovlp
    if stop > img_shp:
        pad[1] = stop - img_shp
        stop = img_shp

    # handle image residuals at boundaries
    if ax_iter == slc_per_ax - 1:
        pad[1] = ovlp
        stop = img_shp

    return start, stop, pad


@njit(cache=True)
def compute_axis_range(ax, ax_iter, slc_shp, img_shp, slc_per_dim, ovlp=0, flip=False):
    """
    Adjust image slice coordinates at boundaries.

    Parameters
    ----------
    ax: int
        image axis

    ax_iter: int
        iteration counter along axis (see pipeline.iterate_frangi_on_slices)

    slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    slc_per_dim: numpy.ndarray (shape=(3,), dtype=int)
        image slices along each axis

    ovlp: int
        overlapping range between slices along each axis [px]

    flip: bool
        flip axes

    Returns
    -------
    start: int
        start index [px]

    stop: int
        stop index [px]

    pad: numpy.ndarray (shape=(2,), dtype=int)
        lower and upper padding ranges [px]
    """

    # compute start and stop coordinates
    start = ax_iter[ax] * slc_shp[ax]
    stop = start + slc_shp[ax]

    # flip coordinates if required
    if flip:
        start_tmp = start
        start = img_shp[ax] - stop
        stop = img_shp[ax] - start_tmp

    # adjust start and stop coordinates
    start, stop, pad = adjust_axis_range(ax_iter[ax], img_shp[ax], slc_per_dim[ax], start, stop, ovlp=ovlp)

    return start, stop, pad


def compute_slice_range(ax_iter, slc_shp, img_shp, slc_per_dim, ovlp=0, flip=False):
    """
    Compute basic slice coordinates from microscopy volume image.

    Parameters
    ----------
    ax_iter: tuple
        iteration counters along each axis

    slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    slc_per_dim: numpy.ndarray (shape=(3,), dtype=int)
        image slices along each axis

    ovlp: int
        overlapping range between slices along each axis [px]

    flip: bool
        flip axes

    Returns
    -------
    rng: np.ndarray
        3D slice index ranges [px]

    pad: np.ndarray (shape=(3,2), dtype=int)
        slice padding range [px]
    """

    # generate axis range and padding array
    dim = len(ax_iter)
    slc = tuple()
    pad = np.zeros((dim, 2), dtype=np.int64)
    for ax in range(dim):
        start, stop, pad[ax] = compute_axis_range(ax, ax_iter, slc_shp, img_shp, slc_per_dim, ovlp=ovlp, flip=flip)
        slc += (slice(start, stop, 1),)

    # generate range array
    rng = np.index_exp[slc]

    return rng, pad


def compute_overlap(smooth_sigma, frangi_sigma_um, px_rsz_ratio=None, trunc=2):
    """
    Compute lateral slice extension range
    for coping with smoothing-related boundary artifacts.

    Parameters
    ----------
    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the smoothing Gaussian filter [px]

    frangi_sigma_um: numpy.ndarray (dtype=float)
        analyzed spatial scales [μm]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    trunc: int
        neglect the Gaussian smoothing kernel
        after this many standard deviations

    Returns
    -------
    ovlp: int
        overlapping range between image slices along each axis [px]

    ovlp_rsz: numpy.ndarray (shape=(3,), dtype=int)
        resized overlapping range between image slices
        (if px_rsz_ratio is not None) [px]
    """
    sigma = np.max(frangi_sigma_um) / px_rsz_ratio
    if smooth_sigma is not None:
        sigma = np.concatenate((smooth_sigma, sigma))

    ovlp = int(np.ceil(2 * trunc * np.max(sigma)) // 2)
    ovlp_rsz = np.multiply(ovlp * np.ones((3,)), px_rsz_ratio).astype(np.int64) if px_rsz_ratio is not None else None

    return ovlp, ovlp_rsz


def compute_slice_shape(img_shp, item_sz, px_sz=None, slc_sz=1e6, ovlp=0):
    """
    Compute basic image chunk shape depending on its maximum size (in bytes).

    Parameters
    ----------
    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    item_sz: int
        image item size [B]

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    slc_sz: float
        maximum memory size of the basic image slices
        analyzed using parallel threads [B]

    ovlp: int
        overlapping range between image slices along each axis [px]

    Returns
    -------
    slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    slc_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the basic image slices
        analyzed using parallel threads [μm]
        (if px_size is provided)
    """
    if len(img_shp) == 4:
        item_sz *= 3

    tot_ovlp = 2 * ovlp
    side_xyz = np.floor((slc_sz / item_sz)**(1/3)).astype(int)
    side_z = min(img_shp[0] + tot_ovlp, side_xyz)
    side_xy = np.floor(np.sqrt(np.abs(slc_sz / side_z / item_sz))).astype(int)
    slc_shp = np.array([side_z, side_xy, side_xy]) - tot_ovlp
    slc_shp_um = np.multiply(slc_shp, px_sz) if px_sz is not None else None

    return slc_shp, slc_shp_um


def compute_slice_size(max_ram, mem_growth, mem_fudge, batch_sz, ns=1):
    """
    Compute the size of the basic microscopy image slices fed to the Frangi filtering stage.

    Parameters
    ----------
    max_ram: float
        available RAM [B]

    mem_growth: float
        empirical memory growth factor

    mem_fudge: float
        memory fudge factor

    batch_sz: int
        slice batch size

    ns: int
        number of spatial scales

    Returns
    -------
    slc_sz: float
        memory size of the basic image slices
        analyzed using parallel threads [B]
    """
    slc_sz = max_ram / (batch_sz * mem_growth * mem_fudge * ns)

    return slc_sz


def config_frangi_batch(px_sz, px_sz_iso, img_shp, item_sz, smooth_sigma, frangi_sigma_um,
                        mem_growth=149.7, mem_fudge=1.0, jobs=None, ram=None, shp_thr=7):
    """
    Compute size and number of the batches of basic microscopy image slices
    analyzed in parallel.

    Parameters
    ----------
    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    px_sz_iso: int
        isotropic pixel size [μm]

    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    item_sz: int
        image item size [B]

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the smoothing Gaussian filter [px]

    frangi_sigma_um: numpy.ndarray (dtype=float)
        Frangi filter scales [μm]

    mem_growth: float
        empirical memory growth factor
        of the Frangi filtering stage

    mem_fudge: float
        memory fudge factor

    jobs: int
        number of parallel jobs (threads)
        used by the Frangi filtering stage

    ram: float
        maximum RAM available to the Frangi filtering stage [B]

    shp_thr: int
        minimum slice side [px]

    Returns
    -------
    batch_sz: int
        slice batch size

    in_slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    in_slc_shp_um: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [μm]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    ovlp: int
        overlapping range between image slices along each axis [px]

    ovlp_rsz: numpy.ndarray (shape=(3,), dtype=int)
        resized overlapping range between image slices [px]
    """
    # maximum RAM not provided: use all
    if ram is None:
        ram = psutil.virtual_memory()[1]

    # number of logical cores
    num_cpu = get_available_cores()
    if jobs is None:
        jobs = num_cpu

    # number of spatial scales
    ns = len(frangi_sigma_um)

    # initialize slice batch size
    batch_sz = np.min([jobs // ns, num_cpu]).astype(int) + 1

    # get pixel resize ratio
    px_rsz_ratio = np.divide(px_sz, px_sz_iso)

    # compute slice overlap (boundary artifacts suppression)
    ovlp, ovlp_rsz = compute_overlap(smooth_sigma, frangi_sigma_um, px_rsz_ratio=px_rsz_ratio)

    # compute the shape of basic microscopy image slices
    in_slc_shp = np.array([-1])
    while np.any(in_slc_shp < shp_thr):
        batch_sz -= 1
        if batch_sz == 0:
            raise ValueError(
                "Basic image slices do not fit the available RAM: decrease spatial scales and/or maximum scale value!")
        else:
            slc_sz = compute_slice_size(ram, mem_growth, mem_fudge, batch_sz, ns)
            in_slc_shp, in_slc_shp_um = compute_slice_shape(img_shp, item_sz, px_sz=px_sz, slc_sz=slc_sz, ovlp=ovlp)

    return batch_sz, in_slc_shp, in_slc_shp_um, px_rsz_ratio, ovlp, ovlp_rsz


def crop(img, rng, ovlp=None, flip=()):
    """
    Shrink image slice at total volume boundaries, for overall shape consistency.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    rng: tuple
        3D slice index range

    ovlp: numpy.ndarray (axis order=(Z,Y,X), dtype=int)
        overlapping range between image slices [px]

    flip: tuple
        flipped axes

    Returns
    -------
    img_out: numpy.ndarray
        cropped microscopy image
    """

    # delete overlapping boundaries
    if ovlp is not None:
        img = img[ovlp[0]:img.shape[0] - ovlp[0], ovlp[1]:img.shape[1] - ovlp[1], ovlp[2]:img.shape[2] - ovlp[2]]

    # check image shape and output index ranges
    crop_rng = np.zeros(shape=(3,), dtype=np.int64)
    for s in range(3):
        crop_rng[s] = img.shape[s] - np.arange(rng[s].start, rng[s].stop, rng[s].step).size

    # crop image if required
    img_out = img[crop_rng[0] or None:, ...] if 0 in flip else img[:-crop_rng[0] or None, ...]
    img_out = img_out[:, crop_rng[1] or None:, ...] if 1 in flip else img_out[:, :-crop_rng[1] or None, ...]
    img_out = img_out[:, :, crop_rng[2] or None:, ...] if 2 in flip else img_out[:, :, :-crop_rng[2] or None, ...]

    return img_out


def crop_lst(img_lst, rng, ovlp=None, flip=()):
    """
    Shrink list of image slices at total volume boundaries, for overall shape consistency.

    Parameters
    ----------
    img_lst: list
        list of images to be cropped

    rng: tuple
        3D slice index range

    ovlp: int
        overlapping range between image slices along each axis [px]

    flip: tuple
        flipped axes

    Returns
    -------
    img_lst: list
        list of cropped image slices
    """

    for s, img in enumerate(img_lst):
        if img is not None:
            img_lst[s] = crop(img, rng, ovlp=ovlp, flip=flip)

    return img_lst


def generate_slice_lists(in_slc_shp, img_shp, batch_sz, px_rsz_ratio, ovlp=0, msk_bc=False, jobs=None):
    """
    Generate image slice ranges for the Frangi filtering stage.

    Parameters
    ----------
    in_slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    batch_sz: int
        slice batch size

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    ovlp: int
        overlapping range between image slices along each axis [px]

    msk_bc: bool
        if True, mask neuronal bodies
        in the optionally provided image channel

    jobs: int
        number of parallel jobs (threads)

    Returns
    -------
    slc_rng: dict
        in_rng_lst: list
            list of input slice index ranges

        in_pad_lst: list
            list of slice padding arrays

        out_rng_lst: list
            list of output slice index ranges

        bc_rng_lst: list
            (optional) list of neuronal body slice index ranges

    out_slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the processed image slices [px]

    tot_slc_num: int
        total number of analyzed image slices

    batch_sz: int
        adjusted slice batch size
    """

    # adjust output shapes
    out_slc_shp = np.ceil(np.multiply(px_rsz_ratio, in_slc_shp)).astype(int)
    out_img_shp = np.ceil(np.multiply(px_rsz_ratio, img_shp)).astype(int)

    # number of image slices along each axis
    slc_per_dim = np.floor(np.divide(img_shp, in_slc_shp)).astype(int)

    # number of logical cores
    if jobs is None:
        jobs = get_available_cores()

    # initialize empty range lists
    in_pad_lst = list()
    in_rng_lst = list()
    bc_rng_lst = list()
    out_rng_lst = list()
    for z, y, x in product(range(slc_per_dim[0]), range(slc_per_dim[1]), range(slc_per_dim[2])):

        # index ranges of analyzed image slices (with padding)
        in_rng, pad = compute_slice_range((z, y, x), in_slc_shp, img_shp, slc_per_dim, ovlp=ovlp)
        in_rng_lst.append(in_rng)
        in_pad_lst.append(pad)

        # output index ranges
        out_rng, _ = compute_slice_range((z, y, x), out_slc_shp, out_img_shp, slc_per_dim)
        out_rng_lst.append(out_rng)

        # (optional) neuronal body channel
        if msk_bc:
            bc_rng, _ = compute_slice_range((z, y, x), in_slc_shp, img_shp, slc_per_dim)
            bc_rng_lst.append(bc_rng)
        else:
            bc_rng_lst.append(None)

    # total number of slices
    tot_slc_num = len(in_rng_lst)

    # adjust slice batch size
    if batch_sz > tot_slc_num:
        batch_sz = tot_slc_num

    # fill slice range dictionary
    slc_rng = dict()
    slc_rng['in_rng_lst'] = in_rng_lst
    slc_rng['in_pad_lst'] = in_pad_lst
    slc_rng['out_rng_lst'] = out_rng_lst
    slc_rng['bc_rng_lst'] = bc_rng_lst

    return slc_rng, out_slc_shp, tot_slc_num, batch_sz


def slice_image(img, rng, ch, ch_ax, ts_msk=None):
    """
    Slice desired channel from input image volume.

    Parameters
    ----------
    img: numpy.ndarray
        3D microscopy image

    rng: tuple (dtype=int)
        3D index ranges

    ch: int
        channel

    ch_ax: int
        RGB image channel axis (either 1 or 3, or None for grayscale images)

    ts_msk: numpy.ndarray (dtype=bool)
        tissue binary mask

    Returns
    -------
    img_slc: numpy.ndarray
        image slice

    ts_msk_slc: numpy.ndarray (dtype=bool)
        tissue mask slice
    """
    z_rng, r_rng, c_rng = rng

    if ch_ax is None:
        img_slc = img[z_rng, r_rng, c_rng]
    elif ch_ax == 1:
        img_slc = img[z_rng, ch, r_rng, c_rng]
    else:
        img_slc = img[z_rng, r_rng, c_rng, ch]

    ts_msk_slc = ts_msk[r_rng, c_rng] if ts_msk is not None else None

    return img_slc, ts_msk_slc
