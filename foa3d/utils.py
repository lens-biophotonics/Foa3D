import gc
import tempfile
from multiprocessing import cpu_count
from os import environ, path, unlink
from shutil import rmtree
from time import perf_counter

import numpy as np
from astropy.visualization import make_lupton_rgb
from h5py import File
from joblib import dump, load
from matplotlib.colors import hsv_to_rgb
from skimage.filters import (threshold_li, threshold_niblack,
                             threshold_sauvola, threshold_triangle,
                             threshold_yen)


def create_background_mask(img, method='yen', black_bg=False):
    """
    Compute background mask.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        microscopy volume image

    method: str
        image thresholding method

    black_bg: bool
        generate foreground mask

    Returns
    -------
    bg_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=bool)
        boolean background mask
    """

    # select thresholding method
    if method == 'li':
        thresh = threshold_li(img)
    elif method == 'niblack':
        thresh = threshold_niblack(img, window_size=15, k=0.2)
    elif method == 'sauvola':
        thresh = threshold_sauvola(img, window_size=15, k=0.2, r=None)
    elif method == 'triangle':
        thresh = threshold_triangle(img, nbins=256)
    elif method == 'yen':
        thresh = threshold_yen(img, nbins=256)
    else:
        raise ValueError("Unsupported thresholding method!!!")

    # compute mask
    bg_msk = img >= thresh if black_bg else img < thresh

    return bg_msk


def create_hdf5_dset(dset_shape, dtype, chunks=True, name='tmp', tmp=None):
    """
    Create HDF5 dataset.

    Parameters
    ----------
    dset_shape: tuple (dtype: int)
        dataset shape

    dtype:
        data type of the HDF5 dataset

    chunks: tuple (dtype: int) or bool
        shape of the chunked storage layout (default: auto chunking)

    name: str
        filename

    tmp: str
        path to existing temporary saving directory

    Returns
    -------
    dset:
        HDF5 dataset

    file_path: str
        path to the HDF5 file
    """
    if tmp is None:
        tmp = tempfile.mkdtemp()

    file_path = path.join(tmp, '{}.h5'.format(name))
    file = File(file_path, 'w')
    dset = file.create_dataset(None, dset_shape, chunks=tuple(chunks), dtype=dtype)

    return dset, file_path


def create_memory_map(shape, dtype, name='tmp', tmp_dir=None, arr=None, mmap_mode='r+'):
    """
    Create a memory-map to an array stored in a binary file on disk.

    Parameters
    ----------
    shape: tuple
        shape of the stored array

    dtype:
        data-type used to interpret the file contents

    name: str
        optional temporary filename

    tmp_dir: str
        temporary file directory

    arr: numpy.ndarray
        array to be mapped

    mmap_mode: str
        file opening mode

    Returns
    -------
    mmap: NumPy memory map
        memory-mapped array
    """

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    mmap_path = path.join(tmp_dir, name + '.mmap')

    if path.exists(mmap_path):
        unlink(mmap_path)

    if arr is None:
        arr = np.zeros(tuple(shape), dtype=dtype)

    _ = dump(arr, mmap_path)
    mmap = load(mmap_path, mmap_mode=mmap_mode)
    del arr
    _ = gc.collect()

    return mmap


def get_available_cores():
    """
    Return the number of available logical cores.

    Returns
    -------
    num_cpu: int
        number of available cores
    """

    num_cpu = environ.pop('OMP_NUM_THREADS', default=None)
    num_cpu = cpu_count() if num_cpu is None else int(num_cpu)

    return num_cpu


def get_item_size(dtype):
    """
    Get the item size in bytes of a data type.

    Parameters
    ----------
    dtype: str
        data type identifier

    Returns
    -------
    item_sz: int
        item size in bytes
    """

    # data type lists
    lst_1 = ['uint8', 'int8']
    lst_2 = ['uint16', 'int16', 'float16', np.float16]
    lst_3 = ['uint32', 'int32', 'float32', np.float32]
    lst_4 = ['uint64', 'int64', 'float64', np.float64]

    if dtype in lst_1:
        item_sz = 1
    elif dtype in lst_2:
        item_sz = 2
    elif dtype in lst_3:
        item_sz = 4
    elif dtype in lst_4:
        item_sz = 8
    else:
        raise ValueError("Unsupported data type!")

    return item_sz


def delete_tmp_folder(tmp_dir):
    """
    Delete temporary folder.

    Parameters
    ----------
    tmp_dir: str
        path to temporary folder to be removed

    Returns
    -------
    None
    """
    try:
        rmtree(tmp_dir)
    except OSError:
        pass


def divide_nonzero(nd_array1, nd_array2, new_val=1e-10):
    """
    Divide two arrays handling zero denominator values.

    Parameters
    ----------
    nd_array1: numpy.ndarray
        dividend array

    nd_array2: numpy.ndarray
        divisor array

    new_val: float
        substituted value

    Returns
    -------
    divided: numpy.ndarray
        divided array
    """

    divisor = np.copy(nd_array2)
    divisor[divisor == 0] = new_val
    divided = np.divide(nd_array1, divisor)

    return divided


def elapsed_time(start_time):
    """
    Compute elapsed time from input start reference.

    Parameters
    ----------
    start_time: float
        start time reference

    Returns
    -------
    tot: float
        total time [s]

    hrs: int
        hours

    mins: int
        minutes

    secs: float
        seconds
    """

    stop_time = perf_counter()
    tot = stop_time - start_time

    secs = tot % 86400
    hrs = int(secs // 3600)
    secs %= 3600
    mins = int(secs // 60)
    secs %= 60

    return tot, hrs, mins, secs


def fwhm_to_sigma(fwhm):
    """
    Compute the standard deviation of a Gaussian distribution
    from its FWHM value.

    Parameters
    ----------
    fwhm: float
        full width at half maximum

    Returns
    -------
    sigma: float
        standard deviation
    """
    sigma = np.sqrt(np.square(fwhm) / (8 * np.log(2)))

    return sigma


def get_item_bytes(data):
    """
    Retrieve data item size in bytes.

    Parameters
    ----------
    data: numpy.ndarray or HDF5 dataset
        input data

    Returns
    -------
    bts: int
        item size in bytes
    """

    # type byte size
    try:
        bts = int(np.iinfo(data.dtype).bits / 8)
    except ValueError:
        bts = int(np.finfo(data.dtype).bits / 8)

    return bts


def get_output_prefix(scales_um, alpha, beta, gamma):
    """
    Generate the output filename prefix including
    pipeline configuration information.

    Parameters
    ----------
    scales_um: list (dtype=float)
        analyzed spatial scales in [μm]

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    Returns
    -------
    pfx: str
        pipeline configuration prefix
    """

    pfx = 'sc'
    for s in scales_um:
        pfx += '{}_'.format(s)

    pfx = 'a{}_b{}_g{}_{}'.format(alpha, beta, gamma, pfx)

    return pfx


def normalize_angle(angle, lower=0.0, upper=360.0, dtype=None):
    """
    Normalize angle to [lower, upper) range.

    Parameters
    ----------
    angle: float or list (dtype=float)
        angular value(s) to be normalized (in degrees)

    lower: float
        lower limit (default: 0.0)

    upper: float
        upper limit (default: 360.0)

    dtype:
        output data type

    Returns
    -------
    norm_angle: float or list (dtype=float)
        angular value(s) (in degrees) normalized within [lower, upper)

    Raises
    ------
    ValueError
      if lower >= upper
    """
    # convert to array if needed
    is_list = False
    if np.isscalar(angle):
        angle = np.array(angle)
    elif isinstance(angle, list):
        angle = np.array(angle)
        is_list = True

    # check limits
    if lower >= upper:
        raise ValueError("Invalid lower and upper limits: (%s, %s)" % (lower, upper))

    # view
    norm_angle = angle

    # correction 1
    c1 = np.logical_or(angle > upper, angle == lower)
    angle[c1] = lower + abs(angle[c1] + upper) % (abs(lower) + abs(upper))

    # correction 2
    c2 = np.logical_or(angle < lower, angle == upper)
    angle[c2] = upper - abs(angle[c2] - lower) % (abs(lower) + abs(upper))

    # correction 3
    angle[angle == upper] = lower

    # cast to desired data type
    if dtype is not None:
        norm_angle = norm_angle.astype(dtype)

    # convert back to list
    if is_list:
        norm_angle = list(norm_angle)

    return norm_angle


def normalize_image(img, max_out_val=255.0, dtype=np.uint8):
    """
    Normalize image data.

    Parameters
    ----------
    img: numpy.ndarray
        input image

    max_out_val: float
        maximum output value

    dtype:
        output data type

    Returns
    -------
    norm_img: numpy.ndarray
        normalized image
    """

    # get min and max values
    min_val = np.min(img)
    max_val = np.max(img)

    # normalize
    if max_val != 0:
        if max_val != min_val:
            norm_img = (((img - min_val) / (max_val - min_val)) * max_out_val).astype(dtype)
        else:
            norm_img = ((img / max_val) * max_out_val).astype(dtype)
    else:
        norm_img = img.astype(dtype)

    return norm_img


def hsv_orient_cmap(vec_img):
    """
    Compute HSV colormap of vector orientations from 3D vector field.

    Parameters
    ----------
    vec_img: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        orientation vectors

    Returns
    -------
    rgb_map: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
        orientation color map
    """

    # extract planar components
    vy, vx = (vec_img[..., 1], vec_img[..., 2])

    # compute the in-plane versor length
    vxy_abs = np.sqrt(np.square(vx) + np.square(vy))
    vxy_abs = divide_nonzero(vxy_abs, np.max(vxy_abs))

    # compute the in-plane angular orientation
    vxy_ang = normalize_angle(np.arctan2(vy, vx), lower=0, upper=np.pi, dtype=np.float32)
    vxy_ang = np.divide(vxy_ang, np.pi)

    # initialize colormap
    rgb_map = np.zeros(shape=tuple(list(vec_img.shape[:-1]) + [3]), dtype=np.uint8)
    for z in range(vec_img.shape[0]):

        # generate HSV colormap slice by slice
        hsv_map = np.stack((vxy_ang[z], vxy_abs[z], vxy_abs[z]), axis=-1)

        # convert to RGB map with 8-bit precision
        rgb_map[z] = (255.0 * hsv_to_rgb(hsv_map)).astype(np.uint8)

    return rgb_map


def ceil_to_multiple(number, multiple):
    """
    Round up number to the nearest multiple.

    Parameters
    ----------
    number:
        number to be rounded

    multiple:
        the input number will be rounded
        to the nearest multiple higher than this value

    Returns
    -------
    rounded:
        rounded up number
    """
    rounded = multiple * np.ceil(number / multiple)

    return rounded


def transform_axes(nd_array, flipped=None, swapped=None, expand=None):
    """
    Manipulate axes and dimensions of the input array.
    The transformation sequence is:
    axes flip >>> axes swap >>> dimensions expansion.

    Parameters
    ----------
    nd_array: numpy.ndarray
        input data array

    swapped: tuple (dtype=int)
        axes to be swapped

    flipped: tuple (dtype=int)
        axes to be flipped

    expand: int
        insert new axis at this position

    Returns
    -------
    nd_array: numpy.ndarray
        transformed data array
    """

    if flipped is not None:
        nd_array = np.flip(nd_array, axis=flipped)

    if swapped is not None:
        swap_src, swap_dest = swapped
        nd_array = np.swapaxes(nd_array, swap_src, swap_dest)

    if expand is not None:
        nd_array = np.expand_dims(nd_array, axis=expand)

    return nd_array


def rgb_orient_cmap(vec_img, minimum=0, stretch=1, q=8):
    """
    Compute RGB colormap of orientation vector components from 3D vector field.

    Parameters
    ----------
    vec_img: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        n-dimensional array of orientation vectors

    minimum: int
        intensity that should be mapped to black (a scalar or array for R, G, B)

    stretch: int
        linear stretch of the image

    q: int
        asinh softening parameter

    Returns
    -------
    rgb_map: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
        orientation color map
    """

    # take absolute value
    vec_img = np.abs(vec_img)

    # initialize colormap
    rgb_map = np.zeros(shape=vec_img.shape, dtype=np.uint8)
    for z in range(vec_img.shape[0]):

        # generate colormap slice by slice
        img_r, img_g, img_b = (vec_img[z, :, :, 2], vec_img[z, :, :, 1], vec_img[z, :, :, 0])
        rgb_map[z] = make_lupton_rgb(img_r, img_g, img_b, minimum=minimum, stretch=stretch, Q=q)

    return rgb_map
