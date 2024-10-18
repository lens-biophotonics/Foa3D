from time import perf_counter

import numpy as np
from joblib import Parallel, delayed

from foa3d.frangi import frangi_filter
from foa3d.input import get_image_info, get_frangi_config, get_resource_config
from foa3d.odf import (compute_odf_map, generate_odf_background,
                       get_sph_harm_ncoeff, get_sph_harm_norm_factors)
from foa3d.output import save_frangi_arrays, save_odf_arrays
from foa3d.preprocessing import correct_image_anisotropy
from foa3d.printing import print_flsh, print_frangi_info, print_odf_info
from foa3d.slicing import (config_frangi_batch, generate_slice_lists,
                           crop, crop_lst, slice_image)
from foa3d.utils import (create_background_mask, create_memory_map,
                         divide_nonzero, elapsed_time,
                         get_available_cores, hsv_orient_cmap,
                         rgb_orient_cmap)


slc_cnt = 0


def parallel_frangi_on_slices(cli_args, save_dirs, img_data, ch_ax, img_name, ts_msk, is_vec):
    """
    Apply 3D Frangi filtering to basic TPFM image slices using parallel threads.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    save_dirs: str
        saving directories

    img_data: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
        3D microscopy image

    ch_ax: int
        RGB image channel axis (either 1, 3, or None for grayscale images)

    img_name: str
        name of the 3D microscopy image

    ts_msk: numpy.ndarray (dtype=bool)
        tissue reconstruction binary mask

    is_vec: bool
        vector field flag

    Returns
    -------
    out_img: dict
        output image dictionary
    """
    # skip Frangi filter stage if orientation vectors were directly provided as input
    if is_vec:
        out_img = dict()
        out_img['fbr_vec'] = img_data
        out_img['iso_fbr'] = None
        out_img['px_sz'] = None

    else:

        # get resources configuration
        ram, jobs = get_resource_config(cli_args)

        # get Frangi filter configuration
        alpha, beta, gamma, frangi_sigma, frangi_sigma_um, smooth_sigma, px_sz, px_sz_iso, \
            z_rng, bc_ch, fb_ch, fb_thr, msk_bc, hsv_vec_cmap, exp_all = get_frangi_config(cli_args)

        # get info about the input microscopy image
        img_shp, img_shp_um, img_item_sz, fb_ch, msk_bc = get_image_info(img_data, px_sz, msk_bc, fb_ch, ch_ax)

        # configure the batch of basic image slices analyzed using parallel threads
        batch_sz, in_slc_shp, in_slc_shp_um, px_rsz_ratio, ovlp, ovlp_rsz = \
            config_frangi_batch(px_sz, px_sz_iso, img_shp, img_item_sz, smooth_sigma, frangi_sigma_um,
                                ram=ram, jobs=jobs)

        # get info about the processed image slices
        slc_rng, out_slc_shp, tot_slc_num, batch_sz = \
            generate_slice_lists(in_slc_shp, img_shp, batch_sz, px_rsz_ratio, ovlp, msk_bc=msk_bc, jobs=jobs)

        # initialize output arrays
        out_img, z_sel = init_frangi_arrays(img_shp, out_slc_shp, px_rsz_ratio, save_dirs['tmp'],
                                             z_rng=z_rng, msk_bc=msk_bc, exp_all=exp_all)

        # print Frangi filter configuration
        print_frangi_info(alpha, beta, gamma, frangi_sigma_um, img_shp_um, in_slc_shp_um, tot_slc_num,
                          px_sz, img_item_sz, msk_bc)

        # parallel Frangi filter-based fiber orientation analysis of microscopy image slices
        print_flsh(f"[Parallel(n_jobs={batch_sz})]: Using backend ThreadingBackend with {batch_sz} concurrent workers.")
        start_time = perf_counter()
        with Parallel(n_jobs=batch_sz, prefer='threads', require='sharedmem') as parallel:
            parallel(
                delayed(fiber_analysis)(i, img_data, ovlp_rsz,
                                        smooth_sigma, frangi_sigma, px_rsz_ratio, z_sel, **slc_rng, **out_img,
                                        ts_msk=ts_msk, bc_ch=bc_ch, fb_ch=fb_ch, fb_thr=fb_thr, ch_ax=ch_ax,
                                        alpha=alpha, beta=beta, gamma=gamma, msk_bc=msk_bc, hsv_vec_cmap=hsv_vec_cmap,
                                        print_info=(start_time, batch_sz, tot_slc_num))
                for i in range(tot_slc_num))

        # save output arrays
        save_frangi_arrays(save_dirs['frangi'], img_name, out_img, px_sz_iso, ram=ram)

        # add isotropic pixel size
        out_img['px_sz'] = px_sz_iso

    return out_img


def init_frangi_arrays(img_shp, slc_shp, rsz_ratio, tmp_dir,
                        z_rng=(0, None), msk_bc=False, exp_all=False):
    """
    Initialize the output datasets of the Frangi filtering stage.

    Parameters
    ----------
    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    tmp_dir: str
        temporary file directory

    z_rng: int
        output z-range in [px]

    msk_bc: bool
        if True, mask neuronal bodies
        in the optionally provided image channel

    exp_all: bool
        export all images

    Returns
    -------
    out_img: dict

        fbr_vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
            initialized fiber orientation volume image

        fbr_vec_clr: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=uint8)
            initialized orientation colormap image

        fa_img: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            initialized fractional anisotropy image

        frangi_img: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            initialized Frangi-enhanced image

        fbr_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            initialized fiber mask image

        iso_fbr: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            initialized fiber image (isotropic resolution)

        bc_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            initialized soma mask image

    z_sel: NumPy slice object
        selected z-depth range
    """
    # shape copies
    img_shp = img_shp.copy()
    slc_shp = slc_shp.copy()

    # adapt output z-axis shape if required
    z_min, z_max = z_rng
    if z_min != 0 or z_max is not None:
        if z_max is None:
            z_max = slc_shp[0]
        img_shp[0] = z_max - z_min
    z_sel = slice(z_min, z_max, 1)

    # output shape
    img_dims = len(img_shp)
    tot_shp = tuple(np.ceil(rsz_ratio * img_shp).astype(int))

    # fiber channel arrays
    iso_fbr_img = create_memory_map(tot_shp, dtype='uint8', name='iso_fiber', tmp_dir=tmp_dir)
    if exp_all:
        frangi_img = create_memory_map(tot_shp, dtype='uint8', name='frangi', tmp_dir=tmp_dir)
        fbr_msk = create_memory_map(tot_shp, dtype='uint8', name='fbr_msk', tmp_dir=tmp_dir)
        fa_img = create_memory_map(tot_shp, dtype='float32', name='fa', tmp_dir=tmp_dir)

        # soma channel array
        if msk_bc:
            bc_msk = create_memory_map(tot_shp, dtype='uint8', name='bc_msk', tmp_dir=tmp_dir)
        else:
            bc_msk = None
    else:
        frangi_img, fbr_msk, fa_img, bc_msk = (None, None, None, None)

    # fiber orientation arrays
    vec_shape = tot_shp + (img_dims,)
    fbr_vec_img = create_memory_map(vec_shape, dtype='float32', name='fiber_vec', tmp_dir=tmp_dir)
    fbr_vec_clr = create_memory_map(vec_shape, dtype='uint8', name='fiber_cmap', tmp_dir=tmp_dir)

    # fill output image dictionary
    out_img = dict()
    out_img['fbr_vec'] = fbr_vec_img
    out_img['fbr_vec_clr'] = fbr_vec_clr
    out_img['fa_img'] = fa_img
    out_img['frangi_img'] = frangi_img
    out_img['iso_fbr'] = iso_fbr_img
    out_img['fbr_msk'] = fbr_msk
    out_img['bc_msk'] = bc_msk

    return out_img, z_sel


def fiber_analysis(i, img, ovlp, smooth_sigma, scales_px, px_rsz_ratio, z_sel, in_rng_lst, bc_rng_lst, out_rng_lst, in_pad_lst,
                   fbr_vec, fbr_vec_clr, fa_img, frangi_img, iso_fbr, fbr_msk, bc_msk, ts_msk=None,
                   bc_ch=0, fb_ch=1, ch_ax=None, alpha=0.05, beta=1, gamma=100, dark=False, msk_bc=False,
                   print_info=None, hsv_vec_cmap=False, pad_mode='reflect', fb_thr='li', bc_thr='yen',
                   ts_thr=0.0001, verbose=10):
    """
    Conduct a Frangi-based fiber orientation analysis on basic slices selected from the whole microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X))
        fiber fluorescence volume image

    in_rng: NumPy slice object
        input image range (fibers)

    bc_rng: NumPy slice object
        input image range (neurons)

    out_rng: NumPy slice object
        output range

    pad: numpy.ndarray (axis order=(Z,Y,X))
        image padding array

    ovlp: int
        overlapping range between slices along each axis [px]

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the smoothing Gaussian filter [px]

    scales_px: numpy.ndarray (dtype=int)
        spatial scales [px]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    z_sel: NumPy slice object
        selected z-depth range

    fbr_vec: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    fbr_vec_clr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap image

    fa_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        Frangi-enhanced volume image (fiber probability volume)

    iso_fbr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    fbr_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image

    bc_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        soma mask image

    ts_msk: numpy.ndarray (dtype=bool)
        tissue reconstruction binary mask

    bc_ch: int
        neuronal bodies channel

    fb_ch: int
        myelinated fibers channel

    ch_ax: int
        RGB image channel axis (either 1 or 3, or None for grayscale images)

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    dark: bool
        if True, enhance black 3D tubular structures
        (i.e., negative contrast polarity)

    hsv_vec_cmap: bool
        if True, generate an HSV orientation color map based on XY-plane orientation angles
        (instead of an RGB map using the cartesian components of the estimated vectors)

    msk_bc: bool
        if True, mask neuronal bodies exploiting the autofluorescence
        signal of lipofuscin pigments

    print_info: tuple
        optional printed information

    pad_mode: str
        image padding mode

    is_tiled: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    fb_thr: str
         thresholding method applied to the myelinated fibers channel

    bc_thr: str
        thresholding method applied to the neuronal bodies channel

    ts_thr: float
        relative threshold on non-zero tissue pixels

    verbose: int
        verbosity level (print progress every N=verbose image slices)

    Returns
    -------
    None
    """
    in_rng = in_rng_lst[i]
    pad = in_pad_lst[i]
    out_rng = out_rng_lst[i]
    bc_rng = bc_rng_lst[i]

    # slice fiber image slice
    fbr_slc, ts_msk_slc = slice_image(img, in_rng, fb_ch, ch_ax, ts_msk=ts_msk)

    # skip background slice
    crt = np.count_nonzero(ts_msk_slc) / np.prod(ts_msk_slc.shape) > ts_thr \
        if ts_msk_slc is not None else np.max(fbr_slc) != 0
    if crt:

        # preprocess fiber slice
        iso_fbr_slc, pad_rsz, ts_msk_slc_rsz = \
            correct_image_anisotropy(fbr_slc, px_rsz_ratio, sigma=smooth_sigma, pad=pad, ts_msk=ts_msk_slc)

        # pad fiber slice if required
        if pad_rsz is not None:
            iso_fbr_slc = np.pad(iso_fbr_slc, pad_rsz, mode=pad_mode)

            # pad tissue mask if available
            if ts_msk_slc_rsz is not None:
                ts_msk_slc_rsz = np.pad(ts_msk_slc_rsz, pad_rsz, mode='constant')

        # 3D Frangi filter
        frangi_slc, fbr_vec_slc, eigenval_slc = \
            frangi_filter(iso_fbr_slc, scales_px=scales_px, alpha=alpha, beta=beta, gamma=gamma, dark=dark)

        # crop resulting slices
        iso_fbr_slc, frangi_slc, fbr_vec_slc, eigenval_slc, ts_msk_slc_rsz = \
            crop_lst([iso_fbr_slc, frangi_slc, fbr_vec_slc, eigenval_slc, ts_msk_slc_rsz], out_rng, ovlp)

        # generate fractional anisotropy image
        fa_slc = compute_fractional_anisotropy(eigenval_slc)

        # generate fiber orientation color map
        fbr_clr_slc = hsv_orient_cmap(fbr_vec_slc) if hsv_vec_cmap else rgb_orient_cmap(fbr_vec_slc)

        # remove background
        fbr_vec_slc, fbr_clr_slc, fa_slc, fbr_msk_slc = \
            mask_background(frangi_slc, fbr_vec_slc, fbr_clr_slc, fa_slc,
                            ts_msk=ts_msk_slc_rsz, method=fb_thr, invert=False)

        # (optional) neuronal body masking
        if msk_bc:

            # get soma image slice
            bc_slc, _ = slice_image(img, bc_rng, bc_ch, ch_ax)

            # resize soma slice (lateral blurring and downsampling)
            iso_bc_slc, _, _ = correct_image_anisotropy(bc_slc, px_rsz_ratio)

            # crop isotropized soma slice
            iso_bc_slc = crop(iso_bc_slc, out_rng)

            # mask neuronal bodies
            fbr_vec_slc, fbr_clr_slc, fa_slc, bc_msk_slc = \
                mask_background(iso_bc_slc, fbr_vec_slc, fbr_clr_slc, fa_slc, method=bc_thr, invert=True)

        else:
            bc_msk_slc = None

        # fill memory-mapped output arrays
        write_frangi_arrays(fbr_vec, fbr_vec_clr, fa_img, frangi_img, iso_fbr, fbr_msk, bc_msk,
                            fbr_vec_slc, fbr_clr_slc, fa_slc, frangi_slc, iso_fbr_slc,
                            fbr_msk_slc, bc_msk_slc, out_rng, z_sel)

    # print progress
    if print_info is not None:
        global slc_cnt
        slc_cnt += 1

        # print only every N=verbose image slices
        start_time, batch_sz, tot_slc_num = print_info
        if slc_cnt % verbose == 0 or slc_cnt == tot_slc_num:
            prog_prc = 100 * slc_cnt / tot_slc_num
            _, hrs, mins, secs = elapsed_time(start_time)
            print_flsh(f"[Parallel(n_jobs={batch_sz})]:\t{slc_cnt}/{tot_slc_num} done\t|\telapsed: {hrs} hrs {mins} min {secs:.1f} s\t{prog_prc:.1f}%")


def compute_fractional_anisotropy(eigenval):
    """
    Compute structure tensor fractional anisotropy
    as in Schilling et al. (2018).

    Parameters
    ----------
    eigenval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        structure tensor eigenvalues (best local spatial scale)

    Returns
    -------
    fa: numpy.ndarray (shape=(3,), dtype=float)
        fractional anisotropy
    """

    fa = np.sqrt(0.5 * divide_nonzero(
                 np.square((eigenval[..., 0] - eigenval[..., 1])) +
                 np.square((eigenval[..., 0] - eigenval[..., 2])) +
                 np.square((eigenval[..., 1] - eigenval[..., 2])),
                 np.sum(eigenval ** 2, axis=-1)))

    return fa


def mask_background(img, fbr_vec_slc, fbr_clr_slc, fa_slc=None, method='yen', invert=False, ts_msk=None):
    """
    Mask fiber orientation arrays.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        fiber (or neuron) fluorescence volume image

    fbr_vec_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        fiber orientation vector slice

    fbr_clr_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
        fiber orientation colormap slice

    fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        fractional anisotropy slice

    method: str
        thresholding method (refer to skimage.filters)

    invert: bool
        mask inversion flag

    ts_msk: numpy.ndarray (dtype=bool)
        tissue reconstruction binary mask

    Returns
    -------
    fbr_vec_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        orientation vector patch (masked)

    orientcol_slice: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap patch (masked)

    fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        fractional anisotropy patch (masked)

    bg: numpy.ndarray (axis order=(Z,Y,X), dtype=bool)
        background mask
    """

    # generate background mask
    bg = create_background_mask(img, method=method)

    # apply tissue reconstruction mask, when provided
    if ts_msk is not None:
        bg = np.logical_or(bg, np.logical_not(ts_msk))

    # invert mask
    if invert:
        bg = np.logical_not(bg)

    # apply mask to input arrays
    fbr_vec_slc[bg, :] = 0
    fbr_clr_slc[bg, :] = 0
    fa_slc[bg] = 0

    return fbr_vec_slc, fbr_clr_slc, fa_slc, bg


def write_frangi_arrays(fbr_vec_img, fbr_vec_clr, fa_img, frangi_img, iso_fbr_img, fbr_msk, bc_msk, fbr_vec_slc,
                        fbr_clr_slc, fa_slc, frangi_slc, iso_fbr_slc, fbr_msk_slc, bc_msk_slc, out_rng, z_sel):
    """
    Fill the memory-mapped output arrays of the Frangi filter stage.

    Parameters
    ----------
    fbr_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    fbr_vec_clr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap image

    fa_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        Frangi-enhanced image (fiber probability image)

    iso_fbr_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    fbr_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image

    bc_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        soma mask image

    fbr_vec_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        fiber orientation vector image slice

    fbr_clr_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        orientation colormap image slice

    fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        fractional anisotropy image slice

    frangi_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        Frangi-enhanced image slice

    iso_fbr_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image slice

    fbr_msk_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image slice

    bc_msk_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        soma mask image slice

    out_rng: tuple
        3D slice output index range

    z_sel: NumPy slice object
        selected z-depth range

    Returns
    -------
    None
    """

    # fill memory-mapped output arrays
    vec_rng_out = tuple(np.append(out_rng, slice(0, 3, 1)))
    fbr_vec_img[vec_rng_out] = fbr_vec_slc[z_sel, ...]
    fbr_vec_clr[vec_rng_out] = fbr_clr_slc[z_sel, ...]
    iso_fbr_img[out_rng] = iso_fbr_slc[z_sel, ...].astype(np.uint8)

    # optional output images: Frangi filter response
    if frangi_img is not None:
        frangi_img[out_rng] = (255 * frangi_slc[z_sel, ...]).astype(np.uint8)

    # optional output images: fractional anisotropy
    if fa_img is not None:
        fa_img[out_rng] = fa_slc[z_sel, ...]

    # optional output images: fiber mask
    if fbr_msk is not None:
        fbr_msk[out_rng] = (255 * (1 - fbr_msk_slc[z_sel, ...])).astype(np.uint8)

    # optional output images: neuronal soma mask
    if bc_msk is not None:
        bc_msk[out_rng] = (255 * bc_msk_slc[z_sel, ...]).astype(np.uint8)


def odf_analysis(fbr_vec_img, iso_fbr_img, px_sz_iso, save_dirs, img_name, odf_scale_um, odf_norm,
                 odf_deg=6, exp_all=False):
    """
    Estimate 3D fiber ODFs from basic orientation data chunks using parallel threads.

    Parameters
    ----------
    fbr_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vectors

    iso_fbr_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber volume

    px_sz_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    save_dirs: dict
        saving directories
        ('frangi': Frangi filter, 'odf': ODF analysis, 'tmp': temporary files)

    img_name: str
        name of the input volume image

    odf_scale_um: float
        fiber ODF resolution (super-voxel side [μm])

    odf_norm: numpy.ndarray (dtype: float)
        2D array of spherical harmonics normalization factors

    odf_deg: int
        degrees of the spherical harmonics series expansion

    exp_all: bool
        export all images

    Returns
    -------
    None
    """

    # derive the ODF kernel size in [px]
    odf_scale = int(np.ceil(odf_scale_um / px_sz_iso[0]))

    # initialize ODF analysis output volumes
    odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, disarray, tensor \
        = init_odf_arrays(fbr_vec_img.shape[:-1], save_dirs['tmp'], odf_scale=odf_scale, odf_degrees=odf_deg, exp_all=exp_all)

    # generate downsampled background for MRtrix3 mrview
    bg_img = fbr_vec_img if iso_fbr_img is None else iso_fbr_img
    generate_odf_background(bg_img, bg_mrtrix, vxl_side=odf_scale)

    # compute ODF coefficients
    odf = compute_odf_map(fbr_vec_img, odf, odi_pri, odi_sec, odi_tot, odi_anis, disarray, tensor,
                          odf_scale, odf_norm, odf_deg=odf_deg)

    # save memory maps to file
    save_odf_arrays(odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, disarray,
                    px_sz_iso, save_dirs['odf'], img_name, odf_scale_um)


def init_odf_arrays(vec_img_shp, tmp_dir, odf_scale, odf_degrees=6, exp_all=False):
    """
    Initialize the output datasets of the ODF analysis stage.

    Parameters
    ----------
    vec_img_shp: tuple
        vector volume shape [px]

    tmp_dir: str
        temporary file directory

    odf_scale: int
        fiber ODF resolution (super-voxel side [px])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    exp_all: bool
        export all images

    Returns
    -------
    odf: NumPy memory-map object (axis order=(X,Y,Z,C), dtype=float32)
        initialized array of ODF spherical harmonics coefficients

    bg_mrtrix: NumPy memory-map object (axis order=(X,Y,Z), dtype=uint8)
        initialized background for ODF visualization in MRtrix3

    odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        initialized array of primary orientation dispersion parameters

    odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        initialized array of secondary orientation dispersion parameters

    odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        initialized array of total orientation dispersion parameters

    odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        initialized array of orientation dispersion anisotropy parameters

    disarray: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        local angular disarray

    vec_tensor_eigen: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        initialized array of fiber orientation tensor eigenvalues
    """

    # ODI maps shape
    odi_shp = tuple(np.ceil(np.divide(vec_img_shp, odf_scale)).astype(int))

    # create downsampled background memory map
    bg_shp = tuple(np.flip(odi_shp))
    bg_mrtrix = create_memory_map(bg_shp, dtype='uint8', name=f'bg_tmp{odf_scale}', tmp_dir=tmp_dir)

    # create ODF memory map
    nc = get_sph_harm_ncoeff(odf_degrees)
    odf_shp = odi_shp + (nc,)
    odf = create_memory_map(odf_shp, dtype='float32', name=f'odf_tmp{odf_scale}', tmp_dir=tmp_dir)

    # create orientation tensor memory map
    vec_tensor_shape = odi_shp + (3,)
    vec_tensor_eigen = create_memory_map(vec_tensor_shape, dtype='float32',
                                         name=f'tensor_tmp{odf_scale}'.format(odf_scale), tmp_dir=tmp_dir)

    # create ODI memory maps
    odi_tot = create_memory_map(odi_shp, dtype='float32', name=f'odi_tot_tmp{odf_scale}', tmp_dir=tmp_dir)
    if exp_all:
        odi_pri = create_memory_map(odi_shp, dtype='float32', name=f'odi_pri_tmp{odf_scale}', tmp_dir=tmp_dir)
        odi_sec = create_memory_map(odi_shp, dtype='float32', name=f'odi_sec_tmp{odf_scale}', tmp_dir=tmp_dir)
        odi_anis = create_memory_map(odi_shp, dtype='float32', name=f'odi_anis_tmp{odf_scale}', tmp_dir=tmp_dir)
        disarray = create_memory_map(odi_shp, dtype='float32', name=f'disarray{odf_scale}', tmp_dir=tmp_dir)
    else:
        odi_pri, odi_sec, odi_anis, disarray = (None, None, None, None)

    return odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, disarray, vec_tensor_eigen


def parallel_odf_at_scales(cli_args, save_dirs, fbr_vec, iso_fbr, px_sz, img_name, backend='loky', verbose=100):
    """
    Iterate over the required spatial scales and apply the parallel ODF analysis
    implemented in parallel_odf_on_slices().

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    save_dirs: dict
        saving directories
        ('frangi': Frangi filter, 'odf': ODF analysis, 'tmp': temporary files)

    fbr_vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    iso_fbr: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    px_sz: numpy.ndarray (axis order=(3,), dtype=float)
        pixel size [μm]

    img_name: str
        name of the input volume image

    backend: str
        backend module employed by joblib.Parallel

    verbose: int
        joblib verbosity level

    Returns
    -------
    None
    """

    # print ODF analysis heading
    print_odf_info(cli_args.odf_res, cli_args.odf_deg)

    # export all images (optional)
    exp_all = cli_args.exp_all

    # compute spherical harmonics normalization factors (once for all scales)
    odf_norm = get_sph_harm_norm_factors(cli_args.odf_deg)

    # get number of logical cores
    num_cpu = get_available_cores()

    # generate pixel size if not provided
    if px_sz is None:
        px_sz = np.array([cli_args.px_size_z, cli_args.px_size_xy, cli_args.px_size_xy])

    # parallel ODF analysis of fiber orientation vectors
    # over the required spatial scales
    n_scales = len(cli_args.odf_res)
    batch_sz = min(num_cpu, n_scales)
    with Parallel(n_jobs=batch_sz, backend=backend, verbose=verbose) as parallel:
        parallel(delayed(odf_analysis)(fbr_vec, iso_fbr, px_sz, save_dirs, img_name,
                                       odf_norm=odf_norm, odf_deg=cli_args.odf_deg, odf_scale_um=s, exp_all=exp_all)
                 for s in cli_args.odf_res)

    # print output directory
    print_flsh(f"\nODF and dispersion maps saved to: {save_dirs['odf']}\n")
