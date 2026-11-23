"""
Bringing the raw sagittal stack to the size the models expect, plus the intensity helpers the
pipeline uses between models. These are the app's functions, unchanged in behaviour.
"""
import numpy as np
import skimage.transform

# Model input size: 28 sagittal slices of 160 x 96 pixels
N_SLICES = 28
CROP_H, CROP_W = 160, 96


def crop_and_normalise(sag_vols):
    """
    The app's "Cropping..." step. (T, Z, H, W) raw cine -> (T, 28, 160, 96), scaled by norm().
    """
    sag_vols = np.asarray(sag_vols)
    sag_vols = sag_vols[:, ::-1, :, :]
    sag_vols = fix_num_slices(sag_vols)

    if sag_vols.shape[2] == 512:
        sag_vols = skimage.transform.rescale(sag_vols, (1, 1, 0.5, 0.5), order=3, anti_aliasing=True)

    sag_vols = sag_vols[:, ::-1, :, :]

    cropped = np.stack([
        np.stack([resize(sag_vols[t, z], CROP_H, CROP_W)[..., 0] for z in range(sag_vols.shape[1])])
        for t in range(sag_vols.shape[0])
    ])
    return norm(cropped)


def fix_num_slices(sag_vols, target=N_SLICES):
    """Centre-crop or zero-pad the slice axis (axis 1) to `target` slices."""
    n = sag_vols.shape[1]

    # Center crop if too many slices
    if n > target:
        start = (n - target) // 2
        return sag_vols[:, start:start + target, :, :]

    # Symmetric pad if too few slices
    if n < target:
        total_pad = target - n
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        return np.pad(
            sag_vols,
            ((0, 0), (pad_before, pad_after), (0, 0), (0, 0)),
            mode="constant",   # zero padding
        )

    return sag_vols


def resize(t1, x, y):
    """
    Centre crop / zero pad a 2D (H, W) or 3D (H, W, C) image to (x, y, C).
    NumPy replacement for tf.image.resize_with_crop_or_pad. For 2D input, C = 1.
    """
    t1 = np.asarray(t1)

    # Add channel axis if input is 2D
    if t1.ndim == 2:
        t1 = t1[..., np.newaxis]
    elif t1.ndim != 3:
        raise ValueError(f"Expected 2D or 3D input, got shape {t1.shape}")

    h, w, c = t1.shape

    # ----- Center crop if too large -----
    if h > x:
        start_h = (h - x) // 2
        t1 = t1[start_h:start_h + x, :, :]
    if w > y:
        start_w = (w - y) // 2
        t1 = t1[:, start_w:start_w + y, :]

    h, w, c = t1.shape

    # ----- Zero pad if too small -----
    pad_top = max((x - h) // 2, 0)
    pad_bottom = max(x - h - pad_top, 0)
    pad_left = max((y - w) // 2, 0)
    pad_right = max(y - w - pad_left, 0)

    if pad_top or pad_bottom or pad_left or pad_right:
        t1 = np.pad(
            t1,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    return t1


def norm(t1):
    """(x - min) / max, as in the app (note: divides by max, not max - min)."""
    im1 = np.asarray(t1)
    return (im1 - np.min(im1)) / np.max(im1)


def z_stand(img):
    """Z-score over the whole array."""
    return (img - np.mean(img)) / np.std(img)
