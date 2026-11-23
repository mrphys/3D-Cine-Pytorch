"""
The three ONNX models (contrast correction, respiratory correction, super-resolution) and the warp
that applies the respiratory-correction deformation fields.

Input/output layouts are the same as in the app:
  deband.onnx            frame (1, Z, H, W, 1)  -> (1, Z, H, W, 1)   [channels last]
  RespCorTorch-37.onnx   frame (1, 1, Z, H, W)  -> (1, 2, Z, H, W)   [x / y displacement]
  E2E-51.onnx            frame (1, 1, Z, H, W)  -> (1, 1, Z', H, W)  [super-resolved]
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.ndimage import map_coordinates

import config

log = logging.getLogger("cine3d")


# ------------------------------------------------------------------
# Sessions: each model is loaded once per process
# ------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_providers(backend="DirectML"):
    available = ort.get_available_providers()
    if backend == "DirectML":
        if "DmlExecutionProvider" in available:
            return ("DmlExecutionProvider", "CPUExecutionProvider")
        log.warning("DirectML is not available in this onnxruntime build "
                    "(pip install onnxruntime-directml); running on the CPU.")
    elif backend != "CPU":
        raise ValueError(f"BACKEND must be 'DirectML' or 'CPU', got {backend!r}")
    return ("CPUExecutionProvider",)


@lru_cache(maxsize=None)
def _get_session(model_path: str, backend: str = "DirectML"):
    so = ort.SessionOptions()
    providers = list(get_providers(backend))

    if providers[0] == "DmlExecutionProvider":
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    log.info(f"  loaded {Path(model_path).name} on {session.get_providers()[0]}")
    return session


# ------------------------------------------------------------------
# Shape helpers
# ------------------------------------------------------------------
def _prepare_input_frames(volume, frames=None):
    """
    Accepts (T, Z, H, W), (T, Z, H, W, C) or (T, 1, Z, H, W, C).
    Returns float32 (T, Z, H, W, C), so each frame x[i:i+1] is (1, Z, H, W, C).
    """
    x = np.asarray(volume, dtype=np.float32)

    if x.ndim == 4:
        x = x[..., np.newaxis]          # (T, Z, H, W) -> (T, Z, H, W, 1)
    elif x.ndim == 5:
        pass                            # already (T, Z, H, W, C)
    elif x.ndim == 6:
        if x.shape[1] != 1:
            raise ValueError(f"Unexpected 6D input shape {x.shape}")
        x = x[:, 0, ...]                # (T, 1, Z, H, W, C) -> (T, Z, H, W, C)
    else:
        raise ValueError(f"Unexpected input shape {x.shape}. Expected 4D, 5D, or 6D input.")

    if frames is None:
        frames = x.shape[0]
    if frames > x.shape[0]:
        raise ValueError(f"'frames' ({frames}) is larger than input time dimension ({x.shape[0]}).")

    return x[:frames]


def _normalize_output_shape(y):
    """
    (B, Z, H, W) -> (B, 1, Z, H, W, 1); (B, Z, H, W, C) -> (B, 1, Z, H, W, C); 6D unchanged.
    """
    y = np.asarray(y)
    if y.ndim == 4:
        y = y[:, None, ..., None]
    elif y.ndim == 5:
        y = y[:, None, ...]
    elif y.ndim != 6:
        raise ValueError(f"Unexpected output shape {y.shape}.")
    return y


# ------------------------------------------------------------------
# The three models
# ------------------------------------------------------------------
def apply_debanding_model(volume, frames=None, backend="DirectML", model_path=None):
    """Contrast correction. (T, Z, H, W) -> (T, 1, Z, H, W, 1)."""
    sess = _get_session(model_path or config.DEBANDING_ONNX_PATH, backend)
    x = _prepare_input_frames(volume, frames=frames)
    input_name = sess.get_inputs()[0].name

    outputs = []
    for i in range(x.shape[0]):
        frame = x[i:i + 1]                                   # (1, Z, H, W, 1)
        y = sess.run(None, {input_name: frame})[0]
        outputs.append(_normalize_output_shape(y))
    return np.concatenate(outputs, axis=0)


def apply_resp_model(volume, frames=None, backend="DirectML", model_path=None):
    """Respiratory correction fields. Returns a list of T arrays (2, Z, H, W): channel 0 = x, 1 = y shift."""
    sess = _get_session(model_path or config.RESP_ONNX_PATH, backend)
    x = _prepare_input_frames(volume, frames=frames)
    input_name = sess.get_inputs()[0].name

    def_fields = []
    for i in range(x.shape[0]):
        frame = np.expand_dims(x[i:i + 1][..., 0], 0)        # (1, 1, Z, H, W)
        outs = sess.run(None, {input_name: frame})
        def_fields.append(outs[0][0])
    return def_fields


def apply_E2E_model(volume, frames=None, backend="DirectML", model_path=None):
    """Super-resolution. (T, Z, H, W) -> (T, Z', H, W)."""
    sess = _get_session(model_path or config.E2E_ONNX_PATH, backend)
    x = _prepare_input_frames(volume, frames=frames)
    input_name = sess.get_inputs()[0].name

    outputs = []
    for i in range(x.shape[0]):
        frame = np.expand_dims(x[i:i + 1][..., 0], 0)        # (1, 1, Z, H, W)
        y = sess.run(None, {input_name: frame})[0]
        outputs.append(y[0, ...])
    out = np.concatenate(outputs, axis=0)
    log.debug(f"  super-resolution output shape: {out.shape}")
    return out


# ------------------------------------------------------------------
# Warp (applies the respiratory-correction fields)
# ------------------------------------------------------------------
def warp_slice_2d(image_2d, field_2d, order=1, mode="nearest", cval=0.0,
                  field_components=("x", "y"), pull=True):
    """
    Warp one 2D slice (H, W) with a field (2, H, W).
    field_components: ('x', 'y') or ('y', 'x'), the channel order of the field.
    pull=True samples at base + displacement, False at base - displacement.
    """
    H, W = image_2d.shape

    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )

    if field_components == ("x", "y"):
        dx = field_2d[0].astype(np.float32)
        dy = field_2d[1].astype(np.float32)
    elif field_components == ("y", "x"):
        dy = field_2d[0].astype(np.float32)
        dx = field_2d[1].astype(np.float32)
    else:
        raise ValueError("field_components must be ('x', 'y') or ('y', 'x')")

    sign = 1.0 if pull else -1.0

    return map_coordinates(
        image_2d,
        [yy + sign * dy, xx + sign * dx],
        order=order,
        mode=mode,
        cval=cval,
    )


def warp_stack_4d(images, fields, order=1, mode="nearest", cval=0.0,
                  field_components=("x", "y"), pull=True, dtype=None):
    """
    Warp images (T, Z, H, W) with fields (T, 2, Z, H, W), or a list of T arrays (2, Z, H, W).
    Returns (T, Z, H, W) in images.dtype (or dtype).
    """
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"images must have shape (T, Z, H, W), got {images.shape}")

    T, Z, H, W = images.shape

    if isinstance(fields, list):
        if len(fields) != T:
            raise ValueError(f"len(fields) must match T={T}, got {len(fields)}")
        fields_arr = np.stack(fields, axis=0)
    else:
        fields_arr = np.asarray(fields)

    if fields_arr.shape != (T, 2, Z, H, W):
        raise ValueError(f"fields must have shape (T, 2, Z, H, W) = {(T, 2, Z, H, W)}, got {fields_arr.shape}")

    warped = np.empty((T, Z, H, W), dtype=images.dtype if dtype is None else dtype)
    for t in range(T):
        for z in range(Z):
            warped[t, z] = warp_slice_2d(
                image_2d=images[t, z],
                field_2d=fields_arr[t, :, z],
                order=order,
                mode=mode,
                cval=cval,
                field_components=field_components,
                pull=pull,
            )
    return warped
