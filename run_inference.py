"""
Headless 3D cine pipeline (no Streamlit):

    raw sagittal cine -> 28 slices x 160 x 96
      -> contrast correction       deband.onnx
      -> respiratory correction    RespCorTorch-37.onnx + warp
      -> super-resolution          E2E-51.onnx
      -> npy volumes, GIF previews, DICOM series

Set the dataset path and options in config.py, then run:

    python run_inference.py
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import config
from data_loading import load_dataset
from models import apply_E2E_model, apply_debanding_model, apply_resp_model, get_providers, warp_stack_4d
from outputs import check_gif_settings, make_gifs, write_dicom_series
from preprocessing import crop_and_normalise, norm, z_stand

log = logging.getLogger("cine3d")


def run_models(volume, backend="DirectML"):
    """
    The app's processing chain for a raw cine (T, Z, H, W). Returns every stage as (T, Z, H, W):
    "raw", "debanded", "resp_cor" (28 slices) and "3D_cine" (super-resolved, values ~0-1).
    """
    with _step("Cropping"):
        cropped = crop_and_normalise(volume)
        log.info(f"  cardiac frames: {cropped.shape[0]}, volume shape: {cropped.shape[1:]}")

    with _step("Contrast correction"):
        debanded = norm(apply_debanding_model(z_stand(cropped), backend=backend))

    with _step("Respiratory correction"):
        fields = apply_resp_model(z_stand(debanded), backend=backend)
        resp_cor = warp_stack_4d(z_stand(debanded[:, 0, ..., 0]), fields)

    with _step("Super-resolution"):
        sr = norm(apply_E2E_model(z_stand(resp_cor), backend=backend))
        log.info(f"  super-resolved volume shape: {sr.shape[1:]}")

    return {"raw": cropped, "debanded": debanded[:, 0, ..., 0], "resp_cor": resp_cor, "3D_cine": sr}


def main():
    _setup_logging()
    t_start = time.perf_counter()

    dataset = Path(config.DATASET_PATH)
    _check_settings(dataset)
    get_providers(config.BACKEND)   # warns here if DirectML isn't available

    run_dir = Path(config.OUTPUT_DIR) / _dataset_name(dataset)
    npy_dir, dicom_dir, gif_dir = run_dir / "npy", run_dir / "dicom", run_dir / "gifs"
    _prepare_folder(npy_dir, "3D_cine_*.npy", config.SAVE_NPY)
    _prepare_folder(dicom_dir, "*.dcm", config.WRITE_DICOM)
    _prepare_folder(gif_dir, "*.gif", bool(config.GIFS and config.GIF_PANELS))

    log.info(f"Dataset: {dataset}")
    with _step("Loading"):
        volume, patient_info = load_dataset(
            dataset,
            array_axes=config.ARRAY_AXES,
            log_path=run_dir / "cine_loader_debug.txt",
            verbose=config.VERBOSE,
        )
        log.info(f"  raw cine (T, Z, H, W): {volume.shape}")

    stages = run_models(volume, backend=config.BACKEND)
    sr = stages["3D_cine"]
    n_frames = sr.shape[0]

    if config.SAVE_NPY:
        for i in range(n_frames):
            np.save(npy_dir / f"3D_cine_{i}.npy", sr[i])
        log.info(f"Saved {n_frames} volumes to {npy_dir}")

    if config.WRITE_DICOM:
        with _step("DICOM export"):
            n_files = write_dicom_series(
                sr, dicom_dir, config.DICOM_TEMPLATE,
                patient_number=config.PATIENT_NUMBER,
                patient_info=patient_info,
            )
            log.info(f"  {n_files} files in {dicom_dir}")

    if config.GIFS and config.GIF_PANELS:
        with _step("GIFs"):
            fps = config.GIF_FPS or n_frames
            for path in make_gifs(stages, config.GIFS, config.GIF_PANELS, gif_dir, fps):
                log.info(f"  {path}")

    log.info(f"Done in {time.perf_counter() - t_start:.1f} s. Outputs: {run_dir}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _setup_logging():
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


@contextmanager
def _step(name):
    log.info(f"{name}...")
    t0 = time.perf_counter()
    yield
    log.info(f"  ({time.perf_counter() - t0:.1f} s)")


def _check_settings(dataset):
    """Catch path and setting mistakes before anything slow runs."""
    if not dataset.exists():
        raise FileNotFoundError(f"DATASET_PATH not found: {dataset}")
    for p in (config.DEBANDING_ONNX_PATH, config.RESP_ONNX_PATH, config.E2E_ONNX_PATH):
        if not Path(p).is_file():
            raise FileNotFoundError(f"Model file not found: {p} (check MODELS_DIR in config.py)")
    if config.WRITE_DICOM and not Path(config.DICOM_TEMPLATE).is_file():
        raise FileNotFoundError(f"DICOM header file not found: {config.DICOM_TEMPLATE}")
    if config.BACKEND not in ("DirectML", "CPU"):
        raise ValueError(f"BACKEND must be 'DirectML' or 'CPU', got {config.BACKEND!r}")
    check_gif_settings(config.GIFS, config.GIF_PANELS)


def _dataset_name(path):
    name = path.name
    for ext in (".nii.gz", ".nii", ".npy", ".zip"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _prepare_folder(folder, pattern, needed):
    """Remove this pipeline's files from an earlier run, so old and new results never mix."""
    if folder.exists():
        for old in folder.glob(pattern):
            old.unlink()
    if needed:
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
