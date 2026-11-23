"""
GIF previews and DICOM export of the super-resolved cine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import skimage.transform
from PIL import Image
from pydicom.uid import generate_uid

from preprocessing import CROP_H, CROP_W, N_SLICES

# Volumes that can be shown in a GIF, in the app's left-to-right order
GIF_STAGES = ("raw", "debanded", "resp_cor", "3D_cine")

# The 28-slice stages are brought onto the super-resolved 112-slice grid for display,
# with the spline orders the app used
SR_Z_FACTOR = 4
GIF_UPSAMPLE_ORDER = {"raw": 2, "debanded": 2, "resp_cor": 3}

GIF_SCALE = 3
_NEAREST = getattr(Image, "Resampling", Image).NEAREST


# ------------------------------------------------------------------
# GIFs
# ------------------------------------------------------------------
def check_gif_settings(gifs, panels):
    """Fail before inference, not after it, if a GIF setting is wrong."""
    for name in panels:
        if name not in GIF_STAGES:
            raise ValueError(f"Unknown GIF panel {name!r}; choose from {GIF_STAGES}")
    limits = {"axial": CROP_H, "coronal": CROP_W, "sagittal": N_SLICES * SR_Z_FACTOR}
    for plane, index in gifs:
        if plane not in limits:
            raise ValueError(f"Unknown GIF plane {plane!r}; choose from {tuple(limits)}")
        if not 0 <= index < limits[plane]:
            raise ValueError(f"{plane} slice {index} is out of range 0-{limits[plane] - 1}")


def plane_frames(vol, plane, index):
    """
    One slice of a (T, Z, H, W) cine as (T, rows, cols), oriented as in the app's GIFs.
      axial    vol[:, :, index, :]  rows = W, columns = Z (reversed)
      coronal  vol[:, :, :, index]  rows = H, columns = Z
      sagittal vol[:, index, :, :]  rows = H, columns = W
    """
    if plane == "axial":
        return vol[:, ::-1, index, :].transpose(0, 2, 1)
    if plane == "coronal":
        return vol[:, :, :, index].transpose(0, 2, 1)
    if plane == "sagittal":
        return vol[:, index, :, :]
    raise ValueError(f"Unknown plane {plane!r}")


def to_uint8(frames):
    """0-255 using the min/max of this slice over all frames."""
    mn, mx = frames.min(), frames.max()
    if not mx > mn:
        return np.zeros(frames.shape, dtype=np.uint8)
    img = np.clip(frames.astype(np.float32), mn, mx)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def write_gif(frames_u8, path, fps):
    """(T, rows, cols) uint8 -> looping GIF, enlarged GIF_SCALE x with nearest neighbour."""
    h, w = frames_u8.shape[1:]
    images = [Image.fromarray(f).resize((w * GIF_SCALE, h * GIF_SCALE), _NEAREST) for f in frames_u8]
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000 / fps, loop=0)


def make_gifs(stages, gifs, panels, out_dir, fps, name_prefix=""):
    """
    stages  dict of (T, Z, H, W) arrays: "raw", "debanded", "resp_cor" (28 slices), "3D_cine" (super-resolved)
    gifs    list of (plane, slice index)
    panels  stage names shown side by side, left to right
    Returns the paths written.
    """
    if not gifs or not panels:
        return []
    out_dir = Path(out_dir)
    gifs = [(plane, int(index)) for plane, index in gifs]

    slices = {g: [] for g in gifs}
    for name in panels:                     # one upsampled volume in memory at a time
        vol = stages[name]
        order = GIF_UPSAMPLE_ORDER.get(name)
        if order is not None:
            vol = skimage.transform.rescale(vol, (1, SR_Z_FACTOR, 1, 1), order=order)
        limits = {"axial": vol.shape[2], "coronal": vol.shape[3], "sagittal": vol.shape[1]}
        for plane, index in gifs:
            if not 0 <= index < limits[plane]:
                raise ValueError(f"{plane} slice {index} is out of range for {name} {vol.shape[1:]}")
            slices[(plane, index)].append(to_uint8(plane_frames(vol, plane, index)))
        del vol

    tag = "SR" if tuple(panels) == ("3D_cine",) else "comparison"
    paths = []
    for (plane, index), imgs in slices.items():
        path = out_dir / f"{name_prefix}{tag}_{plane}_{index}.gif"
        write_gif(np.concatenate(imgs, axis=2), path, fps)
        paths.append(path)
    return paths


# ------------------------------------------------------------------
# DICOM
# ------------------------------------------------------------------
def write_dicom_series(cine, out_dir, template_path, patient_number=0, patient_info=None):
    """
    Writes the super-resolved cine (T, Z, H, W), values ~0-1, as one MR series: a file per frame and slice,
    <patient_number>_VOL_<t>_SLICE_<z>.dcm. Header fields come from the template, with patient/study
    details from patient_info where present. Returns the number of files written.
    """
    template_path = Path(template_path)
    if not template_path.is_file():
        raise FileNotFoundError(f"DICOM header file not found: {template_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cine = np.asarray(cine)
    T, Z, H, W = cine.shape
    info = patient_info or {}

    # The same dataset is updated and written for every slice; everything that differs between
    # slices is set inside the loop.
    ds = pydicom.dcmread(str(template_path))

    # ---- series-level fields ----
    ds.PatientID = info["PatientID"] if info.get("PatientID") is not None else str(patient_number)
    ds.PatientName = info["PatientName"] if info.get("PatientName") is not None else f"PATIENT_{patient_number}"
    for key in ("PatientBirthDate", "PatientSex", "StudyDate", "StudyTime"):
        if info.get(key) is not None:
            setattr(ds, key, info[key])

    ds.StudyDescription = "3D Cine"
    ds.SeriesDescription = "HR"
    # Keep the original study UID if available, otherwise generate one
    ds.StudyInstanceUID = info["StudyInstanceUID"] if info.get("StudyInstanceUID") is not None else generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 1

    ds.SliceThickness = 1.5
    ds.Rows = H
    ds.Columns = W
    ds.AcquisitionMatrix = [0, H, W, 0]
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [1.5, 1.5]
    ds.SmallestImagePixelValue = 0
    ds.LargestImagePixelValue = 255
    ds.AcquisitionTime = str(float(100000))
    ds.SeriesTime = str(float(100000))

    # ---- per-file fields ----
    instance = 1
    for i in range(T):
        for j in range(Z):
            pixel_data = (cine[i, j] * 255).astype(np.uint16)

            ds.InstanceNumber = instance
            instance += 1
            ds.AcquisitionNumber = i
            ds.SliceLocation = -100.0 + ((j - 1) * 1.5)
            ds.ImagePositionPatient = [-100.0 + ((j - 1) * 1.5), -102.81169796382, 250.98882130193]
            ds.ContentTime = str(float(100000 + j * 0.3))
            ds.TriggerTime = str(i * 34)
            ds.PixelData = pixel_data.tobytes()

            sop_uid = generate_uid()
            ds.SOPInstanceUID = sop_uid
            ds.file_meta.MediaStorageSOPInstanceUID = sop_uid

            pydicom.dcmwrite(str(out_dir / f"{patient_number}_VOL_{i}_SLICE_{j}.dcm"), ds)

    return T * Z
