"""
Settings for the headless 3D cine pipeline. Edit the values below, then run:

    python run_inference.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# Input dataset
# ------------------------------------------------------------------
# Any of:
#   - a .zip of DICOMs (what the Streamlit app took as an upload)
#   - a folder of DICOMs (flat or nested)
#   - a .nii / .nii.gz / .npy array ordered (T, Z, H, W) = (frames, sagittal slices, rows, columns)
# Pre-set to the NIfTI that app.py was loading in place of the uploaded data.
DATASET_PATH = Path(r" \path\to\dataset ")

# .nii / .nii.gz / .npy only: axes to transpose the array into (T, Z, H, W), e.g. (3, 2, 1, 0).
# None = the array is already (T, Z, H, W), which is what app.py assumed.
ARRAY_AXES = None

# ------------------------------------------------------------------
# Models: these three are the only ones that run
# ------------------------------------------------------------------
MODELS_DIR = HERE / "models"

DEBANDING_ONNX_PATH = str(MODELS_DIR / "deband.onnx")
RESP_ONNX_PATH      = str(MODELS_DIR / "RespCorTorch-37.onnx")
E2E_ONNX_PATH       = str(MODELS_DIR / "E2E-51.onnx")

# "DirectML" (GPU; falls back to CPU if onnxruntime-directml isn't installed) or "CPU"
BACKEND = "DirectML"

# ------------------------------------------------------------------
# Outputs, written to OUTPUT_DIR / <dataset name> /
#   npy/    3D_cine_<t>.npy, one super-resolved volume per frame (what cine_mpr_viewer.py reads)
#   dicom/  <PATIENT_NUMBER>_VOL_<t>_SLICE_<z>.dcm
#   gifs/   one GIF per entry in GIFS
# Files from a previous run of the same dataset are replaced.
# ------------------------------------------------------------------
OUTPUT_DIR = HERE / "output"

SAVE_NPY = True

WRITE_DICOM = True
DICOM_TEMPLATE = HERE / "dicom_headerfile.dcm"
PATIENT_NUMBER = 0      # used in the file names, and as PatientID / PatientName when the source has none

# One GIF per (plane, slice index). Valid indices: "axial" 0-159, "coronal" 0-95, "sagittal" 0-111.
GIFS = [
    ("axial", 90),
    ("coronal", 40),
]
# Panels shown side by side, left to right:
#   ("raw", "debanded", "resp_cor", "3D_cine")  sequential improvement (the app's default)
#   ("3D_cine",)                                super-resolved cine only
#   ()                                          no GIFs
GIF_PANELS = ("raw", "debanded", "resp_cor", "3D_cine")
GIF_FPS = None          # None = number of cardiac frames, so one cycle per second (the app's default)

VERBOSE = False         # also print the DICOM loader's diagnostics (they always go to the log file)
