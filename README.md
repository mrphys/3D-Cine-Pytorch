# 3D Cine Pytorch

Command-line version of the 3D cine Streamlit app. It takes a sagittal real-time cine and runs three ONNX models:

1. **Contrast correction**: `deband.onnx`
2. **Respiratory correction**: `RespCorTorch-37.onnx`, whose displacement fields are then applied with a warp
3. **Super-resolution**: `E2E-51.onnx`, which gives 4× the input slices

The result is saved as per-frame `.npy` volumes, a DICOM series and GIF previews.

> Not intended to diagnose, treat, cure, or prevent any disease.

## Setup

To install the streamlit app version, simply navigate to releases, download the .exe and install as normal. For code based implementation, see below.

Python 3.10. On Windows (PowerShell), from the repository folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows this installs `onnxruntime-directml` (GPU through DirectML); on other systems, the CPU build of ONNX Runtime.

## Running

Set `DATASET_PATH` in `config.py`, then:

```powershell
python run_inference.py
```

`DATASET_PATH` can be a `.zip` of DICOMs, a folder of DICOMs, or a `.nii` / `.nii.gz` / `.npy` array ordered (frames, slices, rows, columns). Each slice is centre-cropped or zero-padded to 160 × 96 pixels, padded or cropped to 28 slices; 512 × 512 images are halved first.

Results go to `output/<dataset name>/`:

| Folder | Contents |
|---|---|
| `npy/` | `3D_cine_<frame>.npy`, one super-resolved volume per frame |
| `dicom/` | `<PATIENT_NUMBER>_VOL_<frame>_SLICE_<slice>.dcm`, one series |
| `gifs/` | one GIF per entry in `GIFS` |

A run replaces that dataset's results from any earlier run.

## Settings (`config.py`)

| Setting | Default | Notes |
|---|---|---|
| `BACKEND` | `"DirectML"` | Or `"CPU"`. DirectML falls back to the CPU if it isn't available. |
| `GIFS` | axial 60, coronal 60 | One GIF per (plane, slice). Axial 0–159, coronal 0–95, sagittal 0 to 27. |
| `GIF_PANELS` | all four stages | Shown side by side. `("3D_cine",)` gives the super-resolved cine only; `()` turns GIFs off. |
| `GIF_FPS` | `None` | `None` plays one cardiac cycle per second. |
| `SAVE_NPY`, `WRITE_DICOM` | `True` | |
| `ARRAY_AXES` | `None` | Axis order for `.nii` / `.npy` input that isn't (frames, slices, rows, columns), e.g. `(3, 2, 1, 0)`. |

## DICOM output

Headers are built from `dicom_headerfile.dcm`, a header-only MR template with no image or patient data. Patient and study details (name, ID, birth date, sex, study UID, date and time) are copied from the input DICOMs when there are any; `.nii` and `.npy` input gives `PATIENT_<PATIENT_NUMBER>`.
