"""
Reading the input dataset: a ZIP or folder of DICOMs (any vendor layout), or a .nii/.nii.gz/.npy array.

load_dataset() returns the raw cine as (T, Z, H, W) plus the patient/study details used for the DICOM export.
"""
from __future__ import annotations

import glob
import logging
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pydicom
from pydicom.misc import is_dicom
from pydicom.tag import Tag

log = logging.getLogger("cine3d")

PRIVATE_PHASE_TAG = Tag(0x0019, 0x10D7)
PHILIPS_PHASE_TAG = Tag(0x2001, 0x1008)  # [Phase Number]
PHILIPS_SLICE_TAG = Tag(0x2001, 0x100A)  # [Slice Number MR]

PATIENT_FIELDS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
    "StudyInstanceUID", "StudyDate", "StudyTime",
)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def load_dataset(path, array_axes=None, log_path=None, verbose=False):
    """
    Returns (volume, patient_info):
      volume        np.ndarray (T, Z, H, W)
      patient_info  dict with PATIENT_FIELDS (empty for .nii / .npy input)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    name = path.name.lower()
    if path.is_dir():
        return _load_dicom_dir(path, log_path, verbose, path), get_source_patient_info(path)

    if name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_zip(path, temp_dir)
            patient_info = get_source_patient_info(temp_dir)
            volume = _load_dicom_dir(temp_dir, log_path, verbose, path)
        return volume, patient_info

    if name.endswith((".nii", ".nii.gz", ".npy")):
        return _load_array(path, array_axes), {}

    raise ValueError(f"Unsupported dataset: {path} (expected a folder, .zip, .nii, .nii.gz or .npy)")


def _load_dicom_dir(folder, log_path, verbose, source):
    volume, _ = load_cine_any(str(folder), number_of_scans=None, log_path=log_path, verbose=verbose)
    if volume is None or len(volume) == 0:
        where = f" - see {log_path}" if log_path else ""
        raise RuntimeError(f"Could not build a cine from the DICOMs in {source}{where}")
    return volume


def _load_array(path, array_axes=None):
    if path.name.lower().endswith(".npy"):
        volume = np.load(path)
    else:
        import nibabel as nib  # only needed for NIfTI input
        volume = nib.load(str(path)).get_fdata()

    if array_axes is not None:
        volume = np.transpose(volume, array_axes)
    if volume.ndim != 4:
        raise ValueError(f"{path.name} has shape {volume.shape}; expected 4D (T, Z, H, W) "
                         f"(set ARRAY_AXES in config.py if the axes are in another order)")
    return volume


# ------------------------------------------------------------------
# ZIP / patient info
# ------------------------------------------------------------------
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Filter out __MACOSX and dotfiles
        valid_files = [
            f for f in zip_ref.namelist()
            if "__MACOSX" not in f and not os.path.basename(f).startswith("._")
        ]
        zip_ref.extractall(extract_to, members=valid_files)


def get_source_patient_info(source_dir):
    """Patient/study details from the first DICOM in source_dir that has any of them."""
    for p in sorted(Path(source_dir).rglob("*")):
        if not p.is_file() or "__MACOSX" in p.parts:
            continue
        if p.name.startswith("._") or p.suffix.lower() == ".zip":
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            info = {k: getattr(ds, k, None) for k in PATIENT_FIELDS}
        except Exception:
            continue
        if any(v is not None for v in info.values()):
            return info
    return {}


# ------------------------------------------------------------------
# Universal DICOM cine loader (flat or nested folders)
# ------------------------------------------------------------------
def load_cine_any(
    base_dir: str,
    number_of_scans: int = None,                  # if None, use all detected phases
    private_phase_tag: Tag = PRIVATE_PHASE_TAG,
    log_path=None,                                # diagnostics file (None = no file)
    verbose: bool = False,
):
    """
    Returns:
        volume: np.ndarray with shape (T, Z, H, W)
        num_phases_detected: int
    """
    # ---- helpers -------------------------------------------------------------
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    def dlog(msg):
        if log_path is not None:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(str(msg) + "\n")
            except OSError:
                pass
        if verbose:
            log.info(msg)
        else:
            log.debug(msg)

    def safe_int(x):
        try:
            # pydicom DataElement -> value
            if hasattr(x, "value"):
                x = x.value
            if isinstance(x, (list, tuple)):
                return int(str(x[0]))
            return int(str(x))
        except Exception:
            return None

    def safe_float(x):
        try:
            if hasattr(x, "value"):
                x = x.value
            if isinstance(x, (list, tuple)):
                x = x[0]
            return float(str(x))
        except Exception:
            return None

    def tm_to_seconds(tm):
        """DICOM TM (HHMMSS.frac) -> seconds (float)"""
        try:
            if tm is None:
                return None
            s = str(tm).strip()
            if len(s) < 6:
                return None
            hh = float(s[0:2]); mm = float(s[2:4]); ss = float(s[4:])
            return hh * 3600 + mm * 60 + ss
        except Exception:
            return None

    def get_ref_frame(ds):
        """Try to extract Referenced Frame Number (0008,1160) from common nested sequences."""
        RFN = Tag(0x0008, 0x1160)
        CSA = Tag(0x0020, 0x9172)
        REL = Tag(0x0008, 0x1250)
        RIS = Tag(0x0008, 0x1140)

        for seq_tag in (CSA, REL):
            # A) Conversion Source Attributes Sequence (0020,9172)
            # B) Related Series Sequence (0008,1250) -> Referenced Image Sequence (0008,1140)
            try:
                seq = ds.get(seq_tag, None)
                if seq:
                    for item in seq:
                        rfn = item.get(RFN, None)
                        if rfn is not None:
                            return safe_int(rfn)
                        ris = item.get(RIS, None)
                        if ris:
                            for sub in ris:
                                rfn2 = sub.get(RFN, None)
                                if rfn2 is not None:
                                    return safe_int(rfn2)
            except Exception:
                pass
        return None

    def get_slice_axis_distance(ds):
        """
        Position of the slice along the slice-normal axis: s = dot(normal, IPP),
        normal = r x c from ImageOrientationPatient. Falls back to SliceLocation if IOP/IPP missing.
        """
        try:
            iop = getattr(ds, "ImageOrientationPatient", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) >= 6 and len(ipp) >= 3:
                r = np.array(iop[0:3], dtype=float)
                c = np.array(iop[3:6], dtype=float)
                n = np.cross(r, c)  # slice normal
                p = np.array(ipp[0:3], dtype=float)
                return float(np.dot(n, p))
        except Exception:
            pass
        try:
            sl = getattr(ds, "SliceLocation", None)
            return float(sl) if sl is not None else None
        except Exception:
            return None

    def time_sort_key(rec):
        """Best-effort temporal ordering key within a single slice. Prefer explicit phase/frame tags."""
        for name, pri in [
            ("phase_philips", 0),
            ("ref_frame",     1),
            ("phase_private", 2),
            ("tpi",           3),
        ]:
            v = rec.get(name)
            if v is not None:
                return (pri, int(v))

        # then TriggerTime, AcquisitionTime, InstanceNumber
        if rec.get("trigger_time") is not None:
            return (10, rec["trigger_time"])
        if rec.get("acq_seconds") is not None:
            return (11, rec["acq_seconds"])
        if rec.get("inst") is not None:
            return (12, rec["inst"])
        return (99, rec.get("path"))

    # ---- scan files ----------------------------------------------------------
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"No such directory: {base_dir}")

    all_paths = sorted(glob.glob(os.path.join(base_dir, "**", "*"), recursive=True))
    dlog(f"Found files (pre-filter): {len(all_paths)}")

    def header_is_dicom(p):
        try:
            _ = pydicom.dcmread(p, force=True, stop_before_pixels=True)
            return True
        except Exception:
            return False

    files = []
    for p in all_paths:
        if not os.path.isfile(p):
            continue
        if "__MACOSX" in p:
            continue
        bn = os.path.basename(p)
        if bn.startswith("._"):
            continue
        if p.lower().endswith(".zip"):
            continue

        ok = False
        try:
            ok = is_dicom(p)
        except Exception:
            ok = False
        if not ok:
            ok = header_is_dicom(p)

        if ok:
            files.append(p)

    dlog(f"Candidate DICOM files (after checks): {len(files)}")

    if not files:
        dlog("No valid DICOM files found.")
        return np.array([]), 0

    dicom_dirs = {os.path.dirname(p) for p in files}
    dlog(f"Flat folder layout: {len(dicom_dirs) == 1}")

    # ---- parse headers (no pixels) ------------------------------------------
    recs, bad = [], 0
    for p in files:
        try:
            ds = pydicom.dcmread(p, force=True, stop_before_pixels=True)

            # Philips explicit phase / slice tags
            phase_philips = None
            slice_no = None
            try:
                if PHILIPS_PHASE_TAG in ds:
                    phase_philips = safe_int(ds[PHILIPS_PHASE_TAG])
            except Exception:
                pass
            try:
                if PHILIPS_SLICE_TAG in ds:
                    slice_no = safe_int(ds[PHILIPS_SLICE_TAG])
            except Exception:
                pass

            # Private phase tag
            phase_priv = None
            try:
                if private_phase_tag in ds:
                    phase_priv = safe_int(ds[private_phase_tag])
            except Exception:
                pass

            # Standard-ish
            tpi  = safe_int(getattr(ds, "TemporalPositionIdentifier", None))
            inst = safe_int(getattr(ds, "InstanceNumber", None))
            rfn  = get_ref_frame(ds)

            # geometric slice position
            spos = get_slice_axis_distance(ds)

            # timing for ordering fallback
            trigger_time = safe_float(getattr(ds, "TriggerTime", None))
            acq_seconds  = tm_to_seconds(getattr(ds, "AcquisitionTime", None))

            # declared T tags (used only as weak hint)
            ntp = safe_int(getattr(ds, "NumberOfTemporalPositions", None))  # (0020,0105)
            cni = safe_int(getattr(ds, "CardiacNumberOfImages", None))      # (0018,1090)
            declared_T = ntp if ntp not in (None, 0) else cni

            recs.append({
                "path": p,
                "phase_philips": phase_philips,   # (2001,1008)
                "slice_no": slice_no,             # (2001,100A)
                "phase_private": phase_priv,      # (0019,10D7)
                "tpi": tpi,
                "ref_frame": rfn,                 # (0008,1160) nested
                "inst": inst,
                "spos": spos,
                "trigger_time": trigger_time,
                "acq_seconds": acq_seconds,
                "declared_T": declared_T,
            })
        except Exception:
            bad += 1

    dlog(f"Headers parsed: {len(recs)}")
    dlog(f"Headers failed: {bad}")
    for i in range(min(3, len(recs))):
        r = recs[i]
        dlog(f"Sample rec {i + 1}: " + str({k: r[k] for k in (
            "phase_philips", "slice_no", "phase_private", "tpi", "ref_frame", "inst", "spos", "declared_T")}))

    if not recs:
        dlog("No headers could be parsed.")
        return np.array([]), 0

    # ---- choose phase key (prefer explicit phase tags; DO NOT use InstanceNumber as phase) ----
    uniq = lambda k: len({r[k] for r in recs if r.get(k) is not None})
    counts = {k: uniq(k) for k in ("phase_philips", "ref_frame", "phase_private", "tpi", "inst")}
    dlog(f"Phase key candidate counts: {counts}")

    declared_Ts = [r["declared_T"] for r in recs if r.get("declared_T")]
    declared_T = Counter(declared_Ts).most_common(1)[0][0] if declared_Ts else None
    dlog(f"Declared T (from tags): {declared_T}")

    phase_key = None
    for k in ("phase_philips", "ref_frame", "phase_private", "tpi"):
        if counts[k] > 1:
            phase_key = k
            break

    dlog(f"Phase key selected: {phase_key or 'None -> slice/time reconstruction'}")

    # ---- builder using explicit phase key -----------------------------------
    def build_from_phase_key(key_name):
        groups = defaultdict(list)  # phase -> list of (slice_sort, path)
        for r in recs:
            ph = r.get(key_name)
            if ph is None:
                continue

            # slice ordering: prefer geometric spos, else Philips slice_no
            srt = r.get("spos")
            if srt is None:
                srt = r.get("slice_no")
            groups[int(ph)].append((float("inf") if srt is None else float(srt), r["path"]))

        if not groups:
            return None, 0

        phase_ids = sorted(groups.keys())
        if number_of_scans is not None:
            phase_ids = phase_ids[:number_of_scans]

        stacks = []
        slice_counts = []
        for ph in phase_ids:
            entries = groups[ph]
            entries.sort(key=lambda x: x[0])
            imgs = []
            for _, p in entries:
                try:
                    imgs.append(pydicom.dcmread(p, force=True).pixel_array)
                except Exception:
                    pass
            if imgs:
                stacks.append(imgs)
                slice_counts.append(len(imgs))

        if not stacks:
            return None, 0

        # harmonize Z
        if len(set(slice_counts)) > 1:
            common_Z = Counter(slice_counts).most_common(1)[0][0]
            stacks = [s[:common_Z] for s in stacks if len(s) >= common_Z]
            if not stacks:
                return None, 0

        vol = np.array(stacks)  # (T,Z,H,W)
        return vol, vol.shape[0]

    volume = None
    T_detected = 0

    if phase_key is not None:
        volume, T_detected = build_from_phase_key(phase_key)

    # ---- fallback: slice/time reconstruction (NO assumptions about T) --------
    if volume is None:
        dlog("Phase grouping failed or was skipped; reconstructing by slice groups + temporal ordering.")

        # 1) choose slice grouping method
        have_slice_no = any(r.get("slice_no") is not None for r in recs)
        have_spos = any(r.get("spos") is not None for r in recs)

        slice_groups = []  # list of tuples: (slice_sort_value, [recs...])

        if have_slice_no:
            by_slice = defaultdict(list)
            for r in recs:
                sn = r.get("slice_no")
                if sn is None:
                    continue
                by_slice[int(sn)].append(r)
            for sn, lst in by_slice.items():
                # Use mean physical position if available; otherwise fall back to sn
                spos_vals = [rr.get("spos") for rr in lst if rr.get("spos") is not None]
                rep = float(np.mean(spos_vals)) if spos_vals else float(sn)
                slice_groups.append((rep, lst))
            slice_groups.sort(key=lambda x: x[0])
            dlog(f"Slice grouping: Philips slice_no (2001,100A). Groups={len(slice_groups)}")

        elif have_spos:
            # cluster slice positions with tolerance
            try:
                any_ds = pydicom.dcmread(recs[0]["path"], force=True, stop_before_pixels=True)
                sbs = getattr(any_ds, "SpacingBetweenSlices", None)
                tol = float(sbs) / 2.0 if sbs not in (None, 0) else 0.5
            except Exception:
                tol = 0.5

            rs = [r for r in recs if r.get("spos") is not None]
            rs_sorted = sorted(rs, key=lambda r: r["spos"])
            clusters = []
            current = []
            last_s = None
            for r in rs_sorted:
                s = r["spos"]
                if last_s is None or abs(s - last_s) <= tol:
                    current.append(r)
                else:
                    clusters.append(current)
                    current = [r]
                last_s = s
            if current:
                clusters.append(current)

            for c in clusters:
                # representative slice sort = mean spos
                m = float(np.mean([rr["spos"] for rr in c]))
                slice_groups.append((m, c))
            slice_groups.sort(key=lambda x: x[0])
            dlog(f"Slice grouping: geometric spos clustering. tol={tol}. Groups={len(slice_groups)}")

        else:
            dlog("No slice_no and no geometric slice position available; cannot reconstruct.")
            return np.array([]), 0

        if not slice_groups:
            dlog("No slice groups formed; cannot reconstruct.")
            return np.array([]), 0

        # 2) determine T from most common frames-per-slice (mode)
        sizes = [len(g[1]) for g in slice_groups]
        T_mode = Counter(sizes).most_common(1)[0][0]
        if T_mode <= 1:
            dlog(f"Most common frames-per-slice is {T_mode} (<=1); dataset may not be cine.")
            return np.array([]), 0

        # 3) keep only slices with complete frame counts
        slice_groups = [(srt, lst) for (srt, lst) in slice_groups if len(lst) == T_mode]
        if not slice_groups:
            dlog("All slice groups were incomplete after filtering to mode frame count.")
            return np.array([]), 0

        Z = len(slice_groups)
        T = T_mode
        dlog(f"Reconstruction: using T={T}, Z={Z} (from mode frames-per-slice).")

        # 4) sort frames inside each slice by best available temporal key
        ordered_paths_by_slice = []
        for srt, lst in slice_groups:
            lst_sorted = sorted(lst, key=time_sort_key)
            ordered_paths_by_slice.append([r["path"] for r in lst_sorted])

        # 5) build volume (T,Z,H,W)
        frames_TZ = []
        for t in range(T):
            imgs = []
            for z in range(Z):
                p = ordered_paths_by_slice[z][t]
                try:
                    imgs.append(pydicom.dcmread(p, force=True).pixel_array)
                except Exception:
                    pass
            if len(imgs) == Z:
                frames_TZ.append(imgs)

        if not frames_TZ:
            dlog("Could not assemble frames after ordering.")
            return np.array([]), 0

        volume = np.array(frames_TZ)  # (T,Z,H,W)
        T_detected = volume.shape[0]

    dlog(f"Returned volume shape: {volume.shape}")
    return volume, (T_detected if T_detected else volume.shape[0])
