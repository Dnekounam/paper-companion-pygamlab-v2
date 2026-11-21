#!/usr/bin/env python3
"""Compare pygamlab data-analysis functions with SciPy on laboratory files.

Datasets
--------
DSC   DSC-NonIsothermal.xlsx   -> DSC
PL    pl_data.xlsx             -> Photoluminescence_analysis
NMR   nmr_data.xlsx            -> NMR_Analysis
AES   aes.xlsx                 -> Auger_Electron_Spectroscopy_analysis

Accuracy is the difference between the value returned by pygamlab and an
independent SciPy calculation of the same quantity. Performance is the median
build time and the number of rows in the file.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygamlab
from pygamlab.Data_Analysis import (
    Auger_Electron_Spectroscopy_analysis,
    DSC,
    NMR_Analysis,
    Photoluminescence_analysis,
)
from scipy import constants
from scipy.integrate import simpson, trapezoid
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, peak_widths

warnings.filterwarnings("ignore", message="some peaks have a prominence of 0")
warnings.filterwarnings("ignore", message="some peaks have a width of 0")
from scipy.spatial.distance import cdist

plt.show = lambda *args, **kwargs: plt.close("all")

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DATA_DIR = Path(os.environ.get("PYGAMLAB_BENCHMARK_DATA_DIR", "/Users/Dakota/Downloads"))
DATA = {
    "DSC": DATA_DIR / "DSC-NonIsothermal.xlsx",
    "PL": DATA_DIR / "pl_data.xlsx",
    "NMR": DATA_DIR / "nmr_data.xlsx",
    "AES": DATA_DIR / "aes.xlsx",
}

DSC_PROMINENCE = 0.05
DSC_DISTANCE = 20
DSC_MASS = 1.0
DSC_RATE = 1.0
DSC_ORIENTATION = "endo_up"
REPEATS = 5

NMR_MIXTURE = {
    "CompoundA": {"region": (-116.0, -114.5), "protons": 2},
    "CompoundB": {"region": (-111.8, -110.6), "protons": 1},
}
NMR_IMPURITY = {
    "main_compound": {"region": (-111.8, -110.6), "protons": 1},
    "impurity": {"region": (-107.3, -106.2), "protons": 1},
}
AES_SENSITIVITY = {"C": 0.25, "O": 0.66}


def load_tables():
    dsc = pd.read_excel(DATA["DSC"])
    dsc = dsc.iloc[1:].copy()
    dsc["Temperature"] = pd.to_numeric(dsc["Temperature"], errors="coerce")
    dsc["Value"] = pd.to_numeric(dsc["Value"], errors="coerce")
    dsc = dsc.dropna().reset_index(drop=True)

    pl = pd.read_excel(DATA["PL"])
    nmr = pd.read_excel(DATA["NMR"])
    aes = pd.read_excel(DATA["AES"])
    return {"DSC": dsc, "PL": pl, "NMR": nmr, "AES": aes}


def timed(function, repeats: int = REPEATS):
    """Warm up once, then return the last result, captured text, and median time."""
    with redirect_stdout(io.StringIO()):
        function()
    plt.close("all")
    samples = []
    result = None
    text = ""
    for _ in range(repeats):
        buffer = io.StringIO()
        start = time.perf_counter()
        with redirect_stdout(buffer):
            result = function()
        samples.append((time.perf_counter() - start) * 1000.0)
        text = buffer.getvalue()
        plt.close("all")
    return result, text, float(np.median(samples))


def rmsd(left, right):
    left = np.sort(np.asarray(list(left), dtype=float))
    right = np.sort(np.asarray(list(right), dtype=float))
    if left.size == 0 and right.size == 0:
        return 0.0
    if left.size == 0 or right.size == 0:
        return None
    distances = cdist(left[:, None], right[:, None])
    rows, cols = linear_sum_assignment(distances)
    matched = distances[rows, cols]
    return float(np.sqrt(np.mean(matched ** 2)))


def mae(left, right, paired=False):
    left = np.asarray(list(left), dtype=float)
    right = np.asarray(list(right), dtype=float)
    if left.size == 0 or right.size == 0:
        return None
    if paired and left.size == right.size:
        return float(np.mean(np.abs(left - right)))
    n = min(left.size, right.size)
    return float(np.mean(np.abs(np.sort(left)[:n] - np.sort(right)[:n])))


def show(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): show(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [show(item) for item in value]
    return value


def dsc_xy(frame):
    return frame["Temperature"].to_numpy(float), frame["Value"].to_numpy(float)


def scipy_transition(frame, kind):
    x, y = dsc_xy(frame)
    if kind == "Tg":
        slope = np.gradient(y, x)
        return float(x[np.argmax(np.abs(slope))])
    if kind == "Tm":
        peaks, _ = find_peaks(y, prominence=DSC_PROMINENCE, distance=DSC_DISTANCE)
        if peaks.size == 0:
            return None
        return float(x[peaks[np.argmax(np.abs(y[peaks]))]])
    peaks, _ = find_peaks(-y, prominence=DSC_PROMINENCE, distance=DSC_DISTANCE)
    if peaks.size == 0:
        return None
    return float(x[peaks[np.argmax(np.abs(y[peaks]))]])


def scipy_peak_positions(frame):
    x, y = dsc_xy(frame)
    endo, _ = find_peaks(y, prominence=DSC_PROMINENCE, distance=DSC_DISTANCE)
    exo, _ = find_peaks(-y, prominence=DSC_PROMINENCE, distance=DSC_DISTANCE)
    return {
        "endothermic": [float(value) for value in x[endo]],
        "exothermic": [float(value) for value in x[exo]],
    }


def scipy_onset(frame):
    """Onset where the signal has moved 5% of the way from the local baseline to the peak."""
    x, y = dsc_xy(frame)
    peaks, _ = find_peaks(np.abs(y), prominence=DSC_PROMINENCE, distance=DSC_DISTANCE)
    onsets = []
    for index in peaks:
        wing = y[max(0, index - 80):max(0, index - 40)]
        baseline = float(np.median(wing)) if wing.size else float(y[0])
        target = baseline + 0.05 * (float(y[index]) - baseline)
        left = index
        if y[index] >= baseline:
            while left > 0 and y[left] > target:
                left -= 1
        else:
            while left > 0 and y[left] < target:
                left -= 1
        onsets.append(float(x[left]))
    return onsets


def region_integral(frame, start, end):
    window = frame[(frame["ppm"] >= start) & (frame["ppm"] <= end)]
    if window.empty:
        return 0.0
    order = np.argsort(window["ppm"].to_numpy(float))
    ppm = window["ppm"].to_numpy(float)[order]
    spectrum = window["Spectrum"].to_numpy(float)[order]
    return abs(float(simpson(spectrum, x=ppm)))


def scipy_mole_fractions(frame, peak_info):
    per_proton = {}
    for name, info in peak_info.items():
        start, end = info["region"]
        per_proton[name] = region_integral(frame, start, end) / info["protons"]
    total = sum(per_proton.values())
    return {name: value / total for name, value in per_proton.items()}


def scipy_impurity(frame, peak_info):
    fractions = scipy_mole_fractions(frame, {
        "main_compound": peak_info["main_compound"],
        "impurity": peak_info["impurity"],
    })
    return 100.0 * fractions["impurity"]


def scipy_pl(frame):
    wavelength = frame["wavelength"].to_numpy(float)
    intensity = frame["intensity"].to_numpy(float)
    peaks, _ = find_peaks(intensity)
    index = peaks[np.argmax(intensity[peaks])]
    peak_wavelength = float(wavelength[index])
    peak_intensity = float(intensity[index])
    bandgap = (constants.h * constants.c) / (peak_wavelength * 1e-9 * constants.e)
    widths = peak_widths(intensity, [index], rel_height=0.5)
    left = float(np.interp(widths[2][0], np.arange(len(wavelength)), wavelength))
    right = float(np.interp(widths[3][0], np.arange(len(wavelength)), wavelength))
    return {
        "peak_position_nm": peak_wavelength,
        "peak_intensity": peak_intensity,
        "bandgap_eV": float(bandgap),
        "fwhm_nm": right - left,
    }


def scipy_aes_peaks(frame):
    intensity = frame["Intensity (Counts)"].to_numpy(float)
    energy = frame["Energy (eV)"].to_numpy(float)
    peaks, _ = find_peaks(intensity, height=float(np.mean(intensity)))
    return {
        "positions_eV": [float(value) for value in energy[peaks]],
        "intensities": [float(value) for value in intensity[peaks]],
    }


def scipy_aes_atomic(frame, sensitivity):
    intensity = frame["Intensity (Counts)"].to_numpy(float)
    energy = frame["Energy (eV)"].to_numpy(float)
    peaks, _ = find_peaks(intensity, height=float(np.mean(intensity)))
    order = peaks[np.argsort(energy[peaks])]
    elements = list(sensitivity)
    used = order[:len(elements)]
    corrected = {}
    for element, index in zip(elements, used):
        left = max(0, int(index) - 3)
        right = min(len(energy), int(index) + 4)
        area = float(trapezoid(intensity[left:right], energy[left:right]))
        corrected[element] = area / sensitivity[element]
    total = sum(corrected.values())
    return [
        {"Element": element, "Atomic %": 100.0 * value / total}
        for element, value in corrected.items()
    ]


def parse_mole_fractions(text):
    fractions = {}
    for line in text.splitlines():
        prefix = "Mole Fraction of "
        if line.startswith(prefix) and ":" in line:
            name, value = line.split(":", 1)
            fractions[name.replace(prefix, "").strip()] = float(value)
    return fractions


def parse_impurity(text):
    for line in text.splitlines():
        if line.startswith("Estimated Impurity:"):
            return float(line.split(":")[1].strip().rstrip("%"))
    return None


def record(task, reference, dataset, n_points, metric, error, gam_value, scipy_value, gam_ms, scipy_ms, scipy_method):
    return {
        "task": task,
        "reference": reference,
        "dataset": dataset,
        "n_points": int(n_points),
        "accuracy_metric": metric,
        "error": None if error is None else round(float(error), 6),
        "pygamlab_value": show(gam_value),
        "scipy_value": show(scipy_value),
        "time_pygamlab_ms": round(gam_ms, 3),
        "time_scipy_ms": round(scipy_ms, 3),
        "scipy_method": scipy_method,
    }


def absolute_error(gam_value, scipy_value):
    if gam_value is None or scipy_value is None:
        return None
    return abs(float(gam_value) - float(scipy_value))


def run_cases(tables):
    dsc = tables["DSC"]
    pl = tables["PL"]
    nmr = tables["NMR"]
    aes = tables["AES"]
    rows = []

    dsc_call = (
        f"DSC(dsc_data, application={{app!r}}, prominence={DSC_PROMINENCE}, "
        f"distance={DSC_DISTANCE}, sample_mass={DSC_MASS}, heating_rate={DSC_RATE}, "
        f"orientation={DSC_ORIENTATION!r})"
    )

    for application, kind in (("Tg", "Tg"), ("Tm", "Tm"), ("Tc", "Tc")):
        gam, _, gam_ms = timed(lambda application=application: DSC(
            dsc, application=application, prominence=DSC_PROMINENCE, distance=DSC_DISTANCE,
            sample_mass=DSC_MASS, heating_rate=DSC_RATE, orientation=DSC_ORIENTATION,
        ))
        ref, _, ref_ms = timed(lambda kind=kind: scipy_transition(dsc, kind))
        rows.append(record(
            f"dsc_{application}",
            dsc_call.format(app=application),
            DATA["DSC"].name,
            len(dsc),
            "absolute_error_C",
            absolute_error(gam, ref),
            gam, ref, gam_ms, ref_ms,
            "numpy.gradient max |dy/dx|" if application == "Tg" else "scipy.signal.find_peaks",
        ))
        print(f"  dsc_{application}: error {rows[-1]['error']} {rows[-1]['accuracy_metric']}")

    gam, _, gam_ms = timed(lambda: DSC(
        dsc, application="peak_detection", prominence=DSC_PROMINENCE, distance=DSC_DISTANCE,
        sample_mass=DSC_MASS, heating_rate=DSC_RATE, orientation=DSC_ORIENTATION,
    ))
    ref, _, ref_ms = timed(lambda: scipy_peak_positions(dsc))
    gam_positions = [item[0] for item in gam["endothermic"]] + [item[0] for item in gam["exothermic"]]
    ref_positions = ref["endothermic"] + ref["exothermic"]
    rows.append(record(
        "dsc_peak_detection",
        dsc_call.format(app="peak_detection"),
        DATA["DSC"].name, len(dsc), "rmsd_C",
        rmsd(gam_positions, ref_positions),
        gam_positions, ref_positions, gam_ms, ref_ms,
        "scipy.signal.find_peaks",
    ))

    gam, _, gam_ms = timed(lambda: DSC(
        dsc, application="kinetics", prominence=DSC_PROMINENCE, distance=DSC_DISTANCE,
        sample_mass=DSC_MASS, heating_rate=DSC_RATE, orientation=DSC_ORIENTATION,
    ))
    ref, _, ref_ms = timed(lambda: scipy_onset(dsc))
    gam_onset = [item["Onset"] for item in gam] if gam else []
    rows.append(record(
        "dsc_kinetics",
        dsc_call.format(app="kinetics"),
        DATA["DSC"].name, len(dsc), "mae_onset_C",
        mae(gam_onset, ref, paired=True),
        gam_onset, ref, gam_ms, ref_ms,
        "onset at 5% of peak height above the local baseline",
    ))

    pl_ref, _, pl_ref_ms = timed(lambda: scipy_pl(pl))
    pl_commands = {
        "peak_position": ("pl_peak_position", "absolute_error_nm", "peak_position_nm", "scipy.signal.find_peaks"),
        "peak_intensity": ("pl_peak_intensity", "absolute_error", "peak_intensity", "scipy.signal.find_peaks"),
        "bandgap_energy": ("pl_bandgap", "absolute_error_eV", "bandgap_eV", "scipy.constants h*c/lambda"),
        "fwhm": ("pl_fwhm", "absolute_error_nm", "fwhm_nm", "scipy.signal.peak_widths"),
    }
    # One SciPy spectrum analysis is shared. Time each pygamlab application on its own.
    for application, (task, metric, key, method) in pl_commands.items():
        gam, _, gam_ms = timed(lambda application=application: Photoluminescence_analysis(pl, application=application))
        rows.append(record(
            task,
            f"Photoluminescence_analysis(pl_data, application={application!r})",
            DATA["PL"].name, len(pl), metric,
            absolute_error(gam, pl_ref[key]),
            gam, pl_ref[key], gam_ms, pl_ref_ms, method,
        ))
        print(f"  {task}: error {rows[-1]['error']} {metric}")

    _, mixture_text, gam_ms = timed(lambda: NMR_Analysis(
        nmr, application="mixture_composition", peak_info=NMR_MIXTURE,
    ))
    ref, _, ref_ms = timed(lambda: scipy_mole_fractions(nmr, NMR_MIXTURE))
    gam_fractions = parse_mole_fractions(mixture_text)
    rows.append(record(
        "nmr_mixture_composition",
        "NMR_Analysis(nmr_data, application='mixture_composition', peak_info="
        "{'CompoundA': {'region': (-116.0, -114.5), 'protons': 2}, "
        "'CompoundB': {'region': (-111.8, -110.6), 'protons': 1}})",
        DATA["NMR"].name, len(nmr), "mae_mole_fraction",
        mae(gam_fractions.values(), [ref[name] for name in gam_fractions]),
        gam_fractions, ref, gam_ms, ref_ms,
        "scipy.integrate.simpson per proton",
    ))

    _, impurity_text, gam_ms = timed(lambda: NMR_Analysis(
        nmr, application="calculate_impurity", peak_info=NMR_IMPURITY,
    ))
    ref, _, ref_ms = timed(lambda: scipy_impurity(nmr, NMR_IMPURITY))
    gam_impurity = parse_impurity(impurity_text)
    rows.append(record(
        "nmr_impurity",
        "NMR_Analysis(nmr_data, application='calculate_impurity', peak_info="
        "{'main_compound': {'region': (-111.8, -110.6), 'protons': 1}, "
        "'impurity': {'region': (-107.3, -106.2), 'protons': 1}})",
        DATA["NMR"].name, len(nmr), "absolute_error_percent",
        absolute_error(gam_impurity, ref),
        gam_impurity, ref, gam_ms, ref_ms,
        "scipy.integrate.simpson per proton",
    ))

    gam, _, gam_ms = timed(lambda: Auger_Electron_Spectroscopy_analysis(aes, application="peak_position"))
    ref, _, ref_ms = timed(lambda: scipy_aes_peaks(aes))
    rows.append(record(
        "aes_peak_position",
        "Auger_Electron_Spectroscopy_analysis(aes_data, application='peak_position')",
        DATA["AES"].name, len(aes), "rmsd_eV",
        rmsd(gam["Peak Positions (eV)"], ref["positions_eV"]),
        [float(value) for value in gam["Peak Positions (eV)"]],
        ref["positions_eV"], gam_ms, ref_ms,
        "scipy.signal.find_peaks",
    ))

    gam, _, gam_ms = timed(lambda: Auger_Electron_Spectroscopy_analysis(
        aes, application="atomic", sensitivity_factors=AES_SENSITIVITY,
    ))
    ref, _, ref_ms = timed(lambda: scipy_aes_atomic(aes, AES_SENSITIVITY))
    gam_percent = [item["Atomic %"] for item in gam]
    ref_percent = [item["Atomic %"] for item in ref]
    rows.append(record(
        "aes_atomic",
        "Auger_Electron_Spectroscopy_analysis(aes_data, application='atomic', "
        "sensitivity_factors={'C': 0.25, 'O': 0.66})",
        DATA["AES"].name, len(aes), "mae_atomic_percent",
        mae(gam_percent, ref_percent),
        gam, ref, gam_ms, ref_ms,
        "scipy.integrate.trapezoid peak area / sensitivity factor",
    ))
    return rows


def write_csv(rows):
    path = OUT_DIR / "results.csv"
    flat = []
    for row in rows:
        copied = dict(row)
        copied["pygamlab_value"] = json.dumps(copied["pygamlab_value"])
        copied["scipy_value"] = json.dumps(copied["scipy_value"])
        flat.append(copied)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    return path


def write_summary(rows):
    path = OUT_DIR / "summary.txt"
    header = (
        f"{'task':<26} {'n':>6} {'metric':<24} {'error':>12} "
        f"{'t_gam_ms':>10} {'t_scipy_ms':>10}  reference"
    )
    lines = [
        "pygamlab data analysis vs SciPy",
        f"pygamlab {pygamlab.__version__}    SciPy constants {constants.h:.6e} J s",
        "",
        "Accuracy: |pygamlab - SciPy| for one number, RMSD or MAE when a task returns several.",
        "Performance: median call time after one warmup, and number of data rows.",
        "DSC orientation is endo_up. NMR regions are the strong peaks between -116 and -106 ppm.",
        "",
        header,
        "-" * len(header),
    ]
    for row in rows:
        error = "n/a" if row["error"] is None else f"{row['error']:.6g}"
        lines.append(
            f"{row['task']:<26} {row['n_points']:6d} {row['accuracy_metric']:<24} {error:>12} "
            f"{row['time_pygamlab_ms']:10.3f} {row['time_scipy_ms']:10.3f}  {row['reference']}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    print("running DSC, PL, NMR, and AES against SciPy")
    rows = run_cases(tables)
    payload = {
        "pygamlab_version": pygamlab.__version__,
        "datasets": {name: str(path) for name, path in DATA.items()},
        "accuracy_metrics": ["absolute_error", "rmsd", "mae"],
        "performance_metrics": ["time_pygamlab_ms", "time_scipy_ms", "n_points"],
        "results": rows,
    }
    json_path = OUT_DIR / "results.json"
    json_path.write_text(json.dumps(payload, indent=2))
    csv_path = write_csv(rows)
    summary_path = write_summary(rows)
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
