"""
app.py — FastAPI Backend for SPHEREx Epoch Subtraction & Anomaly Discovery
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import traceback
from typing import Any, Dict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from astropy.time import Time

from pipeline import (
    VALID_COLLAPSE_MODES,
    query_spherex_cutouts,
    resolve_datalink_to_cutout_url,
    subtract_epochs,
    detect_anomalies,
    render_images_base64,
    render_single_panel,
    load_and_clean_cutout,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app = FastAPI(
    title="SPHEREx Epoch Subtraction & Anomaly Discovery Engine",
    description="Full-stack astronomical image subtraction, WCS reprojection, and moving/variable source detection.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── NASA Preset Targets ─────────────────────────────────────────────────────

PRESET_TARGETS = [
    {
        "id": "wise_0855",
        "name": "WISE 0855−0714",
        "category": "Cold Brown Dwarf",
        "ra": 133.795129,
        "dec": -7.245147,
        "size": 0.05,
        "bandpass": "SPHEREx-D2",
        "description": "Coldest known brown dwarf (~250 K) with large proper motion (~8.1 arcsec/yr), creating textbook dipole signatures.",
        "expected": "Moving Dipole (High Proper Motion)"
    },
    {
        "id": "barnards_star",
        "name": "Barnard's Star",
        "category": "High Proper Motion Star",
        "ra": 269.452075,
        "dec": 4.693391,
        "size": 0.05,
        "bandpass": "SPHEREx-D2",
        "description": "Second nearest star system to the Sun with the fastest known proper motion across the sky (10.3 arcsec/yr).",
        "expected": "Extreme Proper Motion Shift"
    },
    {
        "id": "epsilon_indi_b",
        "name": "Epsilon Indi B",
        "category": "Binary Brown Dwarf",
        "ra": 330.8402,
        "dec": -56.7859,
        "size": 0.05,
        "bandpass": "SPHEREx-D2",
        "description": "Methane brown dwarf binary system in Indus constellation exhibiting fast motion against the background.",
        "expected": "Proper Motion Dipole"
    },
    {
        "id": "rw_aurigae",
        "name": "RW Aurigae",
        "category": "Extreme Variable T Tauri",
        "ra": 76.9565,
        "dec": 30.4014,
        "size": 0.05,
        "bandpass": "SPHEREx-D2",
        "description": "Young eruptive variable star system undergoing intense occultation and photometric dimming episodes.",
        "expected": "Photometric Flux Variable"
    },
    {
        "id": "r_draconis",
        "name": "R Draconis",
        "category": "Mira Long-Period Variable",
        "ra": 249.5965,
        "dec": 66.7546,
        "size": 0.05,
        "bandpass": "SPHEREx-D2",
        "description": "Pulsating asymptotic giant branch star showing large-amplitude infrared luminosity variations over ~245 days.",
        "expected": "Infrared Amplitude Variation"
    }
]


OBSERVATIONS_CACHE: Dict[tuple, Any] = {}
CUTOUT_URL_CACHE: Dict[str, str] = {}
TIMELINE_PNG_CACHE: Dict[str, str] = {}


class SubtractionRequest(BaseModel):
    ra: float = Field(..., description="Target Right Ascension (degrees, ICRS)")
    dec: float = Field(..., description="Target Declination (degrees, ICRS)")
    size: float = Field(0.05, description="Search and cutout size (degrees)")
    bandpass: str = Field("SPHEREx-D2", description="Bandpass designation")
    epoch1_index: int = Field(0, description="Index of reference/baseline epoch (0=oldest)")
    epoch2_index: int = Field(-1, description="Index of secondary epoch (-1=newest)")
    sigma_thresh: float = Field(3.0, description="Detection significance threshold")
    collapse_mode: str = Field("mid", description="Spectral collapse: mid | sum | median | detector")


class TimelineRequest(BaseModel):
    ra: float
    dec: float
    size: float = 0.05
    bandpass: str = "SPHEREx-D2"
    collapse_mode: str = "mid"
    max_frames: int = Field(8, ge=2, le=12)


def _normalize_collapse(mode: str) -> str:
    m = (mode or "mid").strip().lower()
    return m if m in VALID_COLLAPSE_MODES else "mid"


def _get_obs_table(ra: float, dec: float, size: float, bandpass: str):
    cache_key = (round(ra, 4), round(dec, 4), round(size, 4), bandpass)
    if cache_key in OBSERVATIONS_CACHE:
        return OBSERVATIONS_CACHE[cache_key], cache_key
    table = query_spherex_cutouts(ra, dec, size=size, bandpass=bandpass)
    OBSERVATIONS_CACHE[cache_key] = table
    return table, cache_key


def _resolve_fits_url(datalink: str, ra: float, dec: float, size: float) -> str:
    dl_key = f"{datalink}_{ra}_{dec}_{size}"
    if dl_key not in CUTOUT_URL_CACHE:
        CUTOUT_URL_CACHE[dl_key] = resolve_datalink_to_cutout_url(datalink, ra, dec, size)
    return CUTOUT_URL_CACHE[dl_key]


def _format_epoch_meta(row, idx: int) -> dict:
    t = float(row["t_min"])
    try:
        d_iso = Time(t, format="mjd").isot
        d_human = d_iso.replace("T", " ")[:19] + " UTC"
    except Exception:
        d_iso = f"MJD {t:.3f}"
        d_human = d_iso
    return {
        "index": idx,
        "obs_id": str(row["obs_id"]) if "obs_id" in row.colnames else f"Obs_{idx}",
        "mjd": t,
        "date_iso": d_iso,
        "date_human": d_human,
    }


@app.get("/")
async def serve_index():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(INDEX_FILE))


@app.get("/api/presets")
async def get_presets():
    return {"presets": PRESET_TARGETS}


@app.get("/api/resolve")
async def resolve_name(name: str = Query(..., min_length=1)):
    """Resolve object name → RA/Dec via Sesame / SkyCoord.from_name."""
    try:
        from astropy.coordinates import SkyCoord

        c = SkyCoord.from_name(name.strip())
        return {
            "status": "success",
            "name": name.strip(),
            "ra": float(c.ra.degree),
            "dec": float(c.dec.degree),
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Could not resolve '{name}': {exc}") from exc


@app.get("/api/epochs")
async def get_epochs(
    ra: float = Query(..., description="Right Ascension in degrees"),
    dec: float = Query(..., description="Declination in degrees"),
    size: float = Query(0.05, description="Search size in degrees"),
    bandpass: str = Query("SPHEREx-D2", description="Bandpass name")
):
    try:
        table, _ = _get_obs_table(ra, dec, size, bandpass)

        epochs = []
        for idx, row in enumerate(table):
            t_min = float(row["t_min"])
            t_max = float(row["t_max"]) if "t_max" in row.colnames else t_min

            try:
                date_iso = Time(t_min, format="mjd").isot
                date_human = date_iso.replace("T", " ")[:19] + " UTC"
            except Exception:
                date_iso = f"MJD {t_min:.3f}"
                date_human = date_iso

            epochs.append({
                "index": idx,
                "obs_id": str(row["obs_id"]) if "obs_id" in row.colnames else f"Obs_{idx}",
                "t_min": t_min,
                "t_max": t_max,
                "date_iso": date_iso,
                "date_human": date_human,
                "access_url": str(row["access_url"]),
                "bandpass": str(row["energy_bandpassname"]) if "energy_bandpassname" in row.colnames else bandpass
            })

        return {
            "status": "success",
            "total_epochs": len(epochs),
            "target": {"ra": ra, "dec": dec, "size": size, "bandpass": bandpass},
            "epochs": epochs
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to query TAP epochs: {str(exc)}")


@app.post("/api/subtract")
async def run_subtraction(req: SubtractionRequest):
    collapse_mode = _normalize_collapse(req.collapse_mode)
    try:
        table, _ = _get_obs_table(req.ra, req.dec, req.size, req.bandpass)

        n_epochs = len(table)
        if n_epochs < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Found only {n_epochs} observation(s) for RA={req.ra}, Dec={req.dec}, "
                    f"bandpass={req.bandpass}. Minimum 2 required for temporal subtraction."
                ),
            )

        idx1 = req.epoch1_index % n_epochs
        idx2 = req.epoch2_index % n_epochs

        row1 = table[idx1]
        row2 = table[idx2]

        fits_url1 = _resolve_fits_url(str(row1["access_url"]), req.ra, req.dec, req.size)
        fits_url2 = _resolve_fits_url(str(row2["access_url"]), req.ra, req.dec, req.size)

        epoch1, epoch2_aligned, difference, wcs1, collapse_meta = subtract_epochs(
            fits_url1, fits_url2, collapse_mode=collapse_mode
        )

        anomalies, stats = detect_anomalies(difference, wcs1, sigma_thresh=req.sigma_thresh)
        images_b64 = render_images_base64(epoch1, epoch2_aligned, difference, wcs1, anomalies=anomalies)

        t1 = float(row1["t_min"])
        t2 = float(row2["t_min"])
        delta_days = round(abs(t2 - t1), 2)

        return {
            "status": "success",
            "target": {
                "ra": req.ra,
                "dec": req.dec,
                "size": req.size,
                "bandpass": req.bandpass,
                "collapse_mode": collapse_mode,
            },
            "collapse_mode": collapse_mode,
            "collapse_meta": collapse_meta,
            "epoch1": _format_epoch_meta(row1, idx1),
            "epoch2": _format_epoch_meta(row2, idx2),
            "delta_days": delta_days,
            "grid_shape": [int(s) for s in epoch1.shape],
            "stats": stats,
            "anomalies_count": len(anomalies),
            "anomalies": anomalies,
            "images_base64": images_b64
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Subtraction error: {type(exc).__name__} - {str(exc)}"
        )


@app.post("/api/timeline")
async def build_timeline(req: TimelineRequest):
    """
    Build N cleaned epoch PNGs for multi-frame Blink Timeline mode.
    Reuses observation + DataLink + FITS caches; PNG cache keyed by obs+collapse.
    """
    collapse_mode = _normalize_collapse(req.collapse_mode)
    max_frames = max(2, min(12, int(req.max_frames)))
    try:
        table, _ = _get_obs_table(req.ra, req.dec, req.size, req.bandpass)
        n_epochs = len(table)
        if n_epochs < 1:
            raise HTTPException(status_code=400, detail="No observations found for timeline.")

        # Evenly sample up to max_frames across the archive
        if n_epochs <= max_frames:
            indices = list(range(n_epochs))
        else:
            indices = sorted({int(round(i)) for i in np.linspace(0, n_epochs - 1, max_frames)})

        frames = []
        for i, idx in enumerate(indices):
            row = table[idx]
            meta = _format_epoch_meta(row, idx)
            obs_id = meta["obs_id"]
            cache_key = (
                f"{obs_id}:{req.ra:.4f}:{req.dec:.4f}:{req.size:.4f}:"
                f"{req.bandpass}:{collapse_mode}"
            )

            if cache_key in TIMELINE_PNG_CACHE:
                png_b64 = TIMELINE_PNG_CACHE[cache_key]
                n_bands = 1
            else:
                fits_url = _resolve_fits_url(str(row["access_url"]), req.ra, req.dec, req.size)
                cleaned, wcs, _header, clean_meta = load_and_clean_cutout(
                    fits_url, collapse_mode=collapse_mode
                )
                n_bands = clean_meta.get("n_bands", 1)
                title = f"Epoch {idx} · {meta['date_human'][:10]}"
                png_b64 = render_single_panel(
                    cleaned, wcs, title, cmap="viridis", label="MJy/sr"
                )
                TIMELINE_PNG_CACHE[cache_key] = png_b64

            frames.append({
                "index": idx,
                "obs_id": obs_id,
                "mjd": meta["mjd"],
                "date_iso": meta["date_iso"],
                "date_human": meta["date_human"],
                "n_bands": n_bands,
                "png_base64": png_b64,
                "progress": {"current": i + 1, "total": len(indices)},
            })

        return {
            "status": "success",
            "collapse_mode": collapse_mode,
            "n_frames": len(frames),
            "total_epochs_available": n_epochs,
            "target": {
                "ra": req.ra,
                "dec": req.dec,
                "size": req.size,
                "bandpass": req.bandpass,
            },
            "frames": frames,
        }
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Timeline error: {type(exc).__name__} - {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
