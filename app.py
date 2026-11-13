"""
app.py — FastAPI Backend for SPHEREx Epoch Subtraction & Anomaly Discovery
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import sys
import traceback
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from astropy.time import Time

from pipeline import (
    query_spherex_cutouts,
    resolve_datalink_to_cutout_url,
    subtract_epochs,
    detect_anomalies,
    render_images_base64,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app = FastAPI(
    title="SPHEREx Epoch Subtraction & Anomaly Discovery Engine",
    description="Full-stack astronomical image subtraction, WCS reprojection, and moving/variable source detection.",
    version="1.0.1",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
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


# In-memory cache for fast reuse across calls
OBSERVATIONS_CACHE: Dict[tuple, Any] = {}
CUTOUT_URL_CACHE: Dict[str, str] = {}


# ── Request / Response Models ───────────────────────────────────────────────

class SubtractionRequest(BaseModel):
    ra: float = Field(..., description="Target Right Ascension (degrees, ICRS)")
    dec: float = Field(..., description="Target Declination (degrees, ICRS)")
    size: float = Field(0.05, description="Search and cutout size (degrees)")
    bandpass: str = Field("SPHEREx-D2", description="Bandpass designation")
    epoch1_index: int = Field(0, description="Index of reference/baseline epoch (0=oldest)")
    epoch2_index: int = Field(-1, description="Index of secondary epoch (-1=newest)")
    sigma_thresh: float = Field(3.0, description="Detection significance threshold")


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    """Serve the Vanilla JS + Tailwind CSS frontend."""
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(INDEX_FILE))


@app.get("/api/presets")
async def get_presets():
    """Return pre-configured NASA science targets for instant testing."""
    return {"presets": PRESET_TARGETS}


@app.get("/api/epochs")
async def get_epochs(
    ra: float = Query(..., description="Right Ascension in degrees"),
    dec: float = Query(..., description="Declination in degrees"),
    size: float = Query(0.05, description="Search size in degrees"),
    bandpass: str = Query("SPHEREx-D2", description="Bandpass name")
):
    """
    Query the IRSA TAP service for SPHEREx observations covering (ra, dec)
    and return available observation timestamps sorted by time.
    """
    cache_key = (round(ra, 4), round(dec, 4), round(size, 4), bandpass)
    try:
        if cache_key in OBSERVATIONS_CACHE:
            table = OBSERVATIONS_CACHE[cache_key]
        else:
            table = query_spherex_cutouts(ra, dec, size=size, bandpass=bandpass)
            OBSERVATIONS_CACHE[cache_key] = table

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
    """
    Execute end-to-end epoch subtraction pipeline:
    1. Fetch / resolve cutout FITS files
    2. WCS reprojection & background-subtracted difference
    3. Anomaly detection via sigma-clipped peak analysis
    4. Render base64 PNGs for individual panels and 3-panel comparison
    """
    cache_key = (round(req.ra, 4), round(req.dec, 4), round(req.size, 4), req.bandpass)
    try:
        if cache_key in OBSERVATIONS_CACHE:
            table = OBSERVATIONS_CACHE[cache_key]
        else:
            table = query_spherex_cutouts(req.ra, req.dec, size=req.size, bandpass=req.bandpass)
            OBSERVATIONS_CACHE[cache_key] = table

        n_epochs = len(table)
        if n_epochs < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Found only {n_epochs} observation(s) for RA={req.ra}, Dec={req.dec}, bandpass={req.bandpass}. Minimum 2 required for temporal subtraction."
            )

        # Normalize indices safely
        idx1 = req.epoch1_index % n_epochs
        idx2 = req.epoch2_index % n_epochs

        row1 = table[idx1]
        row2 = table[idx2]

        datalink1 = str(row1["access_url"])
        datalink2 = str(row2["access_url"])

        # Resolve DataLinks to direct FITS cutout URLs
        dl_key1 = f"{datalink1}_{req.ra}_{req.dec}_{req.size}"
        dl_key2 = f"{datalink2}_{req.ra}_{req.dec}_{req.size}"

        if dl_key1 not in CUTOUT_URL_CACHE:
            CUTOUT_URL_CACHE[dl_key1] = resolve_datalink_to_cutout_url(datalink1, req.ra, req.dec, req.size)
        fits_url1 = CUTOUT_URL_CACHE[dl_key1]

        if dl_key2 not in CUTOUT_URL_CACHE:
            CUTOUT_URL_CACHE[dl_key2] = resolve_datalink_to_cutout_url(datalink2, req.ra, req.dec, req.size)
        fits_url2 = CUTOUT_URL_CACHE[dl_key2]

        # Execute subtraction
        epoch1, epoch2_aligned, difference, wcs1 = subtract_epochs(fits_url1, fits_url2)

        # Detect moving/variable anomalies
        anomalies, stats = detect_anomalies(difference, wcs1, sigma_thresh=req.sigma_thresh)

        # Render visualizations
        images_b64 = render_images_base64(epoch1, epoch2_aligned, difference, wcs1, anomalies=anomalies)

        # Temporal delta
        t1 = float(row1["t_min"])
        t2 = float(row2["t_min"])
        delta_days = round(abs(t2 - t1), 2)

        def format_meta(row, idx):
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
                "date_human": d_human
            }

        return {
            "status": "success",
            "target": {
                "ra": req.ra,
                "dec": req.dec,
                "size": req.size,
                "bandpass": req.bandpass
            },
            "epoch1": format_meta(row1, idx1),
            "epoch2": format_meta(row2, idx2),
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
