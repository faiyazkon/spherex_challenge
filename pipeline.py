"""
pipeline.py — Robust SPHEREx epoch-subtraction and anomaly-detection pipeline.

Features:
- Query IRSA TAP for SPHEREx observations
- DataLink resolver with HTTP retry & backoff
- Robust multi-extension FITS loader with local caching:
  ZODI background subtraction and proper flag filtering:
  preserves real astronomical sources (bit 21 = 2097152) and benign flag 2.
- WCS reprojection and epoch difference calculation
- Statistical anomaly detection (sigma-clipped local peak/dipole identification)
- High-quality astronomical image rendering to base64 PNGs
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import io
import time
import hashlib
import base64
from pathlib import Path
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from astropy.io import fits
from astropy.wcs import WCS
from astropy.io.votable import parse as parse_votable
from astropy.stats import sigma_clipped_stats
from reproject import reproject_interp
from scipy.ndimage import maximum_filter, minimum_filter
import pyvo


# Local cache directory for downloaded FITS cutouts
CACHE_DIR = Path("/tmp/spherex_cutouts")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# HTTP Session with Automatic Retries
# ---------------------------------------------------------------------------

def get_http_session(retries=3, backoff_factor=1.0):
    """Create a requests session with automatic retries on transient server errors."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


HTTP_SESSION = get_http_session()


# ---------------------------------------------------------------------------
# 1. Query SPHEREx cutouts from IRSA TAP
# ---------------------------------------------------------------------------

def query_spherex_cutouts(ra, dec, size=0.05, bandpass="SPHEREx-D2"):
    """
    Query the IRSA TAP service for SPHEREx observations covering (ra, dec).

    Parameters
    ----------
    ra, dec : float
        ICRS coordinates in degrees.
    size : float
        Search radius in degrees.
    bandpass : str
        SPHEREx bandpass name to filter on (e.g. 'SPHEREx-D2').

    Returns
    -------
    astropy.table.Table
        Rows sorted by ascending t_min (earliest observation first).
    """
    tap = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")

    query = f"""
        SELECT *
        FROM spherex.obscore
        WHERE CONTAINS(
                POINT('ICRS', {ra}, {dec}),
                s_region
              ) = 1
          AND energy_bandpassname = '{bandpass}'
        ORDER BY t_min ASC
    """
    result = tap.search(query)
    return result.to_table()


# ---------------------------------------------------------------------------
# 1b. Resolve a DataLink URL to a cutout FITS URL
# ---------------------------------------------------------------------------

def resolve_datalink_to_cutout_url(datalink_url, ra, dec, size=0.05):
    """
    Resolve a DataLink access_url to a direct FITS cutout URL with retries.

    Parameters
    ----------
    datalink_url : str
        The access_url column value from the TAP query.
    ra, dec : float
        ICRS coordinates in degrees for the cutout centre.
    size : float
        Cutout size in degrees.

    Returns
    -------
    str
        Direct URL to the FITS cutout file.
    """
    last_err = None
    for attempt in range(3):
        try:
            resp = HTTP_SESSION.get(datalink_url, timeout=45)
            if resp.status_code == 200:
                votable = parse_votable(io.BytesIO(resp.content))
                results_table = None
                for resource in votable.resources:
                    if resource.type == "results" and resource.tables:
                        results_table = resource.tables[0].to_table()
                        break
                if results_table is None:
                    results_table = votable.get_first_table().to_table()

                sem_col = "semantics" if "semantics" in results_table.colnames else "col_4"
                url_col = "access_url" if "access_url" in results_table.colnames else "col_1"
                path_col = "path" if "path" in results_table.colnames else "col_12"

                # Prefer the cutout service URL (much smaller download)
                cutout_mask = results_table[sem_col] == "#cutout"
                if cutout_mask.any():
                    path = str(results_table[path_col][cutout_mask][0])
                    cutout_base = "https://irsa.ipac.caltech.edu/ibe/cutout"
                    return f"{cutout_base}?path={path}&ra={ra}&dec={dec}&size={size}"

                # Fallback: full science image direct URL
                this_mask = results_table[sem_col] == "#this"
                if this_mask.any():
                    return str(results_table[url_col][this_mask][0])

                raise ValueError("No #cutout or #this entry found in DataLink response.")
            else:
                last_err = f"DataLink HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"Failed to resolve DataLink URL after 3 attempts: {last_err}")


# ---------------------------------------------------------------------------
# 2. Load & clean a single SPHEREx cutout
# ---------------------------------------------------------------------------

def download_fits_to_cache(url, timeout=60):
    """
    Download a remote FITS file to local disk cache using robust retries.
    Avoids astropy internal urllib errors and accelerates repeated accesses.
    """
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    cached_path = CACHE_DIR / f"cutout_{url_hash}.fits"

    if cached_path.exists() and cached_path.stat().st_size > 1000:
        return cached_path

    last_exc = None
    for attempt in range(3):
        try:
            resp = HTTP_SESSION.get(url, timeout=timeout, stream=True)
            if resp.status_code == 200:
                tmp_path = CACHE_DIR / f"tmp_{url_hash}_{attempt}.fits"
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                tmp_path.replace(cached_path)
                return cached_path
            else:
                last_exc = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            last_exc = str(exc)
            time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"Could not download FITS from {url}: {last_exc}")


# Spectral collapse modes (mid preserves legacy Faiyaz behavior)
VALID_COLLAPSE_MODES = frozenset({"mid", "sum", "median", "detector"})
# FLAGS: bit 21 = SOURCE, bit 1 = benign — must NOT be masked
_FLAG_KEEP = 2097152 | 2  # 2097154


def collapse_cube(cube, mode="mid"):
    """Collapse a spectral cube [wave, y, x] → 2-D spatial image."""
    data = np.asarray(cube, dtype=np.float64)
    if data.ndim == 2:
        return data
    if data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")
    mode = mode if mode in VALID_COLLAPSE_MODES else "mid"
    if mode == "mid":
        return data[data.shape[0] // 2]
    if mode == "median":
        return np.nanmedian(data, axis=0)
    if mode == "detector":
        n = data.shape[0]
        groups = np.array_split(np.arange(n), 6)
        stacks = [np.nansum(data[g], axis=0) for g in groups if len(g)]
        if not stacks:
            return np.nansum(data, axis=0)
        return np.nanmean(np.stack(stacks, axis=0), axis=0)
    # sum / nansum
    return np.nansum(data, axis=0)


def smart_flag_mask(flags, mode="mid"):
    """True where pixels are bad glitches (exclude SOURCE + benign bits)."""
    if flags is None:
        return None
    f2 = np.asarray(flags)
    bad = (f2.astype(np.int64) & ~_FLAG_KEEP) != 0
    if f2.ndim == 3:
        if mode == "mid":
            return bad[bad.shape[0] // 2]
        # Any wavelength plane with a true glitch → mask spatial pixel
        return np.any(bad, axis=0)
    return bad


def load_and_clean_cutout(fits_url_or_path, collapse_mode="mid"):
    """
    Download a SPHEREx MEF cutout/image, subtract the zodiacal background,
    and mask bad glitch pixels.

    In the SPHEREx FLAGS HDU:
    - Bit 21 (2097152) = astronomical SOURCE pixels (must NOT be masked)
    - Bit 1 (2) = benign calibration/warning flag (must NOT be masked)
    Only bad glitch pixels where (flags & ~(2097152 | 2)) != 0 are masked with NaN.

    Parameters
    ----------
    fits_url_or_path : str
        Direct URL or local filepath to the FITS file.
    collapse_mode : str
        How to reduce spectral cubes: mid | sum | median | detector.

    Returns
    -------
    cleaned : numpy.ndarray (2-D, float64)
        IMAGE − ZODI with bad glitch pixels set to NaN.
    wcs : astropy.wcs.WCS
        2-D celestial WCS parsed from the IMAGE header.
    header : astropy.io.fits.Header
        The IMAGE extension header.
    meta : dict
        Collapse metadata (n_bands, collapse_mode).
    """
    mode = collapse_mode if collapse_mode in VALID_COLLAPSE_MODES else "mid"

    if str(fits_url_or_path).startswith("http://") or str(fits_url_or_path).startswith("https://"):
        local_path = download_fits_to_cache(str(fits_url_or_path))
    else:
        local_path = Path(fits_url_or_path)

    with fits.open(str(local_path), memmap=False) as hdulist:
        image_data = hdulist["IMAGE"].data.astype(np.float64)
        image_header = hdulist["IMAGE"].header
        n_bands = int(image_data.shape[0]) if image_data.ndim == 3 else 1

        zodi_data = hdulist["ZODI"].data.astype(np.float64)
        flags_data = hdulist["FLAGS"].data

        # Align ZODI to IMAGE shape, then collapse both with the same mode
        if zodi_data.shape == image_data.shape:
            cleaned_full = image_data - zodi_data
        elif image_data.ndim == 3 and zodi_data.ndim == 2 and zodi_data.shape == image_data.shape[1:]:
            cleaned_full = image_data - zodi_data[np.newaxis, :, :]
        elif image_data.ndim == 2 and zodi_data.ndim == 3:
            cleaned_full = image_data - collapse_cube(zodi_data, mode)
        else:
            img2 = collapse_cube(image_data, mode)
            zodi2 = collapse_cube(zodi_data, mode)
            if zodi2.shape != img2.shape:
                img2 = collapse_cube(image_data, "mid")
                zodi2 = collapse_cube(zodi_data, "mid")
            cleaned_full = img2 - zodi2

        cleaned = collapse_cube(cleaned_full, mode) if cleaned_full.ndim == 3 else cleaned_full
        bad = smart_flag_mask(flags_data, mode=mode)
        if bad is not None and bad.shape == cleaned.shape:
            cleaned = cleaned.copy()
            cleaned[bad] = np.nan

        wcs = WCS(image_header, naxis=2)
        meta = {"n_bands": n_bands, "collapse_mode": mode}

    return cleaned, wcs, image_header, meta


# ---------------------------------------------------------------------------
# 3. Subtract two epochs (epoch2 − epoch1) after reprojection
# ---------------------------------------------------------------------------

def subtract_epochs(url_epoch1, url_epoch2, collapse_mode="mid"):
    """
    Load two SPHEREx FITS cutouts, reproject Epoch 2 onto Epoch 1's WCS grid,
    and compute difference = epoch2_aligned − epoch1.

    Returns
    -------
    epoch1 : numpy.ndarray
    epoch2_aligned : numpy.ndarray
    difference : numpy.ndarray
    wcs1 : astropy.wcs.WCS
    meta : dict
    """
    epoch1, wcs1, header1, meta1 = load_and_clean_cutout(url_epoch1, collapse_mode=collapse_mode)
    epoch2, wcs2, header2, meta2 = load_and_clean_cutout(url_epoch2, collapse_mode=collapse_mode)

    # Reproject Epoch 2 onto Epoch 1's celestial grid
    epoch2_aligned, footprint = reproject_interp(
        (epoch2, wcs2),
        wcs1,
        shape_out=epoch1.shape,
    )

    epoch2_aligned[footprint == 0] = np.nan
    difference = epoch2_aligned - epoch1

    meta = {
        "collapse_mode": collapse_mode,
        "n_bands_epoch1": meta1.get("n_bands", 1),
        "n_bands_epoch2": meta2.get("n_bands", 1),
    }
    return epoch1, epoch2_aligned, difference, wcs1, meta


# ---------------------------------------------------------------------------
# 4. Anomaly detection via sigma-clipped peak/dipole analysis
# ---------------------------------------------------------------------------

def detect_anomalies(difference, wcs, sigma_thresh=3.0):
    """
    Detect high-SNR positive and negative peaks (dipoles) in the difference array.
    """
    valid_mask = np.isfinite(difference)
    if not np.any(valid_mask):
        return [], {"mean": 0.0, "median": 0.0, "std": 1.0}

    mean, median, std = sigma_clipped_stats(difference, sigma=3.0, maxiters=5)
    if std == 0 or np.isnan(std):
        std = np.nanstd(difference) or 1.0

    snr_array = (difference - median) / std

    diff_filled_pos = np.where(valid_mask, difference, -np.inf)
    diff_filled_neg = np.where(valid_mask, difference, np.inf)

    is_local_max = (diff_filled_pos == maximum_filter(diff_filled_pos, size=3, mode='constant', cval=-np.inf))
    is_local_min = (diff_filled_neg == minimum_filter(diff_filled_neg, size=3, mode='constant', cval=np.inf))

    pos_mask = valid_mask & (snr_array >= sigma_thresh) & is_local_max
    neg_mask = valid_mask & (snr_array <= -sigma_thresh) & is_local_min

    anomalies = []

    # Positive dipoles (New / Brightened in Epoch 2)
    for y, x in zip(*np.where(pos_mask)):
        coord = wcs.pixel_to_world(x, y)
        anomalies.append({
            "pixel_x": int(x),
            "pixel_y": int(y),
            "ra": round(float(coord.ra.deg), 6),
            "dec": round(float(coord.dec.deg), 6),
            "snr": round(float(snr_array[y, x]), 2),
            "flux_diff": round(float(difference[y, x]), 5),
            "type": "Positive Dipole (New / Brightened)"
        })

    # Negative dipoles (Faded / Moved Away from Epoch 1 position)
    for y, x in zip(*np.where(neg_mask)):
        coord = wcs.pixel_to_world(x, y)
        anomalies.append({
            "pixel_x": int(x),
            "pixel_y": int(y),
            "ra": round(float(coord.ra.deg), 6),
            "dec": round(float(coord.dec.deg), 6),
            "snr": round(float(snr_array[y, x]), 2),
            "flux_diff": round(float(difference[y, x]), 5),
            "type": "Negative Dipole (Faded / Moved Away)"
        })

    anomalies.sort(key=lambda a: abs(a["snr"]), reverse=True)

    stats = {
        "mean": round(float(mean), 6),
        "median": round(float(median), 6),
        "std": round(float(std), 6),
        "sigma_threshold": float(sigma_thresh)
    }

    return anomalies, stats


# ---------------------------------------------------------------------------
# 5. Image rendering helpers (base64 PNG output)
# ---------------------------------------------------------------------------

def _fig_to_base64(fig, dpi=160):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#09090b", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def render_single_panel(data, wcs, title, cmap="viridis", symmetric=False, label="MJy/sr", anomalies=None):
    fig = plt.figure(figsize=(7, 6.2), facecolor="#09090b")
    ax = fig.add_subplot(1, 1, 1, projection=wcs, facecolor="#000000")

    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        vmin, vmax = 0, 1
    elif symmetric:
        v1 = np.nanpercentile(valid, 1)
        v2 = np.nanpercentile(valid, 99)
        abs_val = max(abs(v1), abs(v2)) or 1e-4
        vmin, vmax = -abs_val, abs_val
    else:
        vmin = np.nanpercentile(valid, 1)
        vmax = np.nanpercentile(valid, 99)
        if vmin == vmax:
            vmax = vmin + 1e-4

    im = ax.imshow(data, origin="lower", cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))

    if anomalies:
        for anom in anomalies[:8]:
            x, y = anom["pixel_x"], anom["pixel_y"]
            color = "#10b981" if "Positive" in anom["type"] else "#f43f5e"
            circ = Circle((x, y), radius=0.85, fill=False, edgecolor=color, linewidth=1.5, linestyle="-")
            ax.add_patch(circ)

    ax.set_title(title, color="#fafafa", fontsize=11, fontweight="600", pad=12)
    ax.tick_params(colors="#71717a", labelsize=8)
    ax.coords[0].set_axislabel("RA (ICRS)", color="#a1a1aa", fontsize=9)
    ax.coords[1].set_axislabel("Dec (ICRS)", color="#a1a1aa", fontsize=9)
    ax.coords.grid(color="#27272a", alpha=0.6, linestyle=":")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#71717a", labelsize=8)
    cbar.set_label(label, color="#a1a1aa", fontsize=9)

    fig.tight_layout()
    return _fig_to_base64(fig)


def render_combined_panel(epoch1, epoch2_aligned, difference, wcs1, title="SPHEREx Epoch Subtraction", anomalies=None):
    fig = plt.figure(figsize=(18, 5.8), facecolor="#09090b")

    # Panel 1: Epoch 1
    ax1 = fig.add_subplot(1, 3, 1, projection=wcs1, facecolor="#000000")
    v1_min = np.nanpercentile(epoch1[np.isfinite(epoch1)], 1)
    v1_max = np.nanpercentile(epoch1[np.isfinite(epoch1)], 99)
    im1 = ax1.imshow(epoch1, origin="lower", cmap="viridis", norm=Normalize(vmin=v1_min, vmax=v1_max))
    ax1.set_title("Epoch 1 (Baseline)", color="#e4e4e7", fontsize=11, fontweight="600", pad=10)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors="#71717a", labelsize=8)
    cbar1.set_label("MJy/sr", color="#a1a1aa", fontsize=8)

    # Panel 2: Epoch 2
    ax2 = fig.add_subplot(1, 3, 2, projection=wcs1, facecolor="#000000")
    v2_min = np.nanpercentile(epoch2_aligned[np.isfinite(epoch2_aligned)], 1)
    v2_max = np.nanpercentile(epoch2_aligned[np.isfinite(epoch2_aligned)], 99)
    im2 = ax2.imshow(epoch2_aligned, origin="lower", cmap="viridis", norm=Normalize(vmin=v2_min, vmax=v2_max))
    ax2.set_title("Epoch 2 (WCS Aligned)", color="#e4e4e7", fontsize=11, fontweight="600", pad=10)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors="#71717a", labelsize=8)
    cbar2.set_label("MJy/sr", color="#a1a1aa", fontsize=8)

    # Panel 3: Difference
    ax3 = fig.add_subplot(1, 3, 3, projection=wcs1, facecolor="#000000")
    diff_valid = difference[np.isfinite(difference)]
    if len(diff_valid) > 0:
        d_min = np.nanpercentile(diff_valid, 1)
        d_max = np.nanpercentile(diff_valid, 99)
        abs_max = max(abs(d_min), abs(d_max)) or 1e-4
    else:
        abs_max = 1.0
    im3 = ax3.imshow(difference, origin="lower", cmap="RdBu_r", norm=Normalize(vmin=-abs_max, vmax=abs_max))
    ax3.set_title("Difference (Epoch 2 − Epoch 1)", color="#e4e4e7", fontsize=11, fontweight="600", pad=10)
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(colors="#71717a", labelsize=8)
    cbar3.set_label("ΔMJy/sr", color="#a1a1aa", fontsize=8)

    if anomalies:
        for anom in anomalies[:8]:
            x, y = anom["pixel_x"], anom["pixel_y"]
            color = "#10b981" if "Positive" in anom["type"] else "#f43f5e"
            circ = Circle((x, y), radius=0.9, fill=False, edgecolor=color, linewidth=1.5, linestyle="-")
            ax3.add_patch(circ)
            ax3.text(x + 1.2, y + 0.3, f"{anom['snr']:+.1f}σ", color=color, fontsize=8, fontweight="600")

    for ax in (ax1, ax2, ax3):
        ax.tick_params(colors="#71717a", labelsize=8)
        ax.coords[0].set_axislabel("RA (ICRS)", color="#a1a1aa", fontsize=8)
        ax.coords[1].set_axislabel("Dec (ICRS)", color="#a1a1aa", fontsize=8)
        ax.coords.grid(color="#27272a", alpha=0.5, linestyle=":")

    fig.suptitle(title, color="#fafafa", fontsize=12, fontweight="600", y=0.98)
    fig.tight_layout()
    return _fig_to_base64(fig)


def render_images_base64(epoch1, epoch2_aligned, difference, wcs1, anomalies=None):
    epoch1_png = render_single_panel(epoch1, wcs1, "Epoch 1 (Baseline)", cmap="viridis", label="MJy/sr")
    epoch2_png = render_single_panel(epoch2_aligned, wcs1, "Epoch 2 (Aligned)", cmap="viridis", label="MJy/sr")
    difference_png = render_single_panel(
        difference, wcs1, "Difference (Epoch 2 − Epoch 1)",
        cmap="RdBu_r", symmetric=True, label="ΔMJy/sr", anomalies=anomalies
    )
    combined_png = render_combined_panel(epoch1, epoch2_aligned, difference, wcs1, anomalies=anomalies)

    return {
        "epoch1_png": epoch1_png,
        "epoch2_png": epoch2_png,
        "difference_png": difference_png,
        "combined_png": combined_png
    }
