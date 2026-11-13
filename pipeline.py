"""
pipeline.py — SPHEREx epoch-subtraction and anomaly-detection pipeline.

Features:
- Query IRSA TAP for SPHEREx observations
- DataLink resolver for fast spatial cutouts
- Multi-Extension FITS loader with ZODI background subtraction and proper flag filtering:
  preserves real astronomical sources (bit 21 = 2097152) and benign flag 2.
- WCS reprojection and epoch difference calculation
- Statistical anomaly detection (sigma-clipped local peak/dipole identification)
- High-quality astronomical image rendering to base64 PNGs
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import io
import base64
import numpy as np
import requests
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
    Resolve a DataLink access_url to a direct FITS cutout URL.

    The IRSA DataLink response embeds a cutout service descriptor.
    We build the cutout URL from the service's accessURL base,
    the file path, and the requested (ra, dec, size).

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
    resp = requests.get(datalink_url, timeout=60)
    resp.raise_for_status()

    votable = parse_votable(io.BytesIO(resp.content))

    # The results table is in the second resource (type="results")
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

    # Prefer the cutout service URL (much smaller download than 65MB full frame)
    cutout_mask = results_table[sem_col] == "#cutout"
    if cutout_mask.any():
        path = str(results_table[path_col][cutout_mask][0])
        cutout_base = "https://irsa.ipac.caltech.edu/ibe/cutout"
        cutout_url = (
            f"{cutout_base}?path={path}"
            f"&ra={ra}&dec={dec}&size={size}"
        )
        return cutout_url

    # Fallback: use the #this row (full science image direct URL)
    this_mask = results_table[sem_col] == "#this"
    if this_mask.any():
        direct_url = str(results_table[url_col][this_mask][0])
        return direct_url

    raise ValueError(
        f"Could not resolve DataLink URL to a FITS download: {datalink_url}"
    )


# ---------------------------------------------------------------------------
# 2. Load & clean a single SPHEREx cutout
# ---------------------------------------------------------------------------

def load_and_clean_cutout(fits_url):
    """
    Download a SPHEREx MEF cutout/image, subtract the zodiacal background,
    and mask bad glitch pixels.

    In the SPHEREx FLAGS HDU:
    - Bit 21 (2097152) = astronomical SOURCE pixels (must NOT be masked)
    - Bit 1 (2) = benign calibration/warning flag (must NOT be masked)
    Only bad glitch pixels where (flags & ~(2097152 | 2)) != 0 are masked with NaN.

    Parameters
    ----------
    fits_url : str
        Direct URL or local filepath to the FITS file.

    Returns
    -------
    cleaned : numpy.ndarray (2-D, float64)
        IMAGE − ZODI with bad glitch pixels set to NaN.
    wcs : astropy.wcs.WCS
        2-D celestial WCS parsed from the IMAGE header.
    header : astropy.io.fits.Header
        The IMAGE extension header.
    """
    with fits.open(fits_url, memmap=False) as hdulist:
        image_data = hdulist["IMAGE"].data.astype(np.float64)
        image_header = hdulist["IMAGE"].header

        if image_data.ndim == 3:
            mid = image_data.shape[0] // 2
            image_data = image_data[mid]

        zodi_data = hdulist["ZODI"].data.astype(np.float64)
        if zodi_data.ndim == 3:
            mid = zodi_data.shape[0] // 2
            zodi_data = zodi_data[mid]

        flags_data = hdulist["FLAGS"].data
        if flags_data.ndim == 3:
            mid = flags_data.shape[0] // 2
            flags_data = flags_data[mid]

        wcs = WCS(image_header, naxis=2)

        # Subtract zodiacal background
        cleaned = image_data - zodi_data

        # Mask ONLY true glitch pixels:
        # bit 21 (2097152) = SOURCE, bit 1 (2) = benign
        EXCLUDE_MASK = 2097152 | 2  # 2097154
        bad = (flags_data.astype(np.int64) & ~EXCLUDE_MASK) != 0
        cleaned[bad] = np.nan

    return cleaned, wcs, image_header


# ---------------------------------------------------------------------------
# 3. Subtract two epochs (epoch2 − epoch1) after reprojection
# ---------------------------------------------------------------------------

def subtract_epochs(url_epoch1, url_epoch2):
    """
    Load two SPHEREx FITS cutouts, reproject Epoch 2 onto Epoch 1's WCS grid,
    and compute difference = epoch2_aligned − epoch1.

    Returns
    -------
    epoch1 : numpy.ndarray
    epoch2_aligned : numpy.ndarray
    difference : numpy.ndarray
    wcs1 : astropy.wcs.WCS
    """
    epoch1, wcs1, header1 = load_and_clean_cutout(url_epoch1)
    epoch2, wcs2, header2 = load_and_clean_cutout(url_epoch2)

    # Reproject Epoch 2 onto Epoch 1's celestial grid
    epoch2_aligned, footprint = reproject_interp(
        (epoch2, wcs2),
        wcs1,
        shape_out=epoch1.shape,
    )

    epoch2_aligned[footprint == 0] = np.nan
    difference = epoch2_aligned - epoch1

    return epoch1, epoch2_aligned, difference, wcs1


# ---------------------------------------------------------------------------
# 4. Anomaly detection via sigma-clipped peak/dipole analysis
# ---------------------------------------------------------------------------

def detect_anomalies(difference, wcs, sigma_thresh=3.0):
    """
    Detect high-SNR positive and negative peaks (dipoles) in the difference array.

    Parameters
    ----------
    difference : 2D numpy.ndarray
        The subtracted array (epoch2_aligned - epoch1).
    wcs : astropy.wcs.WCS
        2D celestial WCS for converting pixel coords to sky (RA, Dec).
    sigma_thresh : float
        Statistical significance cutoff in standard deviations.

    Returns
    -------
    anomalies : list of dict
        Detected candidates sorted by absolute SNR descending.
    stats : dict
        Sigma-clipped mean, median, and std of the difference array.
    """
    # Filter out NaNs for robust statistics
    valid_mask = np.isfinite(difference)
    if not np.any(valid_mask):
        return [], {"mean": 0.0, "median": 0.0, "std": 1.0}

    mean, median, std = sigma_clipped_stats(difference, sigma=3.0, maxiters=5)
    if std == 0 or np.isnan(std):
        std = np.nanstd(difference) or 1.0

    snr_array = (difference - median) / std

    # Local maximum filter for positive peaks
    # Replace NaNs with -inf for max filter and +inf for min filter
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

def _fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#090d16", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def render_single_panel(data, wcs, title, cmap="viridis", symmetric=False, label="MJy/sr", anomalies=None):
    """
    Render a single astronomical panel with celestial WCS axes and dark styling.
    """
    fig = plt.figure(figsize=(7, 6), facecolor="#090d16")
    ax = fig.add_subplot(1, 1, 1, projection=wcs, facecolor="#040711")

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

    # Mark anomalies if requested
    if anomalies:
        for anom in anomalies[:10]:  # Top 10
            x, y = anom["pixel_x"], anom["pixel_y"]
            color = "#00f0ff" if "Positive" in anom["type"] else "#ff2a6d"
            circ = Circle((x, y), radius=0.8, fill=False, edgecolor=color, linewidth=1.8, linestyle="--")
            ax.add_patch(circ)

    ax.set_title(title, color="#e2e8f0", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.coords[0].set_axislabel("RA (ICRS)", color="#cbd5e1", fontsize=10)
    ax.coords[1].set_axislabel("Dec (ICRS)", color="#cbd5e1", fontsize=10)
    ax.coords.grid(color="#334155", alpha=0.4, linestyle=":")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.set_label(label, color="#cbd5e1", fontsize=10)

    fig.tight_layout()
    return _fig_to_base64(fig)


def render_combined_panel(epoch1, epoch2_aligned, difference, wcs1, title="SPHEREx Epoch Subtraction", anomalies=None):
    """
    Render a 3-panel comparison figure (Epoch 1, Epoch 2 Aligned, Difference) with WCS.
    """
    fig = plt.figure(figsize=(19, 6.2), facecolor="#090d16")

    # Panel 1: Epoch 1
    ax1 = fig.add_subplot(1, 3, 1, projection=wcs1, facecolor="#040711")
    v1_min = np.nanpercentile(epoch1[np.isfinite(epoch1)], 1)
    v1_max = np.nanpercentile(epoch1[np.isfinite(epoch1)], 99)
    im1 = ax1.imshow(epoch1, origin="lower", cmap="viridis", norm=Normalize(vmin=v1_min, vmax=v1_max))
    ax1.set_title("Epoch 1 (Baseline)", color="#38bdf8", fontsize=12, fontweight="bold", pad=10)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar1.set_label("MJy/sr", color="#cbd5e1", fontsize=9)

    # Panel 2: Epoch 2
    ax2 = fig.add_subplot(1, 3, 2, projection=wcs1, facecolor="#040711")
    v2_min = np.nanpercentile(epoch2_aligned[np.isfinite(epoch2_aligned)], 1)
    v2_max = np.nanpercentile(epoch2_aligned[np.isfinite(epoch2_aligned)], 99)
    im2 = ax2.imshow(epoch2_aligned, origin="lower", cmap="viridis", norm=Normalize(vmin=v2_min, vmax=v2_max))
    ax2.set_title("Epoch 2 (Aligned)", color="#38bdf8", fontsize=12, fontweight="bold", pad=10)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar2.set_label("MJy/sr", color="#cbd5e1", fontsize=9)

    # Panel 3: Difference
    ax3 = fig.add_subplot(1, 3, 3, projection=wcs1, facecolor="#040711")
    diff_valid = difference[np.isfinite(difference)]
    if len(diff_valid) > 0:
        d_min = np.nanpercentile(diff_valid, 1)
        d_max = np.nanpercentile(diff_valid, 99)
        abs_max = max(abs(d_min), abs(d_max)) or 1e-4
    else:
        abs_max = 1.0
    im3 = ax3.imshow(difference, origin="lower", cmap="RdBu_r", norm=Normalize(vmin=-abs_max, vmax=abs_max))
    ax3.set_title("Difference (Epoch 2 − Epoch 1)", color="#f43f5e", fontsize=12, fontweight="bold", pad=10)
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar3.set_label("ΔMJy/sr", color="#cbd5e1", fontsize=9)

    # Annotate anomalies on difference panel
    if anomalies:
        for anom in anomalies[:8]:
            x, y = anom["pixel_x"], anom["pixel_y"]
            color = "#38bdf8" if "Positive" in anom["type"] else "#f43f5e"
            circ = Circle((x, y), radius=0.9, fill=False, edgecolor=color, linewidth=1.8, linestyle="--")
            ax3.add_patch(circ)
            ax3.text(x + 1.2, y + 0.3, f"{anom['snr']:+.1f}σ", color=color, fontsize=8, fontweight="bold")

    for ax in (ax1, ax2, ax3):
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.coords[0].set_axislabel("RA (ICRS)", color="#cbd5e1", fontsize=9)
        ax.coords[1].set_axislabel("Dec (ICRS)", color="#cbd5e1", fontsize=9)
        ax.coords.grid(color="#334155", alpha=0.35, linestyle=":")

    fig.suptitle(title, color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    return _fig_to_base64(fig)


def render_images_base64(epoch1, epoch2_aligned, difference, wcs1, anomalies=None):
    """
    Generate base64 PNG data URLs for all views.
    """
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
