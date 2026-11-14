"""
test_pipeline.py — End-to-end test of the SPHEREx epoch-subtraction pipeline.

Uses WISE 0855−0714 (WISE J085510.83−071442.5) as the target:
    RA  = 133.79513°
    Dec =  −7.24514°

Steps:
    1. Query IRSA TAP for all SPHEREx-D2 observations of WISE 0855.
    2. Pick the oldest (first) and newest (last) observations.
    3. Resolve DataLink URLs → direct FITS URLs.
    4. Run the subtraction pipeline.
    5. Save a 3-panel diagnostic PNG.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from pipeline import (
    query_spherex_cutouts,
    resolve_datalink_to_cutout_url,
    subtract_epochs,
)

# ── Target: WISE 0855−0714 ──────────────────────────────────────────────────
RA = 133.79513
DEC = -7.24514
TARGET_NAME = "WISE_0855"

# ── Step 1: Query ────────────────────────────────────────────────────────────
print(f"[1/5] Querying IRSA TAP for SPHEREx-D2 cutouts near "
      f"(RA={RA}, Dec={DEC}) …")
table = query_spherex_cutouts(RA, DEC, size=0.05, bandpass="SPHEREx-D2")
print(f"       → {len(table)} observations returned.")

if len(table) < 2:
    print("ERROR: Need at least 2 observations for epoch subtraction. "
          "Try a wider search or different bandpass.")
    sys.exit(1)

# Show a summary
cols_to_show = [c for c in ("obs_id", "t_min", "t_max", "access_url")
                if c in table.colnames]
print(table[cols_to_show])

# ── Step 2: Pick oldest & newest ─────────────────────────────────────────────
datalink_epoch1 = str(table["access_url"][0])     # oldest
datalink_epoch2 = str(table["access_url"][-1])    # newest
print(f"\n[2/5] Epoch 1 (oldest) DataLink: {datalink_epoch1}")
print(f"       Epoch 2 (newest) DataLink: {datalink_epoch2}")

# ── Step 3: Resolve DataLink → direct FITS URLs ─────────────────────────────
print("\n[3/5] Resolving DataLink URLs to direct FITS URLs …")
fits_url1 = resolve_datalink_to_cutout_url(datalink_epoch1, RA, DEC, size=0.05)
fits_url2 = resolve_datalink_to_cutout_url(datalink_epoch2, RA, DEC, size=0.05)
print(f"       Epoch 1 FITS: {fits_url1}")
print(f"       Epoch 2 FITS: {fits_url2}")

# ── Step 4: Subtract ────────────────────────────────────────────────────────
print("\n[4/5] Loading, cleaning, reprojecting, and subtracting …")
epoch1, epoch2_aligned, difference, wcs1, _meta = subtract_epochs(
    fits_url1, fits_url2
)
print("       Done.")

# ── Step 5: Diagnostic PNG ──────────────────────────────────────────────────
print("\n[5/5] Saving diagnostic PNG …")

# Use a symmetric colour scale for the difference panel
vmin_diff = np.nanpercentile(difference, 1)
vmax_diff = np.nanpercentile(difference, 99)
abs_max = max(abs(vmin_diff), abs(vmax_diff))

fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                         subplot_kw={"projection": wcs1})

# Panel 1 — Epoch 1
im1 = axes[0].imshow(epoch1, origin="lower", cmap="viridis",
                     norm=Normalize(
                         vmin=np.nanpercentile(epoch1, 1),
                         vmax=np.nanpercentile(epoch1, 99)))
axes[0].set_title("Epoch 1 (oldest)")
fig.colorbar(im1, ax=axes[0], shrink=0.8, label="MJy/sr")

# Panel 2 — Epoch 2 (aligned)
im2 = axes[1].imshow(epoch2_aligned, origin="lower", cmap="viridis",
                     norm=Normalize(
                         vmin=np.nanpercentile(epoch2_aligned, 1),
                         vmax=np.nanpercentile(epoch2_aligned, 99)))
axes[1].set_title("Epoch 2 (newest, aligned)")
fig.colorbar(im2, ax=axes[1], shrink=0.8, label="MJy/sr")

# Panel 3 — Difference (symmetric scale)
im3 = axes[2].imshow(difference, origin="lower", cmap="RdBu_r",
                     norm=Normalize(vmin=-abs_max, vmax=abs_max))
axes[2].set_title("Difference (Epoch 2 − Epoch 1)")
fig.colorbar(im3, ax=axes[2], shrink=0.8, label="ΔMJy/sr")

for ax in axes:
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")

fig.suptitle(f"{TARGET_NAME}  |  SPHEREx-D2 Epoch Subtraction", fontsize=14)
fig.tight_layout()

outfile = f"{TARGET_NAME}_epoch_subtraction.png"
fig.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"       Saved → {outfile}")
print("\n✅ test_pipeline.py completed successfully.")
