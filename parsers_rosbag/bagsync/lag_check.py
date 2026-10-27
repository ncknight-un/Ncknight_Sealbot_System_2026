#!/usr/bin/env python3
"""
lag_check.py - estimate the timing offset between the tags and the IMU from a tag_to_imu_format CSV.

Flexible tags bend when the vehicle accelerates, so each tag's position follows the IMU's
acceleration, but through its own bending: a soft part responds later and wobbles at its own
natural frequency. Every tag comes from the same camera image, so the TIMING offset is the same
for all of them; only the bending differs. This script separates the two.

Method
  1. For each tag, pick the tag axis (x/y/z) and IMU acceleration axis that move together most
     (highest coherence, 0.3-12 Hz).
  2. Measure, frequency by frequency, how the tag responds to the acceleration (cross-spectrum,
     Welch averaging, ~4 s Hann segments).
  3. Fit one model to all tags at once:
         tag response(f) = gain_k x bending_k(f) x exp(-i 2 pi f tau)
     bending_k = a spring-mass-damper (natural frequency fn_k, damping zeta_k), different per tag;
     tau = the timing offset, the SAME for every tag. A pure time shift delays every frequency by
     the same time; bending delays them differently (it swings through 180 deg around fn_k), so
     tags that wobble in the measured band pin tau down. Each frequency is weighted by how well
     the two signals agree there (coherence).
  4. Uncertainty: the fit is repeated leaving out each fifth of the recording in turn (jackknife).
  5. Checks before trusting it: the model's residual against the noise level (about 1-2 when it
     fits), whether the tags' fitted bending ran into the model's limits, and whether enough tags
     wobble inside the band. A per-band table shows each tag's shift at 0.3-1, 1-2, 2-4 and 4-8 Hz:
     a pure timing offset is the same in every band, so big differences mean bending dominates.
     When the model doesn't fit, no offset is reported.

  Tested on simulated recordings (38 s, 16 tags, noise, known offsets of -20, 0 and +12 ms):
  within about 1-2 ms when 3 or more tags wobble in the band; 4-5 ms too small with none
  (flagged CAUTION).

    tau > 0 : the tags are stamped LATER than the IMU for the same moment (tags after IMU)
    tau < 0 : the tags are stamped earlier

The simple shift-matching estimate (cross-correlation peak, bending included) is shown per tag
for comparison; it equals tau + that tag's bending delay.

Usage:  python lag_check.py export.csv [--band 0.3 12] [--skip-tags 0,20] [--segment 4]
"""
import argparse
import csv
import re
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def read_csv(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = next(r)
        rows = [[float(v) if v != "" else np.nan for v in row] for row in r]
    arr = np.array(rows, float)
    return {h: arr[:, j] for j, h in enumerate(head)}


def fill(x):
    """Blank cells -> straight line between neighbours (ends held)."""
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return np.zeros_like(x)
    i = np.arange(len(x))
    x[~ok] = np.interp(i[~ok], i[ok], x[ok])
    return x


def find_columns(d, skip):
    tags = {}
    for c in d:
        m = re.search(r"tag_(\d+)_zero->tag_\d+\]\.translation_mm\.([xyz])$", c)
        if m and int(m.group(1)) not in skip:
            tags.setdefault(int(m.group(1)), {})[m.group(2)] = c
    imu = {}
    for c in d:
        m = re.search(r"imu(?:/data)?\.linear_acceleration\.([xyz])$", c)
        if m:
            imu[m.group(1)] = c
    return tags, imu


# --------------------------------------------------------------------------- #
# Spectra (Welch, kept per segment so blocks can be left out for the jackknife)
# --------------------------------------------------------------------------- #
def segment_ffts(x, nper):
    """rfft of each 50 %-overlapping, linearly detrended, Hann-windowed segment: (nseg, nf), starts."""
    x = np.asarray(x, float)
    step = nper // 2
    starts = np.arange(0, len(x) - nper + 1, step)
    win = np.hanning(nper)
    t = np.arange(nper)
    out = []
    for s in starts:
        seg = x[s:s + nper]
        p = np.polyfit(t, seg, 1)
        out.append(np.fft.rfft((seg - np.polyval(p, t)) * win))
    return np.array(out), starts


def cross(X, Y, use):
    """Averaged spectra over the segments in `use`: H = Sxy/Sxx (Y's response to X), coherence."""
    Sxx = np.mean(np.abs(X[use]) ** 2, 0)
    Syy = np.mean(np.abs(Y[use]) ** 2, 0)
    Sxy = np.mean(np.conj(X[use]) * Y[use], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        H = Sxy / Sxx
        coh = np.abs(Sxy) ** 2 / (Sxx * Syy)
    return H, np.nan_to_num(coh)


# --------------------------------------------------------------------------- #
# Model: gain x spring-mass-damper x pure delay; shared delay, per-tag bending
# --------------------------------------------------------------------------- #
FN_GRID = np.geomspace(0.5, 30.0, 70)        # Hz
PHASE_BANDS = {"0.3-1 Hz": (0.3, 1.0), "1-2 Hz": (1.0, 2.0), "2-4 Hz": (2.0, 4.0), "4-8 Hz": (4.0, 8.0)}
MISFIT_OK = 3.0               # model residual / noise level above which the fit is not trusted
ZETA_GRID = np.geomspace(0.03, 2.0, 45)


def bending_shapes(f):
    fn, z = np.meshgrid(FN_GRID, ZETA_GRID, indexing="ij")
    fn, z = fn.ravel(), z.ravel()
    r = f[None, :] / fn[:, None]
    return 1.0 / (1.0 - r ** 2 + 2j * z[:, None] * r), fn, z


def tag_costs(H, w, f, M, taus):
    """For one tag: cost(tau) minimised over bending shape and real gain, plus the best shape
    index and gain per tau. cost = sum_f w |H - G M e^{-i2pi f tau}|^2 / |H|^2 (relative error)."""
    u = w / np.maximum(np.abs(H) ** 2, 1e-30)
    C = np.sum(u * np.abs(H) ** 2)
    A = np.abs(M) ** 2 @ u                                       # (nshape,)
    E = np.exp(1j * 2 * np.pi * np.outer(f, taus))               # conj of the delay factor
    B = np.real(np.conj(M) @ (u[:, None] * H[:, None] * E))      # (nshape, ntau)
    red = B ** 2 / A[:, None]
    best = np.argmax(red, 0)
    cost = C - red[best, np.arange(len(taus))]
    gain = B[best, np.arange(len(taus))] / A[best]
    return cost, best, gain


def fit_tau(pairs, use, f_mask, f, M, taus):
    """Shared tau over all tag pairs, using the segments in `use`."""
    total = np.zeros(len(taus))
    per = []
    nbins = 0
    for X, Y in pairs:
        H, coh = cross(X, Y, use)
        H, coh = H[f_mask], coh[f_mask]
        n = use.sum()
        w = np.where(coh > 0.1, 2 * n * coh / np.maximum(1 - coh, 1e-3), 0.0)
        cost, best, gain = tag_costs(H, w, f, M, taus)
        total += cost
        nbins += int((w > 0).sum())
        per.append((cost, best, gain, coh, H, w))
    fit_tau.nbins = max(nbins, 1)
    return total, per


def xcorr_lag(a, b, maxlag):
    """Simple shift-matching: lag (frames) where b(t - lag) best matches a(t); + = a after b."""
    n = len(a)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    lags = np.arange(-maxlag, maxlag + 1)
    c = np.array([np.mean(a[max(0, L):n - max(0, -L)] * b[max(0, -L):n - max(0, L)]) for L in lags])
    j = int(np.argmax(np.abs(c)))
    sub = 0.0
    if 0 < j < len(c) - 1:
        y0, y1, y2 = np.abs(c[j - 1:j + 2])
        den = y0 - 2 * y1 + y2
        sub = 0.5 * (y0 - y2) / den if den else 0.0
    return lags[j] + sub


def estimate(d, band=(0.3, 12.0), skip=(0, 20), segment_s=4.0, min_coh=0.25, blocks=5, verbose=True):
    dt = float(np.nanmedian(d["dt_s"])) if "dt_s" in d else 1 / 65.0
    fs = 1.0 / dt
    tags, imu = find_columns(d, set(skip))
    if not tags or not imu:
        raise ValueError("need tag deviation columns and IMU linear_acceleration columns (tag_to_imu_format)")
    nper = int(2 ** np.round(np.log2(segment_s * fs)))
    f_all = np.fft.rfftfreq(nper, dt)
    f_mask = (f_all >= band[0]) & (f_all <= band[1])
    f = f_all[f_mask]
    M, fn_s, z_s = bending_shapes(f)

    imu_ffts = {a: segment_ffts(fill(d[c]), nper) for a, c in imu.items()}
    starts = next(iter(imu_ffts.values()))[1]
    alluse = np.ones(len(starts), bool)

    # 1. the best-matching axes per tag
    chosen = []
    for k in sorted(tags):
        best = None
        for ta, tc in tags[k].items():
            Y = segment_ffts(fill(d[tc]), nper)[0]
            for ia, (X, _) in imu_ffts.items():
                _, coh = cross(X, Y, alluse)
                score = float(np.mean(coh[f_mask]))
                if best is None or score > best[0]:
                    best = (score, ta, ia, X, Y, tc, imu[ia])
        chosen.append((k, *best))
    used = [c for c in chosen if c[1] >= min_coh]
    if not used:
        raise ValueError(f"no tag moves with the IMU strongly enough (mean coherence >= {min_coh}); "
                         "try a recording with more motion, or a tap test")

    # 2. shared tau: coarse grid, then fine around the minimum
    pairs = [(c[4], c[5]) for c in used]
    taus = np.arange(-0.150, 0.1501, 0.002)
    total, _ = fit_tau(pairs, alluse, f_mask, f, M, taus)
    t0 = taus[np.argmin(total)]
    taus = np.arange(t0 - 0.006, t0 + 0.0061, 0.0002)
    total, per = fit_tau(pairs, alluse, f_mask, f, M, taus)
    i_best = int(np.argmin(total))
    tau = float(taus[i_best])

    # 3. jackknife over time blocks
    edges = np.linspace(0, starts[-1] + nper, blocks + 1)
    jk = []
    for b in range(blocks):
        seg_end = starts + nper
        use = ~((seg_end > edges[b]) & (starts < edges[b + 1]))
        if use.sum() < 3:
            continue
        tj, _ = fit_tau(pairs, use, f_mask, f, M, np.arange(-0.150, 0.1501, 0.002))
        g = np.arange(-0.150, 0.1501, 0.002)[np.argmin(tj)]
        g_f = np.arange(g - 0.004, g + 0.0041, 0.0002)
        tj2, _ = fit_tau(pairs, use, f_mask, f, M, g_f)
        jk.append(float(g_f[np.argmin(tj2)]))
    jk = np.array(jk)
    se = float(np.sqrt((len(jk) - 1) / len(jk) * np.sum((jk - jk.mean()) ** 2))) if len(jk) > 1 else np.nan

    # 4. per-tag details at the fitted tau
    maxlag = int(round(0.3 * fs))
    rows = []
    misfit = float(total[i_best] / (2 * fit_tau.nbins))     # ~1 = fits within the noise
    for c, (cost, best, gain, coh, H, w) in zip(used, per):
        k, score, ta, ia, X, Y, tc, icol = c
        j = best[i_best]
        fn, z = fn_s[j], z_s[j]
        bend_ms = 2 * z / (2 * np.pi * fn) * 1e3          # bending delay at low frequency
        xl = xcorr_lag(_hp(fill(d[tc]), int(fs)), _hp(fill(d[icol]), int(fs)), maxlag) * dt * 1e3
        edge = fn <= FN_GRID[1] or fn >= FN_GRID[-2] or z >= ZETA_GRID[-2]
        ph = np.unwrap(np.angle(H * (1 if gain[i_best] > 0 else -1)))
        delay = {fb: float(-np.mean(ph[(f >= lo) & (f < hi)] / (2 * np.pi * f[(f >= lo) & (f < hi)])) * 1e3)
                 for fb, (lo, hi) in PHASE_BANDS.items() if np.any((f >= lo) & (f < hi))}
        rows.append({"tag": k, "axes": f"tag {ta} / acc {ia}", "coherence": score, "fn_hz": fn, "zeta": z,
                     "bend_ms": bend_ms, "sign": "+" if gain[i_best] > 0 else "-", "xcorr_ms": xl,
                     "edge": bool(edge), "band_delay_ms": delay})
    skipped = [(c[0], c[1]) for c in chosen if c[1] < min_coh]
    # tau is pinned down by tags whose wobble (natural frequency) sits inside the band and is
    # lightly damped; with none, a stiff tag's bending looks like extra delay or less, and in tests
    # the estimate then came out 2-5 ms too small
    wobbly = [x["tag"] for x in rows if band[0] * 1.2 <= x["fn_hz"] <= band[1] * 0.8 and x["zeta"] < 0.5]
    res = {"tau_ms": tau * 1e3, "se_ms": se * 1e3, "jackknife_ms": (jk * 1e3).tolist(), "rows": rows,
           "skipped": skipped, "wobbly": wobbly, "misfit": misfit, "band": band, "segment_s": nper * dt, "dt_ms": dt * 1e3}
    if verbose:
        report(res)
    return res


def _hp(x, win):
    c = np.r_[0.0, np.cumsum(x)]
    i = np.arange(len(x))
    lo = np.clip(i - win // 2, 0, len(x))
    hi = np.clip(i + win - win // 2, 0, len(x))
    return x - (c[hi] - c[lo]) / (hi - lo)


def report(r):
    print(f"Band {r['band'][0]:g}-{r['band'][1]:g} Hz, {r['segment_s']:.1f} s segments, frame {r['dt_ms']:.1f} ms. "
          "Positive = tags stamped AFTER the IMU.\n")
    print(f"{'tag':>4} {'axes':>16} {'coherence':>9} {'natural freq':>12} {'damping':>8} {'bending delay':>13} "
          f"{'simple shift-match':>19}")
    for x in sorted(r["rows"], key=lambda x: x["tag"]):
        print(f"{x['tag']:>4} {x['axes']:>16} {x['coherence']:9.2f} {x['fn_hz']:9.1f} Hz {x['zeta']:8.2f} "
              f"{x['bend_ms']:10.1f} ms {x['xcorr_ms']:16.1f} ms")
    if r["skipped"]:
        print("left out (too little shared motion): " + ", ".join(f"tag {k} ({c:.2f})" for k, c in r["skipped"]))
    print("\nEach tag's shift by frequency band (ms, + = tag after IMU). A pure timing offset is the SAME in "
          "every band;\nbending changes it with frequency:")
    bands = list(PHASE_BANDS)
    print(f"{'tag':>4} " + " ".join(f"{b:>10}" for b in bands))
    for x in sorted(r["rows"], key=lambda x: x["tag"]):
        print(f"{x['tag']:>4} " + " ".join(f"{x['band_delay_ms'].get(b, float('nan')):10.1f}" for b in bands))
    bad_fit = r["misfit"] > MISFIT_OK or sum(x["edge"] for x in r["rows"]) > len(r["rows"]) / 2
    print(f"\nModel fit: residual {r['misfit']:.1f}x the noise level (about 1 = fits)"
          + (", and most tags' bending ran into the model's limits" if sum(x["edge"] for x in r["rows"]) > len(r["rows"]) / 2 else ""))
    if bad_fit:
        print("NO RELIABLE OFFSET: the tags don't respond like simple springs, so this recording can't separate "
              "timing from bending.\nLook at the band table: if the shift changes a lot between bands, it is "
              "mostly bending. Use a tap test, or a\nrecording with both sensors on ROS time, instead. "
              f"(The fit's value, for reference only: {r['tau_ms']:+.1f} ms.)")
        return
    print(f"\nTiming offset (shared by all tags): {r['tau_ms']:+.1f} ms  ± {r['se_ms']:.1f} ms (jackknife, "
          f"{len(r['jackknife_ms'])} blocks: " + ", ".join(f"{v:+.1f}" for v in r["jackknife_ms"]) + ")")
    print("Bending delay = each tag's own delay at slow motions, from its fitted spring model; "
          "simple shift-match ≈ offset + bending, for comparison.")
    if len(r["wobbly"]) >= 3:
        print(f"Well determined: {len(r['wobbly'])} tags wobble inside the band (tags "
              + ", ".join(map(str, r["wobbly"])) + "), which separates timing from bending.")
    else:
        print(f"CAUTION: only {len(r['wobbly'])} tag(s) wobble inside the band, so timing and bending are hard to "
              "separate. In tests the offset then came out 2-5 ms too small; treat it as a lower bound, or "
              "confirm with a tap test.")


def main():
    ap = argparse.ArgumentParser(description="Estimate the tag-to-IMU timing offset, separated from tag bending")
    ap.add_argument("csv")
    ap.add_argument("--band", nargs=2, type=float, default=(0.3, 12.0), metavar=("LO", "HI"),
                    help="frequency band used, Hz (default 0.3 12)")
    ap.add_argument("--skip-tags", default="0,20", help="stationary reference tags to leave out")
    ap.add_argument("--segment", type=float, default=4.0, help="spectral segment length, s (default 4)")
    ap.add_argument("--min-coh", type=float, default=0.25, help="leave out tags below this mean coherence")
    a = ap.parse_args()
    try:
        estimate(read_csv(a.csv), tuple(a.band), [int(x) for x in a.skip_tags.split(",") if x.strip()],
                 a.segment, a.min_coh)
    except ValueError as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
