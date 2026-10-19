"""
panels.py - turn the tag grid into a few physical "panel" signals and split each into the part
the IMU explains and the part it does not.

Layout (found automatically from where the tags sit in the image):
    the moving tags are split into a LEFT and a RIGHT panel (largest horizontal gap),
    inside a panel tags are grouped into columns (by image u) and rows (by image v);
    the OUTER column is the one farthest from the image centre, the INNER the nearest.
    Reference tags (REFERENCE_TAGS, fixed to the camera) are left out.

Signals, in micrometres, relative to their mean over Start -> Stop, image axes
(x = right in the image, y = down in the image), from each tag's pixel centre:
    <side>.shift_x / shift_y   average motion of all tags on that panel
    <side>.open                outer column minus inner column along x, signed so that
                               + = the outer edge moves AWAY from the image centre (panel opens)
    <side>.bend_y              outer column minus inner column along y
    both.open_sym              (left.open + right.open) / 2  - both panels open together
    both.open_asym             (left.open - right.open) / 2  - one opens while the other closes
    both.shift_x / shift_y     average of the two panels' shifts

For every signal three columns are written:
    <signal>            the measured signal
    <signal>_imu        the part predicted from the IMU (rotation rates, roll/pitch angles and
                        accelerations, each also delayed by 0.15, 0.3 and 0.6 s), fitted by
                        least squares over the window
    <signal>_leftover   measured minus predicted: motion the IMU cannot explain
The summary also reports how well the IMU predicts each signal on data it was NOT fitted on
(fit on one half of the window, test on the other, both ways), which is the honest number.
"""

from __future__ import annotations

import re

import numpy as np

from bagcache import BagData, Series

DERIVED_PANEL = "/derived/panels"
LAGS_S = (0.0, 0.15, 0.3, 0.6)
LOWPASS_HZ = 2.0
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
FOLDS = 5


def _interp_to(t_src, v, t_dst):
    good = np.isfinite(v)
    if good.sum() < 2:
        return np.full(len(t_dst), np.nan)
    out = np.interp(t_dst, t_src[good], v[good])
    out[(t_dst < t_src[good][0]) | (t_dst > t_src[good][-1])] = np.nan
    return out


def _fill(v):
    v = v.copy()
    good = np.isfinite(v)
    if good.sum() >= 2 and not good.all():
        idx = np.arange(len(v))
        v[~good] = np.interp(idx[~good], idx[good], v[good])
    return v


def _lowpass(x, fs, hz):
    from scipy import signal
    if fs <= 2.2 * hz:
        return x
    b, a = signal.butter(2, hz / (fs / 2))
    return signal.filtfilt(b, a, x)


def _layout(tags: dict, img_cx: float):
    """tags: {id: (u, v)} -> {'left': {...}, 'right': {...}} with columns/rows, or None."""
    ids = sorted(tags, key=lambda k: tags[k][0])
    if len(ids) < 4:
        return None
    us = np.array([tags[k][0] for k in ids])
    gap = int(np.argmax(np.diff(us)))
    sides = {"left": ids[:gap + 1], "right": ids[gap + 1:]}
    out = {}
    for side, members in sides.items():
        if len(members) < 2:
            continue
        u = np.array([tags[k][0] for k in members])
        v = np.array([tags[k][1] for k in members])
        def groups(vals, keys):
            """Split sorted values wherever the gap is more than half the largest gap (and > 25 px)."""
            order = np.argsort(vals)
            sv = vals[order]
            d = np.diff(sv)
            thr = max(25.0, 0.5 * d.max()) if len(d) else 1e9
            g, cur = [], [keys[order[0]]]
            for i in range(1, len(order)):
                if sv[i] - sv[i - 1] > thr:
                    g.append(cur)
                    cur = []
                cur.append(keys[order[i]])
            g.append(cur)
            return g

        cols = groups(u, members)
        rows = groups(v, members)
        centre_u = np.mean(u)
        outward = 1.0 if centre_u > img_cx else -1.0
        # outer = farthest from the image centre
        cols_sorted = sorted(cols, key=lambda c: abs(np.mean([tags[k][0] for k in c]) - img_cx))
        out[side] = {"tags": members, "columns": cols, "rows": rows, "inner": cols_sorted[0],
                     "outer": cols_sorted[-1], "outward": outward}
    return out if out else None


def panel_analysis(data: BagData, start_s: float, stop_s: float, reference_tags=(0, 20)) -> Series | None:
    """Build the panel signals for frames in [start_s, stop_s]. Returns a derived Series or None."""
    if data.n_frames < 10:
        return None
    t0, t1 = data.abs_ns(start_s), data.abs_ns(stop_s)
    m = (data.frame_t >= t0) & (data.frame_t <= t1)
    tf = data.frame_t[m]
    if len(tf) < 20:
        return None
    ts = (tf - tf[0]) / 1e9
    fs = 1.0 / np.median(np.diff(ts))

    ci = data.meta.get("camera_info") or {}
    P = np.array(ci.get("p", [0] * 12), float).reshape(3, 4)
    K = np.array(ci.get("k", [0] * 9), float).reshape(3, 3)
    fx = P[0, 0] if P[0, 0] > 0 else K[0, 0]
    img_cx = (P[0, 2] if P[0, 0] > 0 else K[0, 2]) or (ci.get("width", 0) / 2)
    units = "µm" if fx > 0 else "px"

    # tag pixel centres on the frame times
    tag_series = {s.info["tag_id"]: s for s in data.series.values()
                  if s.info.get("kind") == "tag" and "centre.x" in s.cols}
    tf_z = {}
    for s in data.series.values():
        if s.info.get("kind") == "tf" and "translation.z" in s.cols:
            mm = re.search(r"(\d+)\]?$", s.name)
            if mm:
                tf_z[int(mm.group(1))] = float(np.nanmedian(s.cols["translation.z"]))
    pos, xy = {}, {}
    for k, s in tag_series.items():
        if k in reference_tags:
            continue
        u = _interp_to(s.t, s.cols["centre.x"], tf)
        v = _interp_to(s.t, s.cols["centre.y"], tf)
        if np.isfinite(u).mean() < 0.8:
            continue
        u, v = _fill(u), _fill(v)
        scale = (tf_z.get(k, 0.2) / fx * 1e6) if fx > 0 else 1.0       # µm per pixel
        pos[k] = (float(np.mean(u)), float(np.mean(v)))
        xy[k] = ((u - u.mean()) * scale, (v - v.mean()) * scale)
    lay = _layout(pos, img_cx)
    if lay is None:
        return None

    cols, info = {}, {"kind": "derived", "units": units, "layout": {}, "window_t_s": [start_s, stop_s]}

    def mean_of(keys, axis):
        return np.mean([xy[k][axis] for k in keys], axis=0)

    for side, L in lay.items():
        cols[f"{side}.shift_x"] = mean_of(L["tags"], 0)
        cols[f"{side}.shift_y"] = mean_of(L["tags"], 1)
        if L["outer"] != L["inner"]:
            cols[f"{side}.open"] = L["outward"] * (mean_of(L["outer"], 0) - mean_of(L["inner"], 0))
            cols[f"{side}.bend_y"] = mean_of(L["outer"], 1) - mean_of(L["inner"], 1)
        info["layout"][side] = {"outer": L["outer"], "inner": L["inner"], "rows": L["rows"]}
    if "left" in lay and "right" in lay:
        if "left.open" in cols and "right.open" in cols:
            cols["both.open_sym"] = 0.5 * (cols["left.open"] + cols["right.open"])
            cols["both.open_asym"] = 0.5 * (cols["left.open"] - cols["right.open"])
        cols["both.shift_x"] = 0.5 * (cols["left.shift_x"] + cols["right.shift_x"])
        cols["both.shift_y"] = 0.5 * (cols["left.shift_y"] + cols["right.shift_y"])

    # IMU features on the same frame times, low-passed, with delayed copies
    imu = next((s for s in data.series.values() if s.name.endswith("/imu/data") and "angular_velocity.x" in s.cols), None)
    feats = []
    if imu is not None:
        names = [f"angular_velocity.{a}" for a in "xyz"] + ["orientation.roll_deg", "orientation.pitch_deg"] + \
                [f"linear_acceleration.{a}" for a in "xyz"]
        for nme in names:
            if nme not in imu.cols:
                continue
            x = _fill(_interp_to(imu.t, imu.cols[nme], tf))
            if not np.isfinite(x).all():
                continue
            x = _lowpass(x - x.mean(), fs, LOWPASS_HZ)
            for lag in LAGS_S:
                n = int(round(lag * fs))
                feats.append(np.concatenate([np.full(n, x[0]), x[:len(x) - n]]) if n else x)
    X = None
    if feats:
        X = np.stack(feats, 1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)

    def fit(Xa, ya, lam):
        A = Xa.T @ Xa + lam * len(ya) * np.eye(Xa.shape[1])
        return np.linalg.solve(A, Xa.T @ ya)

    # Out-of-fold prediction: the window is cut into FOLDS contiguous blocks; each block's IMU part
    # is predicted by a fit on the OTHER blocks only, so the IMU can't "memorise" the signal.
    # The ridge strength is picked the same way (the one that predicts unseen blocks best).
    n = len(ts)
    edges = np.linspace(0, n, FOLDS + 1).astype(int)
    guard = int(round(max(LAGS_S) * fs)) + 1        # drop samples next to the test block

    def oof(yl, lam):
        pred = np.zeros(n)
        for i in range(FOLDS):
            te = np.zeros(n, bool)
            te[edges[i]:edges[i + 1]] = True
            tr = ~te
            tr[max(0, edges[i] - guard):min(n, edges[i + 1] + guard)] = False
            pred[te] = X[te] @ fit(X[tr], yl[tr], lam)
        return pred

    scores = {}
    out_cols = {}
    for nme, y in cols.items():
        y = y - y.mean()
        out_cols[nme] = y
        if X is None:
            continue
        yl = _lowpass(y, fs, LOWPASS_HZ)
        best = None
        for lam in RIDGES:
            pred = oof(yl, lam)
            r2 = 1 - np.var(yl - pred) / max(np.var(yl), 1e-12)
            if best is None or r2 > best[0]:
                best = (r2, lam, pred)
        r2, lam, pred = best
        if r2 <= 0:                       # the IMU predicts nothing useful: don't subtract anything
            pred = np.zeros(n)
        out_cols[f"{nme}_imu"] = pred
        out_cols[f"{nme}_leftover"] = y - pred
        scores[nme] = {"imu_explains_heldout_pct": round(float(r2) * 100, 1), "ridge": lam,
                       "rms": round(float(np.std(y)), 2)}
    info["scores"] = scores

    # thruster correlation of each leftover (reported only)
    rc = next((s for s in data.series.values() if s.name.endswith("rc/out") and "channels[0]" in s.cols), None)
    if rc is not None and X is not None:
        th = []
        for i in range(8):
            c = rc.cols.get(f"channels[{i}]")
            if c is None:
                continue
            x = _fill(_interp_to(rc.t, c, tf))
            if np.isfinite(x).all() and np.std(x) > 0:
                th.append((i, _lowpass(x - x.mean(), fs, LOWPASS_HZ)))
        for nme in cols:
            lo = _lowpass(out_cols[f"{nme}_leftover"], fs, LOWPASS_HZ)
            if np.std(lo) > 0:
                best = max(((i, float(np.corrcoef(lo, x)[0, 1])) for i, x in th), key=lambda z: abs(z[1]), default=None)
                if best:
                    scores[nme]["leftover_best_thruster"] = {"channel": best[0], "r": round(best[1], 2)}

    kinds = {k: "float" for k in out_cols}
    return Series(DERIVED_PANEL, "derived/panels", "derived", tf, out_cols, kinds, "header", {}, info)


def panel_summary_html(ser: Series) -> str:
    info = ser.info
    u = info.get("units", "µm")
    lay = info.get("layout", {})
    lines = []
    for side in ("left", "right"):
        if side in lay:
            L = lay[side]
            lines.append(f"<b>{side.title()} panel</b>: outer column {L['outer']}, inner column {L['inner']}")
    rows = []
    for nme, sc in info.get("scores", {}).items():
        thr = sc.get("leftover_best_thruster")
        rows.append(f"<tr><td>{nme}</td><td align=right>{sc['rms']:.1f} {u}</td>"
                    f"<td align=right>{sc['imu_explains_heldout_pct']:.0f}%</td>"
                    + (f"<td align=right>ch{thr['channel']} r={thr['r']:+.2f}</td>" if thr else "<td></td>") + "</tr>")
    table = ("<table cellspacing=6><tr><th align=left>signal</th><th>RMS</th>"
             "<th>IMU explains<br>(unseen data)</th><th>leftover vs<br>thrusters</th></tr>" + "".join(rows) + "</table>")
    return "<br>".join(lines) + "<br>" + table
