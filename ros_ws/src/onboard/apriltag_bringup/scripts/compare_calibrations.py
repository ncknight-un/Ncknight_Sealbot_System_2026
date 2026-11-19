#!/usr/bin/env python3
"""Compare two panel_calibration files (e.g. CPU vs Isaac) recorded on the same, unmoved panel.

    ros2 run apriltag_bringup compare_calibrations.py panel_zero_cpu.yaml panel_zero_isaac.yaml

For each tag common to both, prints the position difference and the rotation
`corr` such that  q_isaac * corr = q_cpu.  If every tag gives (nearly) the same
large rotation, Isaac's tag frame axes differ from apriltag_ros': put the
suggested mean quaternion in apriltag_bringup/config/isaac_tags.yaml
(tag_rotation_correction). If it's only a fraction of a degree, leave it identity.
"""

import math
import sys

import numpy as np
import yaml


def load(path):
    with open(path) as f:
        p = next(iter(yaml.safe_load(f).values()))['ros__parameters']
    return p.get('detector', 'unknown'), {
        name: (np.array(t['translation']), np.array(t['rotation']))
        for name, t in p['tags'].items()}


def qmul(a, b):  # [x, y, z, w]
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz])


def qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def angle_deg(q):
    return math.degrees(2 * math.acos(min(1.0, abs(q[3]))))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    det_a, a = load(sys.argv[1])
    det_b, b = load(sys.argv[2])
    common = sorted(set(a) & set(b), key=lambda s: int(''.join(c for c in s if c.isdigit()) or 0))
    if not common:
        sys.exit('No tags in common.')

    print(f'reference (A): {sys.argv[1]} [{det_a}]')
    print(f'compared  (B): {sys.argv[2]} [{det_b}]\n')
    print(f'{"tag":>8} {"|dt| mm":>9} {"corr angle":>11}   corr [x, y, z, w]  (q_B * corr = q_A)')
    corrs = []
    for name in common:
        (ta, qa), (tb, qb) = a[name], b[name]
        c = qmul(qinv(qb), qa)
        c = c if c[3] >= 0 else -c
        corrs.append(c)
        print(f'{name:>8} {np.linalg.norm(ta - tb) * 1e3:9.3f} {angle_deg(c):10.3f}°   '
              f'[{", ".join(f"{v:+.4f}" for v in c)}]')

    # Markley average of the per-tag corrections
    m = sum(np.outer(c, c) for c in corrs)
    mean = np.linalg.eigh(m)[1][:, -1]
    mean = mean if mean[3] >= 0 else -mean
    spread = max(angle_deg(qmul(qinv(mean), c)) for c in corrs)
    print(f'\nmean correction: [{", ".join(f"{v:.6f}" for v in mean)}]  '
          f'({angle_deg(mean):.3f}°, max spread {spread:.3f}°)')
    if angle_deg(mean) < 1.0:
        print('-> Axes agree; keep tag_rotation_correction as identity.')
    elif spread < 2.0:
        print('-> Consistent fixed rotation: set tag_rotation_correction to the mean above.')
    else:
        print('-> Corrections disagree between tags; check the panel did not move between runs.')


if __name__ == '__main__':
    main()
