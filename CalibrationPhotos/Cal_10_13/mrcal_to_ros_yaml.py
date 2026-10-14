import sys, mrcal
m = mrcal.cameramodel(sys.argv[1]); name, out = sys.argv[2], sys.argv[3]
lm, k = m.intrinsics()
dm = {"LENSMODEL_OPENCV8": "rational_polynomial", "LENSMODEL_OPENCV5": "plumb_bob"}[lm]
fx, fy, cx, cy = map(float, k[:4]); d = [float(v) for v in k[4:]]
w, h = map(int, m.imagersize())
if len(sys.argv) == 8:
    ox, oy, w, h = map(int, sys.argv[4:8]); cx -= ox; cy -= oy
f = lambda v: ", ".join(f"{x:.8f}" for x in v)
open(out, "w").write(f"""image_width: {w}
image_height: {h}
camera_name: {name}
camera_matrix:
  rows: 3
  cols: 3
  data: [{f([fx,0,cx,0,fy,cy,0,0,1])}]
distortion_model: {dm}
distortion_coefficients:
  rows: 1
  cols: {len(d)}
  data: [{f(d)}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
projection_matrix:
  rows: 3
  cols: 4
  data: [{f([fx,0,cx,0,0,fy,cy,0,0,0,1,0])}]
""")
print(f"Wrote {out}: {w}x{h} fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
