# bagsync: camera-synced CSV export for ROS 2 bags

A viewer and a command-line exporter for ROS 2 (Humble) bags in MCAP or `.db3` format. Both resample every topic onto the camera's frame times and write one CSV. The synced timestamp is the first column, and each row is one camera frame.

It does not need ROS installed or sourced. It reads bags with the pure-Python `rosbags` library.

## Install (once)

```bash
python3 -m venv ~/bagsync-venv
source ~/bagsync-venv/bin/activate
pip install -r requirements.txt
```

On Ubuntu, if the window doesn't open and you see an error about the "xcb" plugin, run `sudo apt install libxcb-cursor0`.

## Use

```bash
source ~/bagsync-venv/bin/activate
python bag_viewer.py ~/bags/surfaceLine1_tagged
```

Pass the bag **folder** (the one holding `metadata.yaml` and `surfaceLine1_tagged_0.mcap`), or the `.mcap` file inside it.

**First open.** The first time you open a bag, it reads every message once (progress shows in the status bar). It saves the results to `surfaceLine1_tagged/.bagsync_cache/`, so later opens are instant. To force a re-read, delete that folder or click **Rebuild cache**.

**Steps in the viewer:**

1. Tick fields in the left list to plot them. The filter box accepts words, e.g. `imu yaw` or `tag0`.
2. Pick the export window in any of these ways:
   - drag the blue band or its edges,
   - type start/stop times,
   - move the yellow cursor to a frame and press `[` (start) or `]` (stop).
3. Choose whether to export **all fields** or **ticked fields only**.
4. Click **Export CSV**. Optionally also save that window's full-resolution frames as PNG files.

**Keys:**

| Key | Action |
|---|---|
| `←` / `→` | Previous / next frame |
| `Shift + ←` / `Shift + →` | Jump 10 frames |
| `Space` | Play / pause |
| `[` / `]` | Set start / stop at the cursor |
| `Ctrl + E` | Export |

Clicking anywhere on a plot jumps the cursor and the image to that time.

**Zooming:**

- Scroll inside a plot to zoom only that plot's y axis.
- Ctrl+scroll, or scroll over a time axis, to zoom time for all plots together.
- Shift+drag pans y.
- The small **A** button in a plot's corner refits it.
- Y axes otherwise fit the visible data. Detections count is fixed at 0–20; edit `Y_RANGES` in `bag_viewer.py` for others.

If a calculated entry is missing, **Bag info** and the terminal say why.

**Calculated results** are listed first in the field list, in **blue**, and plotted with blue titles. They are post-processing calculations, not recorded topics:

- `derived/imu_vel`: IMU velocity and distance for the current Start → Stop, recalculated whenever the window moves.
- `derived/thrusters`: horizontal thruster activity.
- `derived/heading`: one standard heading from the IMU. `heading_deg` uses the compass convention (0 = north, clockwise, 0–360). `heading_unwrapped_deg` is the same but continuous across north (359 → 361), so it's safe to difference, average or filter.

**More viewer controls:**

- **Full view.** Double-click a plot to fill the window with just that plot. Double-click again or press `Esc` to go back.
- **Plot grouping.** Each topic's plots sit inside a frame headed by the topic name: red for recorded topics, blue for calculated ones. By default a topic's ticked fields share plots, one per field group and unit. Tick the topic's **Separate** box in the field list to give every field its own plot.
- **Plot height.** The **Plot height** box under the field list sets each plot's height.
- **Field list colours.** Recorded topics are listed in green; calculated (post-processing) entries in blue.
- **Rectify image.** This shows the frame undistorted with `camera_info`, the same way `image_proc` makes `image_rect`. Tick it when the AprilTag detector ran on the rectified image; the tag outlines then line up, including near the edges. It is on by default when the camera has distortion.
- **Start/stop limits.** Start and stop are limited to the bag, from 0 to the last frame.

### IMU velocity (BlueROV: no GPS, no DVL)

The IMU measures acceleration, so velocity comes from integrating it, and any small sensor bias grows into drift. Without GPS or a DVL, the fix is to integrate between two moments when the ROV is **at rest**:

- Velocity starts at 0 at **Start**.
- Whatever velocity is left at **Stop** must be bias, so it is removed linearly across the window.

**Choosing Start and Stop.** You place Start and Stop yourself, on moments where the ROV is still. `derived/thrusters.max_dev_us` shows how hard the horizontal thrusters are working (0 = off) to help you choose. That means `rc/out` channels 0–3 are within ±25 µs of 1500. The vertical thrusters (4–5) are ignored because they keep working to hold depth.

- `derived/thrusters.max_dev_us` is plotted by default; flat near 0 means idle.
- You choose Start and Stop yourself. Put Start just before the thrusters engage, and Stop a couple of seconds after they go idle so the ROV has coasted to rest.
- A warning appears if Start or Stop lands while the thrusters are running, or too soon after they stop.
- **Accuracy caveat (surfaceLine1).** The IMU shows about 0.2 m/s² of apparent backward acceleration even while cruising, roughly a 1° attitude error. It changes with thrust, so the at-rest correction can't remove it. Treat IMU distance as rough unless it's checked against a known distance.

**Viewing it:**

- **IMU velocity…** opens a separate window for the current window. It shows:
  - thruster activity,
  - acceleration in the ROV frame,
  - forward, sideways and vertical speed, with the uncorrected forward speed dashed so you can see the drift,
  - distance,
  - a summary of forward distance, peak and mean forward speed, the drift removed, and the bias it implies.
- **Add IMU velocity** shows the same plots in the main view and adds `derived/imu_vel.*` columns to the CSV:

| Column | Meaning |
|---|---|
| `forward_speed` | Speed along the ROV's heading (horizontal), m/s, + = ahead |
| `sideways_speed`, `vertical_speed` | Across the heading (+ left) and up/down (+ up), m/s |
| `forward_speed_raw` | Forward speed before drift correction |
| `surge`, `sway`, `heave` | Velocity along the ROV's own body axes (forward along the nose, + left, + up), m/s. These tilt with the ROV: the frame the water and anything mounted on it feel. `surge_raw`… are before drift correction; `acc_surge`… are accelerations. |
| `forward_distance` | Distance along the heading since Start, m |
| `speed` | Horizontal speed, m/s |
| `distance` | Horizontal path length since Start, m |
| `acc_forward`, `acc_sideways`, `acc_vertical`, `acc_world.*` | Gravity-free acceleration, m/s² |
| `vel.e/n/u`, `disp.e/n/u` | World-frame velocity and displacement |

**Accuracy.** In a simulated run (still → 0.5 m/s forward for 21 s → still) with bias, noise and wave pitching, the result was:
- forward speed within about 0.05 m/s,
- distance within about 7%.

Accuracy drops the longer the window is and the less still the ROV really is at Start and Stop.

From the command line:

```bash
python sync_export.py BAG --start 0.3 --stop 27.8 --integrate-imu -o run.csv
```

The thruster channels are set with `--thrusters 0,1,2,3`.

**Depth.** Depth is `mavros/global_position/rel_alt`, from the external depth sensor (negative = below the surface), and also `global_position/local` `pose.pose.position.z`. `imu/static_pressure` is most likely the autopilot's internal barometer, not the depth sensor.

### Tag positions (pixel centre, unfiltered)

**Tag pixel-centre x/y…** computes each tag's x/y from its detected pixel centre: x = (u − cx)·z/fx and y = (v − cy)·z/fy, using `camera_info` (P when *Rectify image* is ticked, otherwise K). z is the tag's median pose depth, a constant. Nothing is smoothed or filtered, so every frequency the camera captures is kept, up to half the frame rate. The pixel centre is far less noisy than the pose's x/y: on tag 18 the noise is about 6 µm instead of 41 µm in x.

The results are added inside that tf topic, in blue, as `translation.x_px` and `translation.y_px` (metres). They're exported with the rest of the CSV. From the command line: `--tag-px tag_18`.

The earlier Kalman smoothing option has been removed, because it cut off motion above about 1–4 Hz.

### Panel analysis (command line only for now: `--panels`)

The panel analysis (hidden in the viewer for now) produces a window that turns the tag grid into a few physical signals for Start → Stop. The layout is found automatically: the tags are split into a left and a right panel, then into columns and rows; reference tags 0 and 20 are left out. All signals are in µm, measured from each tag's pixel centre.

| Signal | Meaning |
|---|---|
| `left.open`, `right.open` | Outer column minus inner column along x; + = the outer edge moves away from the image centre |
| `both.open_sym` | Both panels opening together |
| `both.open_asym` | One panel opens while the other closes |
| `*.shift_x`, `*.shift_y` | The whole panel moving |
| `*.bend_y` | Outer column minus inner column along y |

Each signal is split three ways:
- **measured** (grey),
- **IMU explains** (blue): predicted from rotation rates, roll/pitch and accelerations, including delayed responses. Each part of the window is predicted from a fit on the *other* parts only, so the IMU can't simply memorise the signal.
- **leftover** (orange): the motion the IMU can't explain, your candidate flow signal.

The table gives each signal's size, how much the IMU explains on data it wasn't fitted to, and which thruster channel the leftover follows. Tick **Include panel signals in CSV** to export `derived/panels.*` (`<signal>`, `<signal>_imu`, `<signal>_leftover`). From the command line, use `--panels`.

### Command line (same export, no GUI)

```bash
python sync_export.py ~/bags/surfaceLine1_tagged --info        # topics, rates, time sources, warnings
python sync_export.py ~/bags/surfaceLine1_tagged --start 4.2 --stop 21.75 -o run1.csv
python sync_export.py BAG --start 4.2 --stop 21.75 --topics imu/data rel_alt detections -o run1_small.csv
python sync_export.py BAG --start 4.2 --stop 21.75 --export-images run1_frames/
python sync_export.py BAG --rate 100 -o run1_100hz.csv          # fixed-rate rows instead of camera rows
```

`--start` and `--stop` are seconds since the first camera frame. That is the same axis the viewer shows.

`--topics` keeps every topic whose name contains one of the words you give, so `imu/data` also matches `imu/data_raw`.

## The CSV

The first three columns are always:

| Column | Meaning |
|---|---|
| `timestamp` | The camera frame's `header.stamp`: Unix seconds with full nanosecond precision, e.g. `1792123034.961455472` |
| `t_s` | Seconds since the first camera frame |
| `frame_idx` | The frame's index in the bag. Exported PNGs are named `<frame_idx>_<timestamp>.png`, so rows and images pair up |

Every other column is named `<topic>.<field>`, e.g. `mavros/imu/data.orientation.yaw_deg`. The repeated `mavros/mavros` prefix is shortened to `mavros`.

A cell is left **empty** rather than invented when:
- the topic has no data before or after that frame,
- the frame falls inside a gap longer than 3× the topic's normal sample period (change this with `gap ×` in the viewer or `--gap-factor`), or
- an AprilTag wasn't seen in that frame.

**Sidecar file (optional).** Tick **Save .json notes** (or pass `--sidecar`) to also write a `*.sync.json` file next to the CSV. It records:
- the window,
- how each column was synced,
- each topic's time source and rate,
- any warnings.

## How each topic in this bag is synced

| Topic | Time used | Brought to the frame time by |
|---|---|---|
| `/camera/image_raw` (≈65 fps) | `header.stamp` | **master clock**: each row is one frame |
| `/detections` | `header.stamp`, which equals the image stamp | **exact match** to its own frame; adds `count` plus per-tag `centre`, `corners`, `decision_margin`, `hamming` |
| `/tf` (apriltag → tag poses) | `header.stamp`, which equals the image stamp | **exact match**; one column group per `parent->child` |
| `/mavros/.../imu/data`, `global_position/local` | `header.stamp` | linear interpolation; quaternions use **SLERP**; `roll/pitch/yaw_deg` added |

**Quaternion signs.** `q` and `−q` are the same rotation, and the AprilTag/tf output picks either sign frame to frame. For tags seen ~180° rotated through the dome (`w ≈ 0`) that made `rotation.x/y/z/w` jump between ± values with no real motion. Since v9 every quaternion series is made sign-continuous on load (first sample's largest component positive), and `roll/yaw_deg` are unwrapped so angles near ±180° don't jump by 360°. Pitch and the actual rotation are unchanged.

| `/mavros/.../imu/data_raw`, `static_pressure`, `rangefinder`, `battery`, `time_reference` | `header.stamp` | linear interpolation |
| `/mavros/.../global_position/compass_hdg` | **receive time** (no header) | linear interpolation **across the 359°→0° wrap** |
| `/mavros/.../global_position/rel_alt`, `/MAVROS_latency` | **receive time** (no header) | linear interpolation |
| `/mavros/.../state`, `rc/in`, `rc/out`, and other integer, boolean or text fields | `header.stamp` | **last value** at or before the frame |
| `/camera/camera_info`, `/tf_static` | n/a | not columns; shown under **Bag info** |

Covariance arrays are left out. Add `--include-covariance` to include them.

### Things to check on the real bag

**Headerless topics are slightly late.** `compass_hdg`, `rel_alt` and `MAVROS_latency` have no header, so they use the time the recorder received them. That is a few milliseconds later than when they were measured. **Bag info** shows each topic's median receive-minus-header latency. The headered mavros topics give a good estimate of that delay.

**Sanity check on header stamps.** If a topic's header stamps are more than 5 s away from its receive times (for example, unset stamps or a different clock), the tool falls back to receive time and lists a warning.

**Dropped frames.** A dropped frame was never recorded, so it shows up as a bigger step in `t_s` between two rows; `frame_idx` stays continuous. The estimated count and the largest gap show in **Bag info**.

**Exact matching is automatic.** A topic is matched exactly (rather than interpolated) when at least 90% of its stamps coincide with camera frame stamps. Anything else on `/tf` at its own rate gets interpolated instead.

**Message types.** The `apriltag_msgs` and `mavros_msgs` types are read from the definitions embedded in the MCAP. Built-in fallbacks are included in case they are missing. If a topic is listed as skipped with an unknown type, run with `--msg-dir path/to/your_pkg/msg`.

## Tuning

| What | Where |
|---|---|
| Which fields are ticked when a bag opens | `DEFAULT_PLOTS` in `bag_viewer.py` |
| Per-column sync overrides (e.g. treat another field as an angle) | `RULES` in `sync_export.py` |
| Gap tolerance | `GAP_FACTOR` in `sync_export.py` (or the UI / `--gap-factor`) |
| Different camera topic | `--camera /other/image_raw` |

## Files

| File | Contents |
|---|---|
| `bagcache.py` | Reads the bag, flattens fields, picks time sources, caches arrays and JPEG thumbnails |
| `sync_export.py` | Interpolation and CSV export; also the command-line tool |
| `bag_viewer.py` | The GUI |
