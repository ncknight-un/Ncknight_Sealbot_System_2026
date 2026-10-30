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
3. Choose whether to export **all fields**, **ticked fields only**, or **tag_to_imu_format** (see below).
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
- **Same y axis for similar plots** (on by default; the box under the field list): plots of the same kind in one topic, e.g. its x, y and z when split into separate plots, or its thruster channels, share one y axis. If their values overlap they get identical limits; if their offsets are far apart (e.g. raw camera→tag z ≈ 0.3 m vs x, y ≈ 0), they get the same scale, each centred on its own data. Zooming or panning a plot takes it out of the group until you press its **A** button.
- Y axes otherwise fit the visible data. Detections count is fixed at 0–20; edit `Y_RANGES` in `bag_viewer.py` for others.

If a calculated entry is missing, **Bag info** and the terminal say why.

**Calculated results** are listed first in the field list, in **blue**, and plotted with blue titles. They are post-processing calculations, not recorded topics:

- `derived/imu_vel`: IMU velocity and distance for the current Start → Stop, recalculated whenever the window moves.
- `derived/forward_motion_diag`: a forward-motion diagnostic. `ch0_us` … `ch3_us` are the four horizontal thrusters (`rc/out` channels 0–3), PWM − 1500 µs. 0 = idle; the sign gives the direction. Set other channels with `--thrusters`.
- **RC channel actions.** Every RC channel shows what it does in brackets, in the field list, plot legends and titles, e.g. `channels[0] (horizontal thruster 1, forward)` on `rc/out` and `channels[4] (forward)` on `rc/in`. The defaults are ArduSub's BlueROV2 vectored frame; edit `RC_OUT_ACTIONS` / `RC_IN_ACTIONS` at the top of `bag_viewer.py` for another vehicle.
- `derived/tf[A->B]`: the measured pose of frame B seen from frame A (see **TF tree** below).
- `derived/heading`: one standard heading from the IMU. `heading_deg` uses the compass convention (0 = north, clockwise, 0–360). `heading_unwrapped_deg` is the same but continuous across north (359 → 361), so it's safe to difference, average or filter.

**More viewer controls:**

- **Full view.** Double-click a plot to fill the window with just that plot. Double-click again or press `Esc` to go back.
- **Plot grouping.** Each topic's plots sit inside a frame headed by the topic name: red for recorded topics, blue for calculated ones. By default a topic's ticked fields share plots, one per field group and unit. Tick the topic's **Separate** box in the field list to give every field its own plot.
- **Plot height.** The **Plot height** box under the field list sets each plot's height.
- **Topic rates.** The **Topic rates** button at the top opens a sortable table of every recorded topic: message type, message count, rate (messages ÷ time over its span, counted like the fps), typical interval, largest gap (red when it's more than 5× the typical interval, i.e. dropped messages) and time source.
- **Field list colours.** Hover over the *Field* header for the key.
  - **Green:** recorded sensor data.
  - **Blue:** calculated here (post-processing).
  - **Orange:** autopilot estimates (EKF outputs, not sensor readings): `imu/data` orientation, `global_position/local`, `compass_hdg`, `rel_alt`, `vfr_hud`, and `derived/heading`, which is computed from the autopilot's attitude. Edit `AUTOPILOT_ESTIMATES` in `bagcache.py` to change the list.
  - **Red:** empty fields with no data: always blank, a measurement or PWM output that's always 0, or the `(0, 0, 0, 1)` placeholder orientation mavros fills in when a message has none. A count or flag that's always 0 (e.g. `hamming`) is a real value, so it's shown as constant, not empty.
  - **Grey:** constant fields that never change.

  **Bag info** lists all the empty and constant fields and the autopilot estimates.
- **Tag overlay hold.** When the detector misses a tag for a frame or two, its last outline stays on screen, dashed and dimmer, for up to *hold* frames (default 3), so the overlay doesn't flicker. Set it to 0 to draw only the tags detected in that exact frame. Each tag keeps one label, so labels never swap between tags.
- **Frame rate.** The fps shown is counted: (frames − 1) ÷ (last stamp − first stamp), i.e. frames per second over the whole recording, dropped frames included. Topic rates in the field list and **Bag info** are counted the same way.
- **Rectify image.** This shows the frame undistorted with `camera_info`, the same way `image_proc` makes `image_rect`. Tick it when the AprilTag detector ran on the rectified image; the tag outlines then line up, including near the edges. It is on by default when the camera has distortion.
- **Start/stop limits.** Start and stop are limited to the bag, from 0 to the last frame.

### IMU velocity (BlueROV: no GPS, no DVL)

The IMU measures acceleration, so velocity comes from integrating it, and any small sensor bias grows into drift. Without GPS or a DVL, the fix is to integrate between two moments when the ROV is **at rest**:

- Velocity starts at 0 at **Start**.
- Whatever velocity is left at **Stop** must be bias, so it is removed linearly across the window.

**How it's calculated: IMU only.** The autopilot's orientation estimate (the quaternion in `imu/data`) is **not used**. Everything comes from the IMU's own gyro (`angular_velocity`) and accelerometer (`linear_acceleration`):

- **Start tilt.** Roll and pitch come from the accelerometer's gravity reading over the first second. Each sample is first rotated back to the Start attitude with the gyro, so a little rocking doesn't bias it. Its magnitude is used as g, instead of an assumed 9.81.
- **Gyro offset.** Measured over that same second if the ROV is still there. If not, the gyro's own boot calibration is trusted.
- **Attitude.** Propagated by integrating the gyro. Yaw starts at 0, so "forward" is relative to the heading at Start; no compass is used.
- **Stop alignment.** If the ROV has stopped at Stop (rocking in place is fine), the accelerometer's average gravity over the last 3 s shows how far the integrated tilt drifted, and the constant gyro offset that explains it is removed (roll and pitch only). On sub5 this found about 0.07°/s roll and −0.05°/s pitch, a normal size for this kind of gyro that still adds up to several degrees over 50 s.
- **Velocity.** `a = R f − [0, 0, g]`, integrated. Velocity is 0 at Start, and whatever is left at Stop is removed linearly. Surge, sway and heave are `Rᵀ v`.
- **New columns** `derived/imu_vel.att.roll_deg / pitch_deg / yaw_deg`: the IMU-only attitude itself.

**Start and Stop must be at rest.** Put Start just before the thrusters engage, and Stop at least 3 s after the ROV has coasted to a stop. The IMU window says whether each end qualified and what was measured there. The gyro offset is only measured at Start if the ROV is truly still; gentle rocking makes that measurement wrong, so it's skipped. A Stop while the ROV is still moving forces its real speed to 0 and distorts the whole run: on sub5, ending at 50.0 s instead of 52.4 s gives 99 m instead of 18 m.

On the whole sub5 run (window 1.0–52.4 s, still before the thrusters engage, stopped at the end): forward distance +18.2 m against about 22 m. Sway is still too large (+68 m, peak 3 m/s): while the thrusters run, the accelerometer reads an extra +0.2 to +0.3 m/s² sideways that vanishes at rest. That can't be real motion and isn't removable from the IMU alone; most likely thruster vibration, or a roll error that comes and goes during the ±35° pitching.

**Travel angle.** `travel angle` (in the IMU window and the status bar) is the direction actually travelled relative to the IMU's nose, + = left: `atan2(sway distance, surge distance)`. It is a diagnostic only: surge and sway are always along the IMU's own axes and are never rotated to make a run look straight (runs with turns would be wrong).

**Checking the maths.** `python sync_export.py --selftest-imu` runs the integration on simulated runs where the true surge is known and the true sway is 0 (straight, ±25° pitching, waves, rocking during Start, a 90° turn, gyro offsets, accelerometer bias and noise, an 8° mounting offset, autopilot attitude errors that are no longer used, a thrust-linked error) and prints the errors.

**Choosing Start and Stop.** You place Start and Stop yourself, on moments where the ROV is still. `derived/forward_motion_diag` (the four horizontal thrusters) is plotted by default. All four within ±25 µs of 0 means idle. Put Start just before they move off 0, and Stop a couple of seconds after it returns, so the ROV has coasted to rest.

- **Accuracy caveat.** The drift correction removes a false acceleration that stays the same through the window. It cannot remove one that changes, e.g. an attitude error that changes with thrust: an AHRS pulled by sustained acceleration tilts its pitch and leaks gravity into surge. On surfaceLine1 the IMU showed about 0.2 m/s² of apparent backward acceleration while cruising. Treat IMU distance as rough unless it's checked against a known distance.

**Viewing it:**

- **IMU velocity…** opens a separate window for the current window. It shows:
  - the four horizontal thrusters (forward-motion diagnostic, channels 0–3),
  - acceleration in the hull frame (surge, sway, heave),
  - surge, sway and heave speed, with the uncorrected surge and sway dashed,
  - surge, sway and total distance,
  - an attitude plot: roll, pitch and yaw from the IMU alone, with the autopilot's roll and pitch dashed for reference only,
  - a summary: forward, surge and sway distance, travel angle, heading change, the drift removed, whether Start and Stop were still, and the remaining accelerometer mismatch at Stop.
- **Add IMU velocity** shows the same plots in the main view and adds `derived/imu_vel.*` columns to the CSV:

| Column | Meaning |
|---|---|
| `forward_speed` | Speed along the ROV's heading (horizontal), m/s, + = ahead |
| `sideways_speed`, `vertical_speed` | Across the heading (+ left) and up/down (+ up), m/s |
| `forward_speed_raw` | Forward speed before drift correction |
| `surge`, `sway`, `heave` | Velocity along the IMU's own axes, from the IMU-only attitude (forward along the nose, + left, + up), m/s. These tilt with the ROV. `surge_raw`… are before drift correction; `acc_surge`… are accelerations. |
| `surge_distance`, `sway_distance` | Integral of surge / sway since Start, m |
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

The horizontal thruster channels are set with `--thrusters 0,1,2,3`.

**Depth.** Depth is `mavros/global_position/rel_alt`, from the external depth sensor (negative = below the surface), and also `global_position/local` `pose.pose.position.z`. `imu/static_pressure` is most likely the autopilot's internal barometer, not the depth sensor.

### Tag positions (pixel centre, unfiltered)

**Tag pixel-centre x/y…** computes each tag's x/y from its detected pixel centre: x = (u − cx)·z/fx and y = (v − cy)·z/fy, using `camera_info` (P when *Rectify image* is ticked, otherwise K). z is the tag's median pose depth, a constant. Nothing is smoothed or filtered, so every frequency the camera captures is kept, up to half the frame rate. The pixel centre is far less noisy than the pose's x/y: on tag 18 the noise is about 6 µm instead of 41 µm in x.

The results are added inside that tf topic, in blue, as `translation.x_px` and `translation.y_px` (metres). They're exported with the rest of the CSV. From the command line: `--tag-px tag_18`.


### TF tree: every live tag against its zero frame

Every `tag_#_zero` frame is automatically used as the reference for its matching live `tag_#`, so you can see how each tag deviates over time. No setup is needed.

- **Pairing.** A frame whose name carries *zero* or *static* (`tag_0_zero`, also `static_tag_0` or `tag_0_static`) is paired with the live `/tf` frame of the same name without that word: `tag_0_zero` ↔ `tag_0`. If the names don't match exactly, it falls back to the same tag number. The zero frame can come from `/tf_static` or `/tf`.
- **The signal.** `derived/tf[tag_0_zero->tag_0]` is the pose of the live `tag_0` in the `tag_0_zero` frame, on every camera frame where tag 0 is detected:

  `T(tag_0_zero → tag_0) = inverse(T(camera → tag_0_zero)) × T(camera → tag_0)`

  If the zero frame hangs from something other than the camera (e.g. another tag's zero frame), the whole chain is followed. It is the identity when the tag sits exactly on its zero pose, so the result *is* the deviation:

| Columns | Meaning |
|---|---|
| `translation_mm.x/y/z` | Deviation along the zero frame's own axes, mm. 0 = on the zero pose |
| `rotation.roll/pitch/yaw_deg` | Rotation away from the zero pose, deg (`rotation.x/y/z/w` is the same as a quaternion) |

  A cell is blank on frames where the tag wasn't detected.
- **Other static edges** (e.g. `base_link → camera_link`) are listed as `tf_static[P->C]` with their constant value (`translation_mm`).
- The raw recorded `/tf` topics (`tf[camera->tag_0]` …) keep `translation.x/y/z` in metres, exactly as published.
- **Any other pair.** **TF between frames…** adds any frame seen from any other, in one or both directions. From the command line, use `--tf A:B`.
- **Bag info** lists every `/tf` and `/tf_static` edge, then each live tag's mean and spread of `translation_mm` in its zero frame, and any zero frame that couldn't be paired, with the reason.

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
| `/tf_static` | n/a | each `tag_N_zero` is used as the reference for live `tag_N` (see **TF tree**) |
| `/camera/camera_info` | n/a | not columns; shown under **Bag info** |

Covariance arrays are left out. Add `--include-covariance` to include them.

### tag_to_imu_format

The third choice in the export dropdown (or `--format tag_to_imu_format` on the command line) writes a compact CSV with a fixed column order, one row per camera frame in the window:

| Columns | Meaning |
|---|---|
| `t_s` | Seconds since the export's first row: **Start is 0**, whatever the window (a window from 9.97 s starts at 0.000). The sidecar `.json` records where that 0 falls on the viewer's time axis (`t_s_zero_in_bag_s`) |
| `dt_s` | Seconds since the previous camera frame in the bag (a dropped frame shows as about 2× the usual value; blank only for the bag's very first frame) |
| `derived/tf[tag_N_zero->tag_N].translation_mm.x/y/z`, `.rotation.x/y/z/w` | Each live tag's deviation from its zero frame (see **TF tree**): mm along the zero frame's axes, and a quaternion that is (0, 0, 0, 1) on the zero pose. Sorted by tag number. Tags without a zero frame are left out and listed in the status bar. Blank when the tag wasn't detected in that frame |
| `derived/tf[tag_N_zero->tag_N].seen` | After each tag's rotation columns: 1 when the tag was detected in that frame, 0 when it wasn't |
| `…imu/data.linear_acceleration.x/y/z` | The accelerometer as recorded, m/s², **gravity included** (≈ +9.8 on z when level and still). No attitude estimate is involved, so nothing can shift it between test days |
| `…imu/data.angular_velocity.x/y/z` | The gyro, rad/s |
| `…global_position/local.pose.pose.position.z` | Depth, last, metres, **negative underwater** (MAVROS's up-positive axis). Published by MAVROS with a header (its stamp of the `GLOBAL_POSITION_INT` it came from), so **Autopilot timing** pairs it exactly |

The IMU columns are interpolated to each frame's time. There are no orientation columns and no `timestamp`/`frame_idx` columns. Tick **Save .json notes** to record the first absolute timestamp and frame range in a sidecar. The **Include IMU velocity** box doesn't apply to this format.

```bash
python sync_export.py BAG --start 4.2 --stop 21.75 --format tag_to_imu_format -o run1_tag_imu.csv
```

### Autopilot timing (exact sample times from the MAVLink stream)

With MAVROS timesync not locked, every MAVROS message is stamped with the Jetson's clock **when it arrives**, not when it was measured. The link delivers readings in bursts, so the stamps bunch up: on sub1 the IMU is exactly 100 Hz overall, but 24 % of its spacings are under 2 ms and 12 % are 20–40 ms pauses. Readings are then matched to the wrong camera frame by up to a few tens of ms, and the pauses show up as blank cells.

The raw MAVLink stream (`/uas1/mavlink_source`, recorded by MAVROS) still carries the autopilot's own sample time in almost every message (`time_boot_ms` in `ATTITUDE`, `GLOBAL_POSITION_INT`, `SCALED_PRESSURE`, ...; `time_usec` in `RAW_IMU`, ...). Tick **Autopilot timing** (in *Calculations*), or pass `--fcu-time`, to use it:

- **Clock offset from the fastest arrivals.** For each MAVLink message type, arrival − autopilot time is taken at its smallest in every 5 s block, and a straight line through those minimums gives the offset (it follows the slow drift between the autopilot's and the Jetson's clocks; an autopilot reboot starts a new line). Corrected time = autopilot time + offset, then mapped onto the camera's clock like any other topic.
- **Every MAVROS topic with a timestamped source** is re-stamped: `imu/data` from `ATTITUDE`; `imu/data_raw` from `RAW_IMU`; `static_pressure` from `SCALED_PRESSURE`; `global_position/local`, `rel_alt`, `compass_hdg` from `GLOBAL_POSITION_INT`; `rc/in`, `rc/out` from `RC_CHANNELS` / `SERVO_OUTPUT_RAW`. The list is `TOPIC_SOURCES` in `fcutime.py`.
- **Pairing.** Each MAVROS message is paired with the MAVLink message it came from: by header stamp, or by the recorder's arrival time for topics without a header (`rel_alt`, `compass_hdg`). Where a value is copied straight from the MAVLink message (`rel_alt` = `relative_alt`, heading = `hdg`, roll rate = `rollspeed`) it must match exactly, so a lost message on either side can't shift the pairing.
- **Left as recorded:** topics whose MAVLink message has no timestamp (`state` = `HEARTBEAT`, battery, `vfr_hud`), and any topic where under 90 % of messages pair up. Messages that can't be paired are dropped from that topic (they would otherwise sit at a wrong time).
- **What remains** is the link's smallest delay, a few ms, the same for every topic.
- **Bag info** lists which clock the camera, IMU and depth are on (`Clocks:`) and, for each topic, which MAVLink message it was paired with, how many paired, how far stamps moved, the link delay and the clock drift. The `.json` notes record it too. The first run reads the bag (a few seconds); the result is cached. The bag itself is never changed; untick the box to go back to the recorded stamps.

On a simulated bag with bursty delivery, stalls and lost messages (autopilot clock 30 ppm off), recorded stamps were off by 9 ms (typical) and over 40 ms (worst 1 %); with autopilot timing every topic was within 0.3 ms (the 1 ms resolution of `time_boot_ms`), plus the constant smallest delay.

**`bluerov` topics are ignored.** Everything under `/bluerov/` (the `bluerov_sensor_node.py` relays) is skipped when the bag is read, and listed under *Skipped* in **Bag info**; the MAVROS topics they relay are used instead (`IGNORED_NAMESPACES` in `bagcache.py`).

### Things to check on the real bag

**Headerless topics are slightly late.** `compass_hdg`, `rel_alt` and `MAVROS_latency` have no header, so they use the time the recorder received them. That is a few milliseconds later than when they were measured. **Bag info** shows each topic's median receive-minus-header latency. The headered mavros topics give a good estimate of that delay.

**Bags recorded from a playback (e.g. tagged at 0.25× speed).** If a bag was made by playing another bag, for example to run the AprilTag detector at 0.25× speed, every message keeps its original header stamp, but the recorder logs arrival times that run 4× slow and start at the re-recording time. The parser detects this from the header-to-receive relationship and uses the original header stamps for every topic. Topics without a header (`rel_alt`, `compass_hdg`, …) are mapped onto that clock. **Bag info** shows a `Clock:` line saying which case it found, and a warning gives the playback speed. Earlier versions used the arrival times instead, which stretched the whole timeline 4×: the camera looked like 16 fps instead of 65, and IMU velocities and distances came out far too large.

**Sanity check on header stamps.** In a live recording, if a topic's header stamps are more than 5 s away from its receive times (for example, unset stamps or a different clock), the tool falls back to receive time and lists a warning. In a playback recording, a topic whose header clock differs from the camera's is converted onto the camera's clock instead.

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
| `sync_export.py` | Interpolation, CSV export, IMU velocity, tf tree; also the command-line tool |
| `bag_viewer.py` | The GUI |
| `clock_check.py` | Checks whether the camera and IMU are stamped on the same clock (helper script) |
| `imu_timing.py` | Shows how evenly the IMU is stamped: rate, spacing, bursts (helper script) |
| `fcutime.py` | Autopilot timing: re-stamps MAVROS topics with the autopilot's sample times from the recorded MAVLink stream |
| `lag_check.py` | Estimates the tag-to-IMU timing offset from a `tag_to_imu_format` CSV (helper script) |

## Timing helper scripts

Run these from the `bagsync` folder. They only read; nothing is changed.

```bash
python clock_check.py BAG      # camera and IMU on the same clock?  "SAME clock" = yes
python imu_timing.py BAG       # IMU rate and stamp spacing; many near-0 spacings + 20-80 ms pauses = stamped on arrival
python lag_check.py run.csv    # timing offset between tags and IMU, separated from tag bending
```

- **`clock_check.py`** fits each topic's header stamps against the recorder's times and compares camera and IMU. It reports the rate difference, the offset in ms and a verdict. On sub1_tagged it found the camera on time-since-boot and the IMU on Unix time: different clocks, same rate. bagsync already maps one onto the other.
- **`imu_timing.py`** on sub1_tagged: 100.0 Hz, but 24 % of spacings under 2 ms and 12 % at 20–40 ms. That is arrival-time bunching, which **Autopilot timing** / `--fcu-time` fixes.
- **`lag_check.py`** compares each tag with the IMU's acceleration frequency by frequency, and fits one shared timing offset plus a spring model of each tag's bending. It reports the offset with a jackknife uncertainty and a table of each tag's shift by frequency band. It checks its own fit, and when the tags don't behave like simple springs (as on sub1, where they trail by 150–350 ms at 0.3–1 Hz but lead by 10–20 ms at 4–8 Hz) it prints **NO RELIABLE OFFSET** instead of a number. On simulated recordings with known offsets it came within 1–2 ms. A tap test, or a recording with both sensors on ROS time, gives the offset directly.

**Recommended export for training:**

```bash
python sync_export.py BAG --fcu-time --gap-factor 10 --format tag_to_imu_format -o run.csv
```

Check that the output says `mavros/imu/data` and `mavros/global_position/local` got "autopilot sample times". Export every bag the same way. A small fixed offset of a frame or two can remain; a model that sees a few frames of context learns it, as long as live data has the same timing.
