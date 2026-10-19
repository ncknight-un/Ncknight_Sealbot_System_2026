#!/usr/bin/env python3
"""
bag_viewer.py - look through a ROS 2 bag against its camera, pick a window, export a synced CSV.

    python bag_viewer.py                      # opens a folder picker
    python bag_viewer.py path/to/bag_folder   # or the .mcap inside it
    python bag_viewer.py BAG --camera /camera/image_raw --msg-dir my_msgs/

Layout
    left    every topic and field; tick fields to plot them (filter box on top)
    middle  stacked plots on one shared time axis (seconds since first camera frame)
            blue band = export window (drag it or its edges), yellow line = cursor
    right   the camera frame at the cursor, with AprilTag outlines and ids
    bottom  start / stop, sync mode, which fields to export, Export CSV

Keys
    Left / Right        previous / next frame (Shift = 10 frames)
    Space               play / pause
    [ and ]             set start / stop to the cursor
    Ctrl+E              export          Ctrl+O   open another bag
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:  # prefer Qt 6 (PySide6) even when PyQt5 from apt is also installed
    import PySide6  # noqa: F401
except ImportError:
    pass
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from bagcache import VERSION, BagData, bag_summary, load_bag
from panels import DERIVED_PANEL, panel_analysis, panel_summary_html
from sync_export import (REFERENCE_TAGS, DERIVED_IMU, DERIVED_THR, GAP_FACTOR, THRUSTER_CHANNELS, export_images,
                         heading_series, tag_pixel_position,
                         imu_summary, integrate_imu, sync,
                         thruster_series, write_csv)

try:
    import cv2
except ImportError:
    cv2 = None

pg.setConfigOptions(antialias=False, imageAxisOrder="row-major", background="#1b1d21", foreground="#c9ccd1")

LINE_COLORS = ["#4e9cf5", "#f5a14e", "#5fd38d", "#e5625c", "#b58cf2", "#e8d25a",
               "#4ed0d6", "#f27fb7", "#a3d65c", "#ff8a65", "#9fa8da", "#ffffff"]
# Units shown in plot titles (regex on '<topic>.<field group>').
UNITS = [
    (r"_deg$|roll/pitch/yaw|compass_hdg", "deg"),
    (r"angular_velocity|twist\.angular", "rad/s"),
    (r"linear_acceleration|acc_world", "m/s²"),
    (r"linear_acceleration|acc_world|^acc_", "m/s²"),
    (r"twist\.linear|\bvel|speed|^surge|^sway|^heave", "m/s"),
    (r"position|disp|distance|rel_alt|\.range|translation", "m"),
    (r"orientation|rotation", "quaternion"),
    (r"voltage", "V"),
    (r"pressure", "Pa"),
    (r"max_dev_us|channels", "µs"),
]
REGION_BRUSH = (78, 156, 245, 38)
CURSOR_PEN = pg.mkPen("#ffd84d", width=1.5)

# Fields ticked when a bag is first opened (regex on '<topic short>.<field>').
DEFAULT_PLOTS = [
    r"^derived/thrusters\.max_dev_us$",
    r"^derived/heading\.heading_deg$",
    r"^derived/imu_vel\.(surge|forward_distance)$",
    r"rel_alt\.data$",
    r"compass_hdg\.data$",
    r"imu/data\.orientation\.(roll|pitch|yaw)_deg$",
    r"global_position/local\.twist\.twist\.linear\.[xyz]$",
    r"rangefinder.*\.range$",
    r"^detections\.count$",
]
MAX_PLOTS = 40
PLOT_HEIGHT = 210              # default height of each field's plot, px (changeable in the UI)
TOPIC_FRAME_COLOR = "#a970ff"   # purple frame around the plots of one recorded topic
NATIVE_COLOR = "#2e9e4f"       # recorded topics in the field list   # red frame around the plots of one recorded topic
DERIVED_COLOR = "#4e9cf5"     # post-processing results are shown in this colour in the field list
# Fixed y ranges (regex on '<topic>.<field>'); every field in a plot must match for it to apply.
Y_RANGES = [
    (r"^detections\.count$", (0, 20)),
]


def shrinkable(label):
    """A pyqtgraph LabelItem normally demands its full text width as the layout's minimum
    width, so a long title made the plots wider than the panel and pushed them under the
    camera image. This lets the label shrink (the text is clipped at the plot's edge)."""
    def update_min(lab=label):
        b = lab.itemRect()
        lab.setMinimumWidth(10)
        lab.setMinimumHeight(b.height())
        lab._sizeHint = {
            QtCore.Qt.SizeHint.MinimumSize: (10, b.height()),
            QtCore.Qt.SizeHint.PreferredSize: (10, b.height()),
            QtCore.Qt.SizeHint.MaximumSize: (-1, -1),
            QtCore.Qt.SizeHint.MinimumDescent: (0, 0),
        }
        lab.updateGeometry()
    label.updateMin = update_min
    label.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True)
    update_min()
    return label


class TopicFrame(pg.GraphicsLayout):
    """A sub-layout with a coloured border: holds the separate plots of one topic."""

    def __init__(self, color):
        super().__init__()
        self._pen = pg.mkPen(color, width=2)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemHasNoContents, False)
        self.layout.setContentsMargins(8, 4, 8, 10)
        self.layout.setVerticalSpacing(10)

    def paint(self, p, *args):
        p.setPen(self._pen)
        p.drawRect(self.boundingRect().adjusted(1, 1, -1, -1))


class ZoomViewBox(pg.ViewBox):
    """
    Scroll in a plot      -> zoom that plot's y axis only.
    Scroll on the time axis, or Ctrl+scroll in a plot -> zoom time (all plots stay in sync).
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.manual_y = False

    def wheelEvent(self, ev, axis=None):
        ctrl = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        if axis == 0 or (axis is None and ctrl):
            super().wheelEvent(ev, axis=0)
        else:
            self.manual_y = True
            super().wheelEvent(ev, axis=1)

    def mouseDragEvent(self, ev, axis=None):
        if axis == 1 or (axis is None and ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
            self.manual_y = True
            super().mouseDragEvent(ev, axis=1)
        else:
            super().mouseDragEvent(ev, axis=0)
QShortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut
DEVICE_CACHE = QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache


# --------------------------------------------------------------------------- #
# Background workers
# --------------------------------------------------------------------------- #
class Worker(QtCore.QThread):
    progress = QtCore.Signal(float, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self):
        try:
            self.done.emit(self.fn(*self.args, progress=self._prog, **self.kwargs))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _prog(self, frac, text):
        self.progress.emit(float(frac), str(text))


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data: BagData | None = None
        self.frame = -1
        self._thumb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._plots: list[pg.PlotItem] = []
        self._cursors: list[pg.InfiniteLine] = []
        self._regions: list[pg.LinearRegionItem] = []
        self._syncing = False
        self._pending_x: float | None = None
        self._worker = None
        self._playing = False
        self._play_t0 = 0.0
        self._play_x0 = 0.0
        self._plot_keys: list[tuple] = []
        self._solo: tuple | None = None
        self._split_sizes = None
        self._t_end = 0.0
        self._imu_series = None
        self._imu_curves: list = []
        self._imu_win = None
        self._panel_win = None
        self._derived_status: dict[str, str] = {}
        self._separate: dict[str, bool] = {}
        self.thruster_channels = list(THRUSTER_CHANNELS)

        self.setWindowTitle(f"Bag viewer — {VERSION}")
        import bagcache as _bc
        import sync_export as _se
        print(f"[bagsync] {VERSION}  viewer: {Path(__file__).resolve()}\n"
              f"[bagsync]   bagcache: {Path(_bc.__file__).resolve()}\n"
              f"[bagsync]   sync_export: {Path(_se.__file__).resolve()}", flush=True)
        self.resize(1600, 950)
        self._build_ui()
        self._build_shortcuts()

        self._cursor_timer = QtCore.QTimer(self, singleShot=True, interval=15)
        self._cursor_timer.timeout.connect(self._apply_pending_cursor)
        self._replot_timer = QtCore.QTimer(self, singleShot=True, interval=60)
        self._replot_timer.timeout.connect(self._rebuild_plots)
        self._play_timer = QtCore.QTimer(self, interval=15)
        self._play_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._play_timer.timeout.connect(self._play_tick)
        self._label_timer = QtCore.QTimer(self, singleShot=True, interval=80)
        self._label_timer.timeout.connect(self._on_view_changed)
        self._end_labels = []
        self._imu_timer = QtCore.QTimer(self, singleShot=True, interval=200)
        self._imu_timer.timeout.connect(self._update_imu_velocity)
        self._imu_timer.timeout.connect(self._update_panels)

        if args.bag:
            QtCore.QTimer.singleShot(0, lambda: self.open_bag(args.bag))

    @property
    def _imu_series(self):
        return self.data.series.get(DERIVED_IMU) if getattr(self, "data", None) else None

    @_imu_series.setter
    def _imu_series(self, ser):
        if getattr(self, "data", None) is None:
            return
        if ser is None:
            self.data.series.pop(DERIVED_IMU, None)
        else:
            self.data.series[DERIVED_IMU] = ser

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.addAction("Open bag…", self.choose_bag)
        tb.addAction("Bag info", self.show_info)
        tb.addAction("Rebuild cache", lambda: self.open_bag(self.data.bag_dir, rebuild=True) if self.data else None)
        tb.addSeparator()
        self.bag_label = QtWidgets.QLabel("  No bag open")
        tb.addWidget(self.bag_label)

        # left: field tree
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        self.filter_edit = QtWidgets.QLineEdit(placeholderText="Filter fields…  (e.g. imu yaw)")
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Field", "Separate", "Info"])
        self.tree.headerItem().setToolTip(1, "Ticked: every field of this topic gets its own plot.\n"
                                             "Unticked: the topic's fields share plots (one per unit).")
        self.tree.setUniformRowHeights(True)
        self.tree.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.tree.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.tree.header().setStretchLastSection(False)
        self.tree.itemChanged.connect(lambda *_: self._replot_timer.start())
        btns = QtWidgets.QHBoxLayout()
        b_clear = QtWidgets.QPushButton("Untick all")
        b_clear.clicked.connect(self._untick_all)
        btns.addWidget(b_clear)
        btns.addStretch(1)
        self.plot_h = QtWidgets.QSpinBox(minimum=80, maximum=1000, singleStep=20, value=PLOT_HEIGHT, suffix=" px")
        self.plot_h.setToolTip("Height of each plot")
        self.plot_h.valueChanged.connect(lambda *_: self._replot_timer.start())
        btns.addWidget(QtWidgets.QLabel("Plot height"))
        btns.addWidget(self.plot_h)
        self.end_labels = QtWidgets.QCheckBox("Units at line ends", checked=True)
        self.end_labels.toggled.connect(lambda *_: self._rebuild_plots())
        btns.addWidget(self.end_labels)
        lv.addWidget(self.filter_edit)
        lv.addWidget(self.tree, 1)
        lv.addLayout(btns)

        # middle: plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setMinimumWidth(200)
        self.plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                       QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.plot_widget.ci.setSpacing(14)
        self.plot_widget.scene().sigMouseClicked.connect(self._plot_clicked)
        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setWidget(self.plot_widget)
        self.plot_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # right: image
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        self.frame_label = QtWidgets.QLabel("—")
        self.frame_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        # Ignored width + fixed height: label text changes (tags appearing) never resize the layout.
        self.frame_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.frame_label.setFixedHeight(self.frame_label.fontMetrics().height() * 2 + 6)
        self.frame_label.setWordWrap(True)
        self.img_widget = pg.GraphicsLayoutWidget()
        self.img_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.img_view = self.img_widget.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.img_item = pg.ImageItem()
        self.img_view.addItem(self.img_item)
        self.tag_curve = pg.PlotCurveItem(pen=pg.mkPen("#39ff7a", width=2), connect="finite")
        self.img_view.addItem(self.tag_curve)
        self._tag_texts: list[pg.TextItem] = []
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(lambda v: self.set_frame(v, move_cursor=True))
        ctl = QtWidgets.QHBoxLayout()
        self.b_prev = QtWidgets.QPushButton("◀")
        self.b_play = QtWidgets.QPushButton("Play")
        self.b_next = QtWidgets.QPushButton("▶")
        self.speed = QtWidgets.QComboBox()
        self.speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.speed.setCurrentIndex(2)
        self.b_prev.clicked.connect(lambda: self.step(-1))
        self.b_next.clicked.connect(lambda: self.step(1))
        self.b_play.clicked.connect(self.toggle_play)
        self.show_tags = QtWidgets.QCheckBox("Tag overlay", checked=True)
        self.show_tags.toggled.connect(lambda *_: self._draw_tags())
        self.rectify = QtWidgets.QCheckBox("Rectify image")
        self.rectify.setToolTip("Undistort the frame with camera_info (K, D, R, P), the same as image_proc's\n"
                                "image_rect. Tick this when apriltag ran on the rectified image, so the tag\n"
                                "outlines line up, especially near the edges.")
        self.rectify.toggled.connect(self._rectify_toggled)
        for w in (self.b_prev, self.b_play, self.b_next, self.speed):
            ctl.addWidget(w)
        ctl.addStretch(1)
        ctl2 = QtWidgets.QHBoxLayout()
        for w in (self.show_tags, self.rectify):
            ctl2.addWidget(w)
        ctl2.addStretch(1)
        rv.addWidget(self.frame_label)
        rv.addWidget(self.img_widget, 1)
        rv.addWidget(self.slider)
        rv.addLayout(ctl)
        rv.addLayout(ctl2)

        self.left_panel, self.right_panel = left, right
        right.setMinimumWidth(320)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.split = split
        split.addWidget(left)
        split.addWidget(self.plot_scroll)
        split.addWidget(right)
        split.setSizes([420, 760, 440])
        split.setStretchFactor(1, 1)

        # bottom: window + export
        bottom = QtWidgets.QWidget()
        bv = QtWidgets.QVBoxLayout(bottom)
        bv.setContentsMargins(6, 4, 6, 6)
        bv.setSpacing(4)
        bl = QtWidgets.QHBoxLayout()
        bl.setSpacing(8)
        bv.addLayout(bl)

        def spin():
            s = QtWidgets.QDoubleSpinBox(decimals=3, singleStep=0.1, suffix=" s", keyboardTracking=False)
            s.setRange(0.0, 0.0)
            s.setMinimumWidth(95)
            return s

        self.start_spin, self.stop_spin = spin(), spin()
        self.start_spin.valueChanged.connect(self._spins_changed)
        self.stop_spin.valueChanged.connect(self._spins_changed)
        b_s = QtWidgets.QPushButton("Start = cursor  [")
        b_e = QtWidgets.QPushButton("Stop = cursor  ]")
        b_all = QtWidgets.QPushButton("Whole bag")
        b_s.clicked.connect(lambda: self.set_window(self.cursor_x(), None))
        b_e.clicked.connect(lambda: self.set_window(None, self.cursor_x()))
        b_all.clicked.connect(self._full_window)
        self.window_label = QtWidgets.QLabel("")
        self.window_label.setMinimumWidth(170)

        self.sync_mode = QtWidgets.QComboBox()
        self.sync_mode.addItems(["Rows = camera frames", "Rows = fixed rate"])
        self.rate_spin = QtWidgets.QDoubleSpinBox(decimals=1, suffix=" Hz", value=50.0)
        self.rate_spin.setRange(0.1, 10000)
        self.rate_spin.setEnabled(False)
        self.sync_mode.currentIndexChanged.connect(
            lambda i: (self.rate_spin.setEnabled(i == 1), self._update_window_label()))
        self.rate_spin.valueChanged.connect(lambda *_: self._update_window_label())
        self.cols_mode = QtWidgets.QComboBox()
        self.cols_mode.addItems(["Export all fields", "Export ticked fields only"])
        self.gap_spin = QtWidgets.QDoubleSpinBox(decimals=1, value=GAP_FACTOR, prefix="gap × ")
        self.gap_spin.setRange(1.0, 1000.0)
        self.gap_spin.setToolTip("Leave a value blank when the frame falls in a gap longer than this many "
                                 "times the topic's normal sample period.")
        self.png_check = QtWidgets.QCheckBox("Frames as PNG")
        self.json_check = QtWidgets.QCheckBox("Save .json notes")
        self.json_check.setToolTip("Also write <name>.sync.json next to the CSV: the window, how each column was\n"
                                   "synced, time sources and warnings. Off by default.")
        self.panel_btn = QtWidgets.QPushButton("Panel analysis…")
        self.panel_btn.setToolTip("Open a window with the panel signals (open/close, shifts) for Start→Stop,\n"
                                  "split into the part the IMU explains and the leftover it can't.")
        self.panel_btn.clicked.connect(self.open_panel_window)
        self.panel_check = QtWidgets.QCheckBox("Include panel signals in CSV")
        self.panel_check.setToolTip("Add derived/panels.* columns (signal, _imu, _leftover) to the exported CSV.")
        self.kf_btn = QtWidgets.QPushButton("Tag pixel-centre x/y…")
        self.kf_btn.setToolTip("Add low-noise tag x/y (metres) from the detection pixel centre.\n"
                               "No smoothing or filtering: every frequency the camera sees is kept.\n"
                               "Adds blue translation.x_px / y_px fields inside each tf topic.")
        self.kf_btn.clicked.connect(self.open_kalman_dialog)
        self.imu_check = QtWidgets.QCheckBox("Include IMU velocity", checked=True)
        self.imu_check.setToolTip(
            "The IMU is integrated from Start to Stop (gravity removed) and shown in blue in the field\n"
            "list as derived/imu_vel (forward_speed = speed along the heading, forward_distance...).\n"
            "Tick this to add those columns to the exported CSV.")
        self.imu_mode = QtWidgets.QComboBox()
        self.imu_mode.addItems(["ROV at rest at Start & Stop", "odometry velocity (GPS/DVL)"])
        self.imu_mode.setToolTip(
            "At rest: velocity is 0 at Start and Stop; the velocity left over at Stop is treated as\n"
            "accelerometer bias and removed. Put Start and Stop on moments where the ROV is still.\n"
            "Anchor to odometry: only when global_position/local has a real velocity (GPS or DVL).")
        self.imu_mode.currentIndexChanged.connect(lambda *_: self._update_imu_velocity(rebuild=True))
        self.export_btn = QtWidgets.QPushButton("Export CSV…  (Ctrl+E)")
        self.export_btn.setStyleSheet("font-weight: 600; padding: 4px 14px;")
        self.export_btn.clicked.connect(self.export)

        self.imu_btn = QtWidgets.QPushButton("IMU velocity…")
        self.imu_btn.setToolTip("Open a window with the IMU acceleration, velocity and distance for Start→Stop.")
        self.imu_btn.clicked.connect(self.open_imu_window)

        def group(title, rows):
            box = QtWidgets.QGroupBox(title)
            v = QtWidgets.QVBoxLayout(box)
            v.setContentsMargins(8, 4, 8, 6)
            v.setSpacing(4)
            for widgets in rows:
                h = QtWidgets.QHBoxLayout()
                h.setSpacing(6)
                for w in widgets:
                    if w is None:
                        h.addStretch(1)
                    elif isinstance(w, str):
                        h.addWidget(QtWidgets.QLabel(w))
                    else:
                        h.addWidget(w)
                v.addLayout(h)
            return box

        g_win = group("Window (Start → Stop)", [
            ["Start", self.start_spin, b_s, "  Stop", self.stop_spin, b_e, None],
            [b_all, self.window_label, None],
        ])
        g_calc = group("Calculations", [
            [self.imu_btn, "IMU assumes:", self.imu_mode, None],
            [self.kf_btn, None],
            # Panel analysis is hidden for now (still available from the command line: --panels).
            # To show it again, add: [self.panel_btn, self.panel_check, None],
        ])
        g_exp = group("Export CSV", [
            [self.sync_mode, self.rate_spin, self.cols_mode, self.gap_spin, None],
            [self.imu_check, self.png_check, self.json_check, None, self.export_btn],
        ])
        for g in (g_win, g_calc, g_exp):
            bl.addWidget(g)
        bl.setStretch(0, 3)
        bl.setStretch(1, 2)
        bl.setStretch(2, 3)

        central = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(central)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(split, 1)
        line = QtWidgets.QFrame(frameShape=QtWidgets.QFrame.Shape.HLine)
        cv.addWidget(line)
        cv.addWidget(bottom)
        self.setCentralWidget(central)

        self.progress = QtWidgets.QProgressBar(maximum=1000, maximumWidth=260, visible=False)
        self.statusBar().addPermanentWidget(self.progress)
        self._set_enabled(False)

    def _build_shortcuts(self):
        def sc(key, fn):
            s = QShortcut(QtGui.QKeySequence(key), self)
            s.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            s.activated.connect(fn)
        sc("Right", lambda: self.step(1))
        sc("Left", lambda: self.step(-1))
        sc("Shift+Right", lambda: self.step(10))
        sc("Shift+Left", lambda: self.step(-10))
        sc("Space", self.toggle_play)
        sc("[", lambda: self.set_window(self.cursor_x(), None))
        sc("]", lambda: self.set_window(None, self.cursor_x()))
        sc("Ctrl+E", self.export)
        sc("Ctrl+O", self.choose_bag)
        sc("Escape", lambda: self._set_solo(None))

    def _set_enabled(self, on: bool):
        for w in (self.tree, self.slider, self.b_prev, self.b_play, self.b_next, self.export_btn,
                  self.start_spin, self.stop_spin):
            w.setEnabled(on)

    # ----------------------------------------------------------- loading ---
    def choose_bag(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open ROS 2 bag folder (contains metadata.yaml)")
        if d:
            self.open_bag(d)

    def open_bag(self, path, rebuild=False):
        if self._worker and self._worker.isRunning():
            return
        self.stop_play()
        self._set_enabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.statusBar().showMessage(f"Opening {path} …")
        self._worker = Worker(load_bag, path, camera_topic=self.args.camera, rebuild=rebuild,
                              msg_dir=self.args.msg_dir, include_covariance=self.args.include_covariance)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, frac, text):
        self.progress.setValue(int(frac * 1000))
        self.statusBar().showMessage(text)

    def _on_failed(self, msg):
        self.progress.setVisible(False)
        self._set_enabled(self.data is not None)
        QtWidgets.QMessageBox.critical(self, "Could not open bag", msg)

    def _on_loaded(self, data: BagData):
        if self.data:
            self.data.close()
        self.data = data
        self._thumb_cache.clear()
        self.progress.setVisible(False)
        fi = data.meta.get("frame_info", {})
        self.bag_label.setText(
            f"  {data.bag_dir.name}   ·   {data.camera_topic or 'no camera'}  "
            f"{data.n_frames} frames @ {fi.get('rate_hz', 0):.1f} Hz")
        self.setWindowTitle(f"Bag viewer — {data.bag_dir.name} — {VERSION}")
        warn = data.meta.get("warnings", []) + [f"skipped {k}: {v}" for k, v in data.meta.get("skipped", {}).items()]
        self.statusBar().showMessage(
            ("Loaded.  " + ("  |  ".join(warn) if warn else "No warnings.")), 15000)

        if data.n_frames:
            self._t_end = float(data.rel(data.frame_t[-1]))
        else:
            ends = [s.t[-1] for s in data.series.values() if len(s.t)]
            self._t_end = float(data.rel(max(ends))) if ends else 0.0
        for sp in (self.start_spin, self.stop_spin):
            sp.blockSignals(True)
            sp.setRange(0.0, self._t_end)
            sp.blockSignals(False)
        self._imu_series = None
        self._solo = None
        self.rectify.blockSignals(True)
        self.rectify.setChecked(data.has_distortion)
        self.rectify.setEnabled(data.meta.get("camera_info") is not None)
        self.rectify.blockSignals(False)
        self._compute_thrusters()
        try:
            hdg = heading_series(data)
            if hdg is not None:
                data.series[hdg.name] = hdg
            self._derived_note("derived/heading", "ok" if hdg is not None else "not created - no IMU orientation")
        except Exception as e:  # noqa: BLE001
            self._derived_note("derived/heading", f"failed: {type(e).__name__}: {e}")
        self._populate_tree()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(data.n_frames - 1, 0))
        self.slider.blockSignals(False)
        if data.n_frames == 0:
            self.sync_mode.setCurrentIndex(1)
        self._set_enabled(True)
        self._rebuild_plots()
        self._full_window()
        self._update_imu_velocity()
        self.frame = -1
        self.set_frame(0, move_cursor=True)
        self.img_view.autoRange(padding=0)

    def show_info(self):
        if not self.data:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Bag info")
        dlg.resize(980, 620)
        v = QtWidgets.QVBoxLayout(dlg)
        t = QtWidgets.QPlainTextEdit(readOnly=True)
        t.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        text = bag_summary(self.data)
        self.data.tags_in_frame(0)
        off = getattr(self.data, "tag_stamp_offset_ms", None)
        if off is not None:
            text += f"\nAprilTag detections vs camera frame stamps: median offset {off:.2f} ms"
        text += "\n\nCalculated (post-processing) entries:\n" + "\n".join(
            f"  {k}: {v}" for k, v in self._derived_status.items()) if self._derived_status else ""
        ci = self.data.meta.get("camera_info")
        if ci:
            text += f"\n\nCamera info ({ci['topic']}): {ci['width']}x{ci['height']} {ci['distortion_model']}\n" \
                    f"  K = {ci['k']}\n  D = {ci['d']}\n  R = {ci.get('r')}\n  P = {ci['p']}"
        for tf in self.data.meta.get("tf_static", []):
            text += f"\n/tf_static {tf['parent']} -> {tf['child']}: t={tf['translation']} q={tf['rotation_xyzw']}"
        t.setPlainText(text)
        v.addWidget(t)
        dlg.exec()

    # -------------------------------------------------------------- tree ---
    def _populate_tree(self):
        d = self.data
        prev_ticks = set(self.ticked()) if self.tree.topLevelItemCount() else None
        prev_series = {self.tree.topLevelItem(i).data(0, QtCore.Qt.ItemDataRole.UserRole)
                       for i in range(self.tree.topLevelItemCount())}
        self.tree.blockSignals(True)
        self.tree.clear()
        pats = [re.compile(p) for p in DEFAULT_PLOTS]
        blue = QtGui.QBrush(QtGui.QColor(DERIVED_COLOR))
        green = QtGui.QBrush(QtGui.QColor(NATIVE_COLOR))
        sep_widgets = []
        order = sorted(d.series.values(), key=lambda s: (s.msgtype != "derived", s.short))
        for s in order:
            derived = s.msgtype == "derived"
            top = QtWidgets.QTreeWidgetItem([s.short, "", "calculated (Start → Stop)" if s.name == DERIVED_IMU else
                                             ("calculated" if derived else
                                              f"{len(s.t)} · {s.rate_hz:.1f} Hz · {s.time_source}")])
            top.setFlags(top.flags() | QtCore.Qt.ItemFlag.ItemIsAutoTristate | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            top.setData(0, QtCore.Qt.ItemDataRole.UserRole, s.name)
            tip = (f"{s.short}: post-processing calculation (not a recorded topic)" if derived
                   else f"{s.name}\n{s.msgtype}\nstamps: {s.time_source}")
            top.setToolTip(0, tip)
            top.setForeground(0, blue if derived else green)
            top.setForeground(2, blue if derived else green)
            sep_widgets.append((top, s.name))
            keep = prev_ticks is not None and s.name in prev_series
            for c in s.cols:
                kind = s.kinds.get(c, "float")
                it = QtWidgets.QTreeWidgetItem([c, "", kind + (" (derived)" if c in s.derived else "")])
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                on = ((s.name, c) in prev_ticks) if keep else any(p.search(f"{s.short}.{c}") for p in pats)
                it.setCheckState(0, QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked)
                it.setData(0, QtCore.Qt.ItemDataRole.UserRole, (s.name, c))
                calc_col = derived or c in s.info.get("calc_cols", ())
                it.setForeground(0, blue if calc_col else green)
                it.setForeground(2, blue if calc_col else green)
                if calc_col:
                    it.setToolTip(0, tip if derived else f"{c}: post-processing calculation (pixel-centre position, unfiltered)")
                top.addChild(it)
            self.tree.addTopLevelItem(top)
        self.tree.setColumnWidth(1, 64)
        self.tree.setColumnWidth(2, 150)
        for top, name in sep_widgets:
            cb = QtWidgets.QCheckBox()
            cb.setChecked(self._separate.get(name, False))
            cb.setToolTip("Give every field of this topic its own plot")
            cb.toggled.connect(lambda on, n=name: (self._separate.__setitem__(n, on), self._replot_timer.start()))
            holder = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(cb, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.tree.setItemWidget(top, 1, holder)
        self.tree.blockSignals(False)
        self._apply_filter(self.filter_edit.text())

    def _apply_filter(self, text):
        words = text.lower().split()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            any_vis = False
            for j in range(top.childCount()):
                ch = top.child(j)
                hay = f"{top.text(0)}.{ch.text(0)}".lower()
                vis = all(w in hay for w in words)
                ch.setHidden(not vis)
                any_vis |= vis
            top.setHidden(not any_vis)
            top.setExpanded(bool(words) and any_vis)

    def _untick_all(self):
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        self.tree.blockSignals(False)
        self._rebuild_plots()

    def ticked(self) -> list[tuple[str, str]]:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                ch = top.child(j)
                if ch.checkState(0) == QtCore.Qt.CheckState.Checked:
                    out.append(ch.data(0, QtCore.Qt.ItemDataRole.UserRole))
        return out

    # ------------------------------------------------------------- plots ---
    @staticmethod
    def _field_group(s, col: str) -> str:
        if col in s.derived:
            return s.derived[col]["quat"] + " → roll/pitch/yaw"
        base = re.sub(r"\[\d+\]$", "", col)
        return base.rsplit(".", 1)[0] if "." in base else ""

    @staticmethod
    def _unit(field: str, topic: str = "") -> str:
        """Unit from the field name; the topic name is only used when the field says nothing
        (so 'global_position/local.pose.orientation' is not mistaken for a position)."""
        for text in (field, f"{topic}.{field}"):
            for pat, u in UNITS:
                if re.search(pat, text):
                    return u
        return ""

    def _plot_specs(self):
        """[(series, [[col, ...] per plot])]. Separate topics get one plot per field;
        others share plots, one per field group and unit, so units never mix."""
        d = self.data
        groups: OrderedDict[str, list[str]] = OrderedDict()
        for name, col in self.ticked():
            ser = d.series.get(name)
            if ser is not None and col in ser.cols:
                groups.setdefault(name, []).append(col)
        out = []
        for name, cols in groups.items():
            ser = d.series[name]
            if self._separate.get(name, False):
                plots = [[c] for c in cols]
            else:
                by: OrderedDict[tuple, list[str]] = OrderedDict()
                for c in cols:
                    by.setdefault((self._field_group(ser, c), self._unit(c, ser.short)), []).append(c)
                plots = list(by.values())
            out.append((ser, plots))
        return out

    def _rebuild_plots(self):
        if not self.data:
            return
        d = self.data
        keep_range = self._plots[0].vb.viewRange()[0] if self._plots else None
        cur_x = self.cursor_x() if self._cursors else 0.0
        self.plot_widget.clear()
        self._plots, self._cursors, self._regions, self._plot_keys, self._imu_curves = [], [], [], [], []
        self._end_labels = []

        groups = self._plot_specs()                     # [(series, [[cols of plot 1], ...])]
        if self._solo is not None:
            groups = [(ser, [pc for pc in plots if (ser.name, tuple(pc)) == self._solo]) for ser, plots in groups]
            groups = [g for g in groups if g[1]]
            if not groups:
                self._set_solo(None, rebuild=False)
                groups = self._plot_specs()
        n_total = sum(len(p) for _, p in groups)
        if n_total > MAX_PLOTS:
            self.statusBar().showMessage(f"Showing the first {MAX_PLOTS} plots; untick some to see others.", 6000)
            left, capped = MAX_PLOTS, []
            for ser, plots in groups:
                if left <= 0:
                    break
                capped.append((ser, plots[:left]))
                left -= len(plots[:left])
            groups, n_total = capped, MAX_PLOTS
        h = self.plot_h.value()

        first = None
        k_plot = 0
        for gi, (ser, plots) in enumerate(groups):
            derived = ser.msgtype == "derived"
            color = DERIVED_COLOR if derived else TOPIC_FRAME_COLOR
            frame = TopicFrame(color)
            self.plot_widget.addItem(frame, row=gi, col=0)
            shrinkable(frame.addLabel(f"<b>{ser.short}</b>" + ("  (calculated)" if derived else ""),
                                      row=0, col=0, color=color, size="9pt"))
            frame.setCacheMode(DEVICE_CACHE)
            x = d.rel(ser.t)
            for j, pcols in enumerate(plots):
                k_plot += 1
                grp = self._field_group(ser, pcols[0])
                unit = self._unit(pcols[0], ser.short)
                unit = "" if unit == "quaternion" else unit
                calc_plot = all(c in ser.info.get("calc_cols", ()) for c in pcols)
                if len(pcols) == 1:
                    label = pcols[0]
                else:
                    label = grp or ", ".join(pcols)
                tcol = f"color:{DERIVED_COLOR};" if calc_plot and not derived else ""
                p = frame.addPlot(row=j + 1, col=0, viewBox=ZoomViewBox())
                if not self._solo:
                    p.setMinimumHeight(h)
                    p.setMaximumHeight(h)
                p._fixed_y = next((rng for pat, rng in Y_RANGES
                                   if all(re.search(pat, f"{ser.short}.{c}") for c in pcols)), None)
                p.autoBtn.clicked.connect(lambda *_, vb=p.vb: self._refit(vb))
                knote = ""
                shrinkable(p.titleLabel)
                p.setTitle(f"<span style='font-size:8pt;{tcol}'>{label}" + (f"  [{unit}]" if unit else "")
                           + ("  (calculated)" if calc_plot and not derived else "") + "</span>" + knote, justify="left")
                p.showGrid(x=True, y=True, alpha=0.18)
                p.setMenuEnabled(False)
                p.getAxis("left").setWidth(70)
                p.getAxis("left").enableAutoSIPrefix(False)
                if unit:
                    p.getAxis("left").setLabel(unit, **{"font-size": "9pt", "color": "#c9ccd1"})
                p.setDownsampling(auto=True, mode="peak")
                p.setClipToView(True)
                multi = len(pcols) > 1
                if multi:
                    p.addLegend(offset=(6, 4), labelTextSize="8pt", verSpacing=-6, brush=(27, 29, 33, 170))
                for k, col in enumerate(pcols):
                    v = ser.cols[col]
                    if col in ser.derived:
                        short = ser.derived[col]["part"]
                    elif grp and col.startswith(grp + "."):
                        short = col[len(grp) + 1:]
                    else:
                        short = col
                    line_color = LINE_COLORS[(k if multi else j) % len(LINE_COLORS)]
                    pen = pg.mkPen(line_color, width=1.3)
                    if ser.kinds.get(col) == "str":
                        p._categorical = True
                        cats = list(dict.fromkeys(v.tolist()))
                        y = np.array([cats.index(z) for z in v.tolist()], float)
                        p.getAxis("left").setTicks([[(i, c[:10]) for i, c in enumerate(cats)]])
                        if len(x):
                            p.plot(x, y, pen=pen, stepMode="left", name=short if multi else None)
                        continue
                    step = ser.kinds.get(col) in ("int", "bool")
                    item = p.plot(x, v, pen=pen, connect="finite", name=short if multi else None,
                                  stepMode="left" if step and len(x) > 1 else None)
                    if self.end_labels.isChecked() and len(x) and (unit or multi):
                        ti = pg.TextItem(f"{short} {unit}".strip() if multi else unit, color=line_color, anchor=(1, 1))
                        ti.setFont(QtGui.QFont("", 8))
                        p.addItem(ti, ignoreBounds=True)
                        self._end_labels.append((p, ti, x, v, item))
                    if ser.name == DERIVED_IMU:
                        self._imu_curves.append((item, col))
                if first is None:
                    first = p
                else:
                    p.setXLink(first)
                region = pg.LinearRegionItem(brush=REGION_BRUSH, pen=pg.mkPen((78, 156, 245, 200), width=1.5),
                                             movable=True, bounds=(0.0, self._t_end))
                region.setZValue(-10)
                region.sigRegionChanged.connect(self._region_moved)
                p.addItem(region, ignoreBounds=True)
                cur = pg.InfiniteLine(pos=cur_x, angle=90, movable=True, pen=CURSOR_PEN,
                                      hoverPen=pg.mkPen("#fff1a8", width=3))
                cur.sigPositionChanged.connect(self._cursor_dragged)
                p.addItem(cur, ignoreBounds=True)
                # Moving the cursor repaints a thin strip of every plot. Caching the static
                # items as pixmaps makes that a blit instead of re-drawing text and lines.
                for item in [p.getAxis("left"), p.getAxis("bottom"), region] + [c.curve for c in p.listDataItems()]:
                    item.setCacheMode(DEVICE_CACHE)
                self._plots.append(p)
                self._cursors.append(cur)
                self._regions.append(region)
                self._plot_keys.append((ser.name, tuple(pcols)))
                if k_plot < n_total:
                    p.getAxis("bottom").setStyle(showValues=False)
        if self._plots:
            self._plots[-1].setLabel("bottom", "time since first camera frame [s]")
            self.plot_widget.setToolTip("Scroll: zoom this plot's y · Ctrl+scroll or scroll on a time axis: zoom time\n"
                                        "Shift+drag: pan y · 'A' button: refit · double-click: full view (Esc to go back)")
            self.plot_widget.setMinimumHeight(0 if self._solo else len(self._plots) * (h + 12) + len(groups) * 50 + 40)
            # never let the splitter squeeze the plots narrower than they can draw (they would
            # otherwise overflow under the camera panel)
            QtCore.QTimer.singleShot(0, self._lock_plot_min_width)
            self._set_regions(self.start_spin.value(), self.stop_spin.value())
            # One explicit, shared time range: no plot auto-fits x to its own data (that made the
            # linked plots fight and jump when resizing or when a calculated plot updated).
            for p in self._plots:
                p.enableAutoRange(x=False)
                p.setLimits(xMin=-1.0, xMax=self._t_end + 1.0)
            lo_x, hi_x = keep_range if keep_range else (0.0, self._t_end)
            first.setXRange(lo_x, hi_x, padding=0)
            first.vb.sigXRangeChanged.connect(lambda *_: self._label_timer.start())
            self._label_timer.start()
        else:
            self.plot_widget.setMinimumHeight(0)
            self.plot_widget.addLabel("Tick fields on the left to plot them.", row=0, col=0)

    def _lock_plot_min_width(self):
        try:
            mw = int(np.ceil(self.plot_widget.ci.minimumWidth())) + 4
        except RuntimeError:
            return
        self.plot_widget.setMinimumWidth(max(200, mw))

    def _refit(self, vb):
        """'A' button: show the whole bag in time and refit this plot's y."""
        vb.manual_y = False
        if self._plots:
            self._plots[0].setXRange(0.0, self._t_end, padding=0)
            for p in self._plots:
                p.enableAutoRange(x=False)
        self._fit_y()

    def _on_view_changed(self):
        self._fit_y()
        self._place_end_labels()

    def _fit_y(self):
        """Fit each plot's y axis to the min/max of its data inside the visible time range."""
        for p in self._plots:
            if getattr(p, "_categorical", False) or getattr(p.vb, "manual_y", False):
                continue
            if getattr(p, "_fixed_y", None):
                p.enableAutoRange(y=False)
                p.setYRange(*p._fixed_y, padding=0.02)
                continue
            lo, hi = p.vb.viewRange()[0]
            ymin, ymax = np.inf, -np.inf
            for it in p.listDataItems():
                x, y = it.xData, it.yData
                if x is None or y is None or len(x) == 0:
                    continue
                i0 = max(int(np.searchsorted(x, lo)) - 1, 0)
                i1 = int(np.searchsorted(x, hi, side="right")) + 1
                seg = y[i0:i1]
                seg = seg[np.isfinite(seg)]
                if len(seg):
                    ymin, ymax = min(ymin, float(seg.min())), max(ymax, float(seg.max()))
            if not np.isfinite(ymin):
                continue
            if ymax - ymin < 1e-9:
                pad = max(abs(ymax) * 0.05, 0.5)
                ymin, ymax = ymin - pad, ymax + pad
            p.enableAutoRange(y=False)
            p.setYRange(ymin, ymax, padding=0.06)

    def _place_end_labels(self):
        """Put each line's name + unit at the right end of the visible part of that line."""
        for p, ti, x, v, item in self._end_labels:
            if item in self._imu_curves_items():
                x, v = item.xData, item.yData
            if x is None or not len(x):
                ti.setVisible(False)
                continue
            right = min(p.vb.viewRange()[0][1], float(x[-1]))
            i = int(np.searchsorted(x, right, side="right")) - 1
            while i >= 0 and not np.isfinite(v[i]):
                i -= 1
            if i < 0:
                ti.setVisible(False)
                continue
            (y0, y1) = p.vb.viewRange()[1]
            yv = float(np.clip(v[i], y0 + 0.12 * (y1 - y0), y1))
            ti.setPos(float(x[i]), yv)
            ti.setVisible(True)
        # nudge labels in the same plot apart so they don't print on top of each other
        by_plot = {}
        for p, ti, *_ in self._end_labels:
            if ti.isVisible():
                by_plot.setdefault(id(p), (p, []))[1].append(ti)
        for p, tis in by_plot.values():
            vb = p.vb
            pts = sorted(((vb.mapViewToScene(ti.pos()), ti) for ti in tis), key=lambda a: a[0].y())
            gap = 12.0
            last = -1e9
            for sp, ti in pts:
                y = max(sp.y(), last + gap)
                last = y
                if y != sp.y():
                    ti.setPos(vb.mapSceneToView(QtCore.QPointF(sp.x(), y)))

    def _imu_curves_items(self):
        return [it for it, _ in self._imu_curves]

    def _set_solo(self, key, rebuild=True):
        """Full view of one plot: hide the field list, camera and other plots. key=None restores."""
        if key is not None and self._solo is None:
            self._split_sizes = self.split.sizes()
            self.left_panel.hide()
            self.right_panel.hide()
        elif key is None and self._solo is not None:
            self.left_panel.show()
            self.right_panel.show()
            if self._split_sizes:
                self.split.setSizes(self._split_sizes)
        elif key is None:
            return
        self._solo = key
        if rebuild:
            self._rebuild_plots()

    def _plot_clicked(self, ev):
        if not self._plots or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        for p, key in zip(self._plots, self._plot_keys):
            if p.sceneBoundingRect().contains(ev.scenePos()):
                if ev.double():
                    self._set_solo(None if self._solo else key)
                    return
                x = p.vb.mapSceneToView(ev.scenePos()).x()
                self.set_cursor(x)
                self.set_frame(self.data.frame_at(x), move_cursor=False)
                return

    # ------------------------------------------------------- IMU velocity ---
    def _imu_mode_name(self) -> str:
        return "rest" if self.imu_mode.currentIndex() == 0 else "odom"

    def _derived_note(self, key, text):
        self._derived_status[key] = text
        print(f"[bagsync] {key}: {text}", flush=True)

    def _compute_thrusters(self):
        """Add the blue derived/thrusters entry. Never lets a failure stop the bag from loading."""
        d = self.data
        try:
            thr = thruster_series(d, self.thruster_channels)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            thr = None
            self._derived_note("derived/thrusters", f"failed: {type(e).__name__}: {e}")
        else:
            if thr is None:
                rc = [s.name for s in d.series.values() if "rc/out" in s.name]
                self._derived_note("derived/thrusters", "not created - " + (
                    f"{rc[0]} has none of channels {self.thruster_channels}" if rc else "no rc/out topic in this bag"))
            else:
                self._derived_note("derived/thrusters", f"ok, channels {thr.info['channels']} of {thr.info['source']}")
        if thr is not None:
            d.series[thr.name] = thr
        else:
            d.series.pop(DERIVED_THR, None)
        return thr

    def open_kalman_dialog(self):
        """Pixel-centre tag x/y. Deliberately no smoothing: nothing is low-passed."""
        d = self.data
        if not d:
            return
        def tag_id(name):
            m = re.search(r"(\d+)\]?$", name)
            return int(m.group(1)) if m else None
        tfs = sorted((s for s in d.series.values() if s.info.get("kind") == "tf"
                      and tag_id(s.name) not in REFERENCE_TAGS), key=lambda s: s.short)
        if not tfs:
            QtWidgets.QMessageBox.information(self, "No tf", "This bag has no /tf tag transforms.")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Tag pixel-centre x/y")
        lay = QtWidgets.QVBoxLayout(dlg)
        lab = QtWidgets.QLabel(
            "Adds translation.x_px / y_px (metres, camera optical frame) to each ticked tf topic:\n"
            "    x = (u − cx)·z/fx,   y = (v − cy)·z/fy\n"
            "u, v = the tag's detected pixel centre, z = the tag's median pose depth (a constant).\n"
            "No smoothing or filtering of any kind, so every frequency the camera captures is kept\n"
            "(up to half the frame rate). Noise is ~4–6 µm instead of 2–41 µm from the pose x/y.\n"
            "Uses P (rectified) when 'Rectify image' is ticked, otherwise K.\n"
            f"Reference tags {', '.join(map(str, REFERENCE_TAGS))} (fixed to the camera) are not listed.")
        lay.addWidget(lab)
        lst = QtWidgets.QListWidget()
        for s in tfs:
            it = QtWidgets.QListWidgetItem(s.short)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, s.name)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            on = "tag_18" in s.name or "tag18" in s.name or "pixel_position" in s.info
            it.setCheckState(QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked)
            lst.addItem(it)
        lay.addWidget(lst, 1)
        row = QtWidgets.QHBoxLayout()
        for txt, st in (("All", QtCore.Qt.CheckState.Checked), ("None", QtCore.Qt.CheckState.Unchecked)):
            b_ = QtWidgets.QPushButton(txt)
            b_.clicked.connect(lambda *_, st=st: [lst.item(i).setCheckState(st) for i in range(lst.count())])
            row.addWidget(b_)
        row.addStretch(1)
        lay.addLayout(row)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                        | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.resize(560, 480)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        chosen = [lst.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(lst.count())
                  if lst.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        if not chosen:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        msgs, done = [], []
        try:
            for name in chosen:
                ser = d.series[name]
                for c in ser.info.get("calc_cols", []):          # start fresh
                    ser.cols.pop(c, None)
                    ser.kinds.pop(c, None)
                ser.info["calc_cols"] = []
                ser.info.pop("kalman", None)
                ser.info.pop("pixel_position", None)
                pp = tag_pixel_position(d, ser, rectified=self.rectify.isChecked())
                if pp:
                    done.append(name)
                    msgs.append(f"{ser.short}: {pp['matched']}/{pp['of']} frames, {pp['intrinsics']}")
                else:
                    msgs.append(f"{ser.short}: no matching detection or camera_info")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        for n in done:
            self._separate[n] = True
        self._populate_tree()
        self._tick([(n, f"translation.{a}_px") for n in done for a in "xy"])
        self.statusBar().showMessage("Pixel-centre x/y (unfiltered): " + "; ".join(msgs), 20000)

    def _tick(self, pairs):
        want = set(pairs)
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                ch = top.child(j)
                if ch.data(0, QtCore.Qt.ItemDataRole.UserRole) in want:
                    ch.setCheckState(0, QtCore.Qt.CheckState.Checked)
        self.tree.blockSignals(False)
        self._rebuild_plots()

    def set_thruster_channels(self, chans):
        self.thruster_channels = list(chans)
        if not self.data:
            return
        had = DERIVED_THR in self.data.series
        thr = self._compute_thrusters()
        if had != (thr is not None):
            self._populate_tree()
        self._rebuild_plots()
        self._update_imu_velocity(rebuild=True)

    def open_panel_window(self):
        if not self.data:
            return
        if self._panel_win is None:
            self._panel_win = PanelWindow(self)
        self._panel_win.show()
        self._panel_win.raise_()
        self._update_panels(force=True)

    def _update_panels(self, force=False):
        """Recompute the panel signals for Start→Stop while the panel window is open."""
        if not self.data or not (force or (self._panel_win is not None and self._panel_win.isVisible())):
            return
        lo, hi = self.start_spin.value(), self.stop_spin.value()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            ser = panel_analysis(self.data, lo, hi, REFERENCE_TAGS)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ser = None
            self._derived_note("derived/panels", f"failed: {type(e).__name__}: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        had = DERIVED_PANEL in self.data.series
        if ser is None:
            self.data.series.pop(DERIVED_PANEL, None)
        else:
            self.data.series[DERIVED_PANEL] = ser
        if had != (ser is not None):
            self._populate_tree()
        self._rebuild_plots() if any(n == DERIVED_PANEL for n, _ in self.ticked()) else None
        if self._panel_win is not None and self._panel_win.isVisible():
            self._panel_win.refresh(ser)

    def open_imu_window(self):
        if not self.data:
            return
        if self._imu_win is None:
            self._imu_win = ImuWindow(self)
        self._imu_win.show()
        self._imu_win.raise_()
        self._update_imu_velocity()

    def _imu_win_visible(self) -> bool:
        return self._imu_win is not None and self._imu_win.isVisible()

    def _update_imu_velocity(self, rebuild=False):
        """Re-integrate the IMU for the current Start/Stop. The result is the blue derived/imu_vel entry."""
        if not self.data:
            return
        lo, hi = self.start_spin.value(), self.stop_spin.value()
        had = self._imu_series is not None
        try:
            new = integrate_imu(self.data, lo, hi, self._imu_mode_name())
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            new = None
            self._derived_note("derived/imu_vel", f"failed: {type(e).__name__}: {e}")
        else:
            self._derived_note("derived/imu_vel", "ok" if new is not None else
                               "not created - no IMU topic with orientation + linear_acceleration in the window")
        self._imu_series = new
        if new is None:
            self.statusBar().showMessage("IMU velocity: no IMU with orientation and linear_acceleration in the "
                                         "window" + (", or no odometry velocity at Start/Stop"
                                                     if self._imu_mode_name() == "odom" else "") + ".", 8000)
        else:
            self.statusBar().showMessage(imu_summary(new.info), 20000)
        if had != (new is not None):
            self._populate_tree()
            self._rebuild_plots()
        elif rebuild:
            self._rebuild_plots()
        elif new is not None:
            x = self.data.rel(new.t)
            for item, col in self._imu_curves:
                if col in new.cols:
                    item.setData(x, new.cols[col])
            self._label_timer.start()
        if self._imu_win_visible():
            self._imu_win.refresh()

    # ------------------------------------------------------------ cursor ---
    def cursor_x(self) -> float:
        if self._cursors:
            return float(self._cursors[0].value())
        if self.data and self.frame >= 0:
            return float(self.data.rel(self.data.frame_t[self.frame]))
        return 0.0

    def set_cursor(self, x):
        self._syncing = True
        for c in self._cursors:
            c.setValue(x)
        self._syncing = False

    def _cursor_dragged(self, line):
        if self._syncing:
            return
        x = float(line.value())
        self.set_cursor(x)
        self._pending_x = x
        if not self._cursor_timer.isActive():
            self._cursor_timer.start()

    def _apply_pending_cursor(self):
        if self._pending_x is not None and self.data:
            self.set_frame(self.data.frame_at(self._pending_x), move_cursor=False)
            self._pending_x = None

    # ------------------------------------------------------------ frames ---
    def step(self, n):
        if self.data and self.data.n_frames:
            self.set_frame(int(np.clip(self.frame + n, 0, self.data.n_frames - 1)), move_cursor=True)

    def _thumb_rgb(self, i):
        img = self._thumb_cache.get(i)
        if img is None:
            img = self.data.thumb(i)
            if img is not None and cv2 is not None:
                if self.rectify.isChecked():
                    maps = self.data.undistort_maps((img.shape[1], img.shape[0]))
                    if maps is not None:
                        img = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._thumb_cache[i] = img
            if len(self._thumb_cache) > 200:
                self._thumb_cache.popitem(last=False)
        else:
            self._thumb_cache.move_to_end(i)
        return img

    def _rectify_toggled(self, *_):
        self._thumb_cache.clear()
        if self.data and self.frame >= 0:
            i, self.frame = self.frame, -1
            self.set_frame(i, move_cursor=False)

    def set_frame(self, i, move_cursor=True):
        d = self.data
        if not d or d.n_frames == 0 or i < 0:
            if d and d.n_frames == 0:
                self.frame_label.setText("No camera topic - rows use a fixed rate.")
            return
        i = int(np.clip(i, 0, d.n_frames - 1))
        if i != self.frame:
            self.frame = i
            img = self._thumb_rgb(i)
            if img is not None:
                self.img_item.setImage(img, autoLevels=False, levels=(0, 255))
            self._draw_tags()
            self.slider.blockSignals(True)
            self.slider.setValue(i)
            self.slider.blockSignals(False)
            t = int(d.frame_t[i])
            tags = d.tags_in_frame(i)
            self.frame_label.setText(
                f"Frame {i} / {d.n_frames - 1}    t = {(t - d.t0) / 1e9:.3f} s    "
                f"stamp {t // 1_000_000_000}.{t % 1_000_000_000:09d}"
                + (f"    tags: {', '.join(str(k) for k, _ in tags)}" if tags else ""))
        if move_cursor:
            self.set_cursor(float(d.rel(d.frame_t[i])))

    def _draw_tags(self):
        for t in self._tag_texts:
            t.setVisible(False)
        if not (self.data and self.show_tags.isChecked() and self.frame >= 0):
            self.tag_curve.setData([], [])
            return
        tags = self.data.tags_in_frame(self.frame)
        sc = self.data.thumb_scale
        xs, ys = [], []
        for n, (tid, c) in enumerate(tags):
            poly = np.vstack([c, c[:1]]) * sc
            xs += list(poly[:, 0]) + [np.nan]
            ys += list(poly[:, 1]) + [np.nan]
            if n >= len(self._tag_texts):
                ti = pg.TextItem(color="#39ff7a", anchor=(0.5, 0.5))
                ti.setFont(QtGui.QFont("", 11, QtGui.QFont.Weight.Bold))
                self.img_view.addItem(ti)
                self._tag_texts.append(ti)
            ti = self._tag_texts[n]
            ti.setText(str(tid))
            ti.setPos(*(c.mean(axis=0) * sc))
            ti.setVisible(True)
        self.tag_curve.setData(np.array(xs, float), np.array(ys, float))

    # -------------------------------------------------------------- play ---
    def toggle_play(self):
        if self._playing:
            self.stop_play()
        elif self.data and self.data.n_frames:
            if self.frame >= self.data.n_frames - 1:
                self.set_frame(0)
            self._playing = True
            self._play_t0 = time.perf_counter()
            self._play_x0 = float(self.data.rel(self.data.frame_t[self.frame]))
            self.b_play.setText("Pause")
            self._play_timer.start()

    def stop_play(self):
        was = self._playing
        self._playing = False
        self._play_timer.stop()
        self.b_play.setText("Play")
        if was and self.data and self.frame >= 0:
            self.set_cursor(float(self.data.rel(self.data.frame_t[self.frame])))

    def _play_tick(self):
        speed = float(self.speed.currentText().rstrip("×"))
        x = self._play_x0 + (time.perf_counter() - self._play_t0) * speed
        i = self.data.frame_at(x)
        # The image updates every tick; the plot cursors (which repaint every plot) only ~10x/s,
        # so playback stays smooth even with many plots open.
        now = time.perf_counter()
        move = now - getattr(self, "_last_cursor_move", 0.0) > 0.1
        self.set_frame(i, move_cursor=move)
        if move:
            self._last_cursor_move = now
        if i >= self.data.n_frames - 1:
            self.stop_play()

    # ------------------------------------------------------------ window ---
    def _full_window(self):
        d = self.data
        if not d:
            return
        self.set_window(0.0, self._t_end)

    def set_window(self, lo, hi):
        lo = self.start_spin.value() if lo is None else lo
        hi = self.stop_spin.value() if hi is None else hi
        if hi < lo:
            lo, hi = hi, lo
        lo = float(np.clip(lo, 0.0, self._t_end))
        hi = float(np.clip(hi, 0.0, self._t_end))
        for s, v in ((self.start_spin, lo), (self.stop_spin, hi)):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self._set_regions(lo, hi)
        self._update_window_label()
        self._imu_timer.start()

    def _set_regions(self, lo, hi):
        self._syncing = True
        for r in self._regions:
            r.setRegion((lo, hi))
        self._syncing = False

    def _region_moved(self, region):
        if self._syncing:
            return
        lo, hi = region.getRegion()
        lo, hi = max(lo, 0.0), min(hi, self._t_end)
        self._syncing = True
        for r in self._regions:
            if r is not region:
                r.setRegion((lo, hi))
        self._syncing = False
        for s, v in ((self.start_spin, lo), (self.stop_spin, hi)):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self._update_window_label()
        self._imu_timer.start()

    def _spins_changed(self, *_):
        self.set_window(self.start_spin.value(), self.stop_spin.value())

    def _update_window_label(self):
        d = self.data
        if not d:
            return
        lo, hi = self.start_spin.value(), self.stop_spin.value()
        if self.sync_mode.currentIndex() == 0 and d.n_frames:
            ft = d.rel(d.frame_t)
            n = int(np.sum((ft >= lo - 1e-6) & (ft <= hi + 1e-6)))
        else:
            n = int(np.floor((hi - lo) * self.rate_spin.value())) + 1
        self.window_label.setText(f"  {hi - lo:.3f} s  ·  {n} rows")

    # ------------------------------------------------------------ export ---
    def export(self):
        d = self.data
        if not d:
            return
        lo, hi = self.start_spin.value(), self.stop_spin.value()
        default = d.bag_dir.parent / f"{d.bag_dir.name}_{lo:.2f}-{hi:.2f}s.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export synced CSV", str(default), "CSV (*.csv)")
        if not path:
            return
        cols = None
        if self.cols_mode.currentIndex() == 1:
            cols = self.ticked()
            if not cols:
                QtWidgets.QMessageBox.warning(self, "Nothing ticked", "Tick some fields, or export all fields.")
                return
        rate = self.rate_spin.value() if self.sync_mode.currentIndex() == 1 else None
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        live = d.series.pop(DERIVED_IMU, None)       # the plotted result; restored afterwards
        live_panel = d.series.pop(DERIVED_PANEL, None)
        imu = integrate_imu(d, lo, hi, self._imu_mode_name()) if self.imu_check.isChecked() else None
        panel = panel_analysis(d, lo, hi, REFERENCE_TAGS) if self.panel_check.isChecked() else None
        try:
            if imu is not None:
                d.series[DERIVED_IMU] = imu
                if cols is not None:
                    cols = [c for c in cols if c[0] != DERIVED_IMU] + [(DERIVED_IMU, c) for c in imu.cols]
            elif cols is not None:
                cols = [c for c in cols if c[0] != DERIVED_IMU]
            if cols is not None:
                cols = [c for c in cols if c[0] != DERIVED_PANEL]
            if panel is not None:
                d.series[DERIVED_PANEL] = panel
                if cols is not None:
                    cols += [(DERIVED_PANEL, c) for c in panel.cols]
            elif self.panel_check.isChecked():
                self.statusBar().showMessage("Panel signals not added: need at least two panels of tags in the window.", 10000)
            res = sync(d, lo, hi, cols, self.gap_spin.value(), rate)
            if res.n_rows == 0:
                raise RuntimeError("No rows inside that window.")
            write_csv(res, path, {"bag": str(d.bag_dir), "camera_topic": d.camera_topic,
                                  "gap_factor": self.gap_spin.value(), "fixed_rate_hz": rate,
                                  "warnings": d.meta.get("warnings", [])}, sidecar=self.json_check.isChecked())
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
            return
        finally:
            d.series.pop(DERIVED_IMU, None)
            if live is not None:
                d.series[DERIVED_IMU] = live
            d.series.pop(DERIVED_PANEL, None)
            if live_panel is not None:
                d.series[DERIVED_PANEL] = live_panel
        QtWidgets.QApplication.restoreOverrideCursor()
        msg = f"Wrote {Path(path).name}: {res.n_rows} rows × {len(res.names) + 3} columns"
        self.statusBar().showMessage(msg, 12000)

        if self.png_check.isChecked() and res.frame_idx is not None:
            out_dir = Path(path).with_suffix("")
            out_dir = out_dir.parent / f"{out_dir.name}_frames"
            self.progress.setVisible(True)
            self._set_enabled(False)
            self._worker = Worker(export_images, d, res, out_dir)
            self._worker.progress.connect(self._on_progress)
            self._worker.done.connect(lambda n: (self.progress.setVisible(False), self._set_enabled(True),
                                                 self.statusBar().showMessage(f"{msg}  ·  {n} frames → {out_dir}", 15000)))
            self._worker.failed.connect(self._on_failed)
            self._worker.start()

    def closeEvent(self, ev):
        if self.data:
            self.data.close()
        super().closeEvent(ev)


class PanelWindow(QtWidgets.QWidget):
    """Separate window: panel signals for Start→Stop, measured vs IMU-explained vs leftover."""

    SIGNALS = ["both.open_sym", "both.open_asym", "left.open", "right.open", "both.shift_x", "both.shift_y",
               "left.shift_x", "right.shift_x", "left.bend_y", "right.bend_y", "left.shift_y", "right.shift_y"]

    def __init__(self, viewer):
        super().__init__(viewer, QtCore.Qt.WindowType.Window)
        self.v = viewer
        self.setWindowTitle("Panel analysis")
        self.resize(1200, 900)
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        help_lbl = QtWidgets.QLabel(
            "<small><b>open</b>: outer column minus inner column along x, + = outer edge moves away from the image "
            "centre.<br><b>open_sym</b>: both panels opening together; <b>open_asym</b>: one opens while the other "
            "closes.<br><b>shift</b>: whole panel moving.<br>Grey = measured, <span style='color:#4e9cf5'>blue</span> "
            "= what the IMU predicts (fitted on other parts of the window), <span style='color:#ff8a3d'>orange</span> "
            "= leftover the IMU can't explain.</small>")
        help_lbl.setWordWrap(True)
        self.pick = QtWidgets.QListWidget()
        self.pick.setMaximumWidth(330)
        self.pick.itemChanged.connect(lambda *_: self.refresh(self.v.data.series.get(DERIVED_PANEL)))
        left.addWidget(self.summary)
        left.addWidget(QtWidgets.QLabel("Signals to plot:"))
        left.addWidget(self.pick, 1)
        left.addWidget(help_lbl)
        holder = QtWidgets.QWidget()
        holder.setLayout(left)
        holder.setMaximumWidth(360)
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(holder)
        lay.addWidget(self.glw, 1)
        self._picked = {"both.open_sym", "both.open_asym", "left.open", "right.open", "both.shift_x", "both.shift_y"}

    def refresh(self, ser):
        self.glw.clear()
        if ser is None:
            self.summary.setText("<b>No panel signals.</b> They need tag detections on two panels "
                                 "(left and right groups of tags) between Start and Stop.")
            return
        lo, hi = self.v.start_spin.value(), self.v.stop_spin.value()
        self.summary.setText(f"<b>Window</b> {lo:.2f} → {hi:.2f} s<br>" + panel_summary_html(ser))
        available = [n for n in self.SIGNALS if n in ser.cols]
        self.pick.blockSignals(True)
        if self.pick.count() != len(available):
            self.pick.clear()
            for n in available:
                it = QtWidgets.QListWidgetItem(n)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.CheckState.Checked if n in self._picked else QtCore.Qt.CheckState.Unchecked)
                self.pick.addItem(it)
        self._picked = {self.pick.item(i).text() for i in range(self.pick.count())
                        if self.pick.item(i).checkState() == QtCore.Qt.CheckState.Checked}
        self.pick.blockSignals(False)
        x = self.v.data.rel(ser.t)
        units = ser.info.get("units", "µm")
        first = None
        shown = [n for n in available if n in self._picked]
        for row, n in enumerate(shown):
            p = self.glw.addPlot(row=row, col=0)
            sc = ser.info.get("scores", {}).get(n, {})
            p.setTitle(f"<span style='font-size:9pt'>{n}  [{units}]   IMU explains "
                       f"{sc.get('imu_explains_heldout_pct', float('nan')):.0f}%</span>")
            p.showGrid(x=True, y=True, alpha=0.18)
            p.setMenuEnabled(False)
            p.getAxis("left").setWidth(60)
            p.getAxis("left").enableAutoSIPrefix(False)
            p.getAxis("left").setLabel(units)
            p.addLegend(offset=(6, 4), labelTextSize="8pt", verSpacing=-6, brush=(27, 29, 33, 170))
            p.plot(x, ser.cols[n], pen=pg.mkPen((175, 180, 188, 200), width=1.2), name="measured")
            if f"{n}_imu" in ser.cols:
                p.plot(x, ser.cols[f"{n}_imu"], pen=pg.mkPen("#4e9cf5", width=1.6), name="IMU explains")
                p.plot(x, ser.cols[f"{n}_leftover"], pen=pg.mkPen("#ff8a3d", width=1.6), name="leftover")
            if first is None:
                first = p
            else:
                p.setXLink(first)
            if row < len(shown) - 1:
                p.getAxis("bottom").setStyle(showValues=False)
        if first is not None:
            self.glw.getItem(len(shown) - 1, 0).setLabel("bottom", "time since first camera frame [s]")
            for r in range(len(shown)):
                self.glw.getItem(r, 0).enableAutoRange(x=False)
            first.setXRange(float(x[0]), float(x[-1]), padding=0.01)


class ImuWindow(QtWidgets.QWidget):
    """Separate window: thrusters, acceleration, velocity and distance for the current Start→Stop."""

    def __init__(self, viewer: Viewer):
        super().__init__(viewer, QtCore.Qt.WindowType.Window)
        self.v = viewer
        self.setWindowTitle("IMU velocity")
        self.resize(1150, 880)
        lay = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        row = QtWidgets.QHBoxLayout()
        self.chan_edit = QtWidgets.QLineEdit(",".join(map(str, viewer.thruster_channels)))
        self.chan_edit.setMaximumWidth(110)
        self.chan_edit.setToolTip("rc/out channel indexes of the horizontal thrusters (BlueROV2: 0,1,2,3).\n"
                                  "Vertical thrusters hold depth all the time, so leave them out.")
        self.chan_edit.editingFinished.connect(self._chans_changed)
        for w in (QtWidgets.QLabel("Horizontal thruster channels:"), self.chan_edit):
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Mode:"))
        row.addWidget(QtWidgets.QLabel(""))
        self.mode_label = row.itemAt(row.count() - 1).widget()
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.summary)
        lay.addLayout(row)
        lay.addWidget(self.glw, 1)

    def _chans_changed(self):
        try:
            chans = [int(x) for x in self.chan_edit.text().replace(" ", "").split(",") if x != ""]
        except ValueError:
            return
        if chans and chans != self.v.thruster_channels:
            self.v.set_thruster_channels(chans)

    def refresh(self):
        v, d = self.v, self.v.data
        ser = v._imu_series
        self.mode_label.setText(v.imu_mode.currentText())
        self.glw.clear()
        lo, hi = v.start_spin.value(), v.stop_spin.value()
        if ser is None:
            self.summary.setText("<b>No IMU result</b> for this window. It needs <i>imu/data</i> with orientation "
                                 "and linear_acceleration between Start and Stop.")
            return
        info = ser.info
        err = np.round(info.get("end_error_before_correction", [0, 0, 0]), 3).tolist()
        bias = np.round(info.get("implied_accel_bias", [0, 0, 0]), 4).tolist()
        lines = [
            f"<b>Window</b> {lo:.2f} → {hi:.2f} s ({hi - lo:.1f} s) &nbsp;·&nbsp; "
            f"<b>Forward distance</b> {info.get('forward_distance_m', 0):.2f} m &nbsp;·&nbsp; "
            f"<b>Peak forward speed</b> {info.get('peak_forward_m_s', 0):.2f} m/s &nbsp;·&nbsp; "
            f"<b>Mean forward speed</b> {info.get('mean_forward_m_s', 0):.2f} m/s",
            f"Drift removed at Stop: {err} m/s (east, north, up), i.e. an accelerometer bias of about {bias} m/s². "
            "Velocity is set to 0 at your Start and Stop markers.",
        ]
        self.summary.setText("<br>".join(lines))

        x = d.rel(ser.t)
        thr = d.series.get(DERIVED_THR)
        specs = []
        if thr is not None:
            specs.append(("Horizontal thrusters: max |PWM − 1500|  [µs]   (0 = off)", d.rel(thr.t),
                          [("max_dev_us", thr.cols["max_dev_us"], LINE_COLORS[5], False)]))
        specs += [
            ("Acceleration along / across the heading, gravity removed  [m/s²]", x,
             [("forward", ser.cols["acc_forward"], LINE_COLORS[0], False),
              ("sideways", ser.cols["acc_sideways"], LINE_COLORS[1], False),
              ("vertical", ser.cols["acc_vertical"], LINE_COLORS[2], False)]),
            ("Speed  [m/s]   (dashed = forward speed before drift correction)", x,
             [("forward", ser.cols["forward_speed"], LINE_COLORS[0], False),
              ("sideways", ser.cols["sideways_speed"], LINE_COLORS[1], False),
              ("vertical", ser.cols["vertical_speed"], LINE_COLORS[2], False),
              ("forward raw", ser.cols["forward_speed_raw"], LINE_COLORS[0], True)]),
            ("Distance since Start  [m]", x, [("forward", ser.cols["forward_distance"], LINE_COLORS[4], False),
                                              ("total path", ser.cols["distance"], LINE_COLORS[3], True)]),
        ]
        first = None
        x0, x1 = max(lo - 2.0, 0.0), min(hi + 2.0, v._t_end)
        for row, (title, xs, curves) in enumerate(specs):
            p = self.glw.addPlot(row=row, col=0)
            p.setTitle(f"<span style='font-size:9pt'>{title}</span>")
            p.showGrid(x=True, y=True, alpha=0.18)
            p.setMenuEnabled(False)
            p.getAxis("left").setWidth(56)
            p.getAxis("left").enableAutoSIPrefix(False)
            p.addLegend(offset=(6, 4), labelTextSize="8pt", verSpacing=-6, brush=(27, 29, 33, 170))
            for name, ys, color, dashed in curves:
                pen = pg.mkPen(color, width=1.4 if not dashed else 1.2,
                               style=QtCore.Qt.PenStyle.DashLine if dashed else QtCore.Qt.PenStyle.SolidLine)
                p.plot(xs, ys, pen=pen, name=name, connect="finite")
            for xv, col in ((lo, "#ffd84d"), (hi, "#4e9cf5")):
                p.addItem(pg.InfiniteLine(xv, angle=90, pen=pg.mkPen(col, width=1.5)), ignoreBounds=True)
            if first is None:
                first = p
            else:
                p.setXLink(first)
            if row < len(specs) - 1:
                p.getAxis("bottom").setStyle(showValues=False)
        first.setXRange(x0, x1, padding=0)
        self.glw.getItem(len(specs) - 1, 0).setLabel("bottom", "seconds since first camera frame   "
                                                     "(yellow = Start, blue = Stop)")
        for p in (self.glw.getItem(i, 0) for i in range(len(specs))):
            p.enableAutoRange(y=True)
            p.setAutoVisible(y=True)


def main():
    ap = argparse.ArgumentParser(description="ROS 2 bag viewer with camera-synced CSV export")
    ap.add_argument("bag", nargs="?", help="bag folder (with metadata.yaml) or the .mcap inside it")
    ap.add_argument("--camera", help="image topic to sync to (default: first */image_raw)")
    ap.add_argument("--msg-dir", help="folder with .msg files for custom types not embedded in the bag")
    ap.add_argument("--include-covariance", action="store_true")
    args = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Bag viewer")
    app.setStyleSheet("""
        QScrollBar:vertical { width: 14px; background: #2a2533; margin: 0; }
        QScrollBar:horizontal { height: 14px; background: #2a2533; margin: 0; }
        QScrollBar::handle:vertical { background: #9b59ff; min-height: 30px; border-radius: 6px; margin: 2px; }
        QScrollBar::handle:horizontal { background: #9b59ff; min-width: 30px; border-radius: 6px; margin: 2px; }
        QScrollBar::handle:hover { background: #b98aff; }
        QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
        QScrollBar::add-page, QScrollBar::sub-page { background: none; }
    """)
    w = Viewer(args)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
