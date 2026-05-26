"""
V2_4_review_crops.py
=====================
CNRS IPHC Strasbourg — Orang-outan V2 pipeline
Author: Titouane

Crop reviewer with native Windows drag-and-drop (PyQt5).
Inspired by 3c_corriger_crops.py — same interaction model.

INTERACTION
-----------
  Mouse on existing box:
    - Click + drag INSIDE box  → move the box
    - Click + drag on HANDLE   → resize (8 handles: corners + edges)
    - Right-click anywhere     → center box on cursor
    - Z                        → reset box to original position

  D key → draw mode: left-click + drag to draw a completely new box
          (drawing a box does NOT validate — you still need Enter/D to save)

  Enter / D (not in draw mode) → validate + save + load next
  S                            → skip (no change)
  A / Left arrow               → go back to previous
  Delete / X                   → delete crop file + original photo + JSON entry
  Escape                       → cancel draw mode
  Q                            → quit

PREVIEW
-------
  224×224 crop updates live as you move/resize the box.

JSON SAFETY
-----------
  Atomic write (tmp → rename) + .bak backup before every save.

RUN
---
  conda activate orangs
  python D:\OrangIdentifier\V2\scripts\V2_4_review_crops.py

  # Pass files directly:
  python V2_4_review_crops.py path/to/crop1.jpg path/to/crop2.jpg
"""

import sys
import json
import shutil
import threading
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QSizePolicy, QFrame, QSplitter
)
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont,
    QBrush, QCursor, QDragEnterEvent, QDropEvent, QPaintEvent, QMouseEvent
)

# ==============================================================================
# PATHS
# ==============================================================================

DEFAULT_JSON = Path(r"D:\OrangIdentifier\V2\WILD_CROPS\boxes_wild.json")
CROPS_DIR    = Path(r"D:\OrangIdentifier\V2\WILD_CROPS\crops")
CROP_SIZE    = 224
HANDLE_RADIUS = 7   # px, in widget space

# ==============================================================================
# JSON MANAGER — atomic + backup
# ==============================================================================

class JsonManager:
    def __init__(self, path: Path):
        self.path  = path
        self._lock = threading.Lock()
        self._data = {}

    def load(self) -> dict:
        with self._lock:
            if not self.path.exists():
                self._data = {}
                return {}
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
            return dict(self._data)

    def save(self, data: dict) -> bool:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            bak = self.path.with_suffix(".bak")
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                if self.path.exists():
                    shutil.copy2(self.path, bak)
                tmp.replace(self.path)
                self._data = data
                return True
            except Exception as e:
                print(f"[ERROR] JSON save failed: {e}")
                if tmp.exists():
                    try: tmp.unlink()
                    except: pass
                return False

    def get_data(self) -> dict:
        with self._lock:
            return dict(self._data)

# ==============================================================================
# FIND JSON ENTRY FROM CROP FILENAME
# ==============================================================================

def find_entry(crop_path: Path, db: dict):
    name = crop_path.name
    stem = crop_path.stem

    # 1. Exact match on crop_file
    for key, val in db.items():
        if not isinstance(val, dict): continue
        if Path(val.get("crop_file", "")).name == name:
            return key, val
        for det in val.get("detections", []):
            if Path(det.get("crop_file", "")).name == name:
                return key, det

    # 2. Stem matching: "{orig_stem}_face{n}" or "{orig_stem}_lowconf_{n}"
    for suffix in ("_lowconf", "_face"):
        if suffix in stem:
            orig_stem = stem[:stem.rfind(suffix)]
            for key, val in db.items():
                if not isinstance(val, dict): continue
                if Path(val.get("photo_source", "")).stem == orig_stem:
                    return key, val
                for det in val.get("detections", []):
                    if Path(det.get("photo_source", "")).stem == orig_stem:
                        return key, det

    # 3. Prefix match
    for key, val in db.items():
        if not isinstance(val, dict): continue
        src = val.get("photo_source", "")
        if src and name.startswith(Path(src).stem):
            return key, val

    return None, {}

# ==============================================================================
# CROP GENERATION
# ==============================================================================

def regen_crop(photo_src: str, x1: int, y1: int, x2: int, y2: int,
               dest: Path) -> np.ndarray | None:
    """Regenerates 224x224 crop. Returns the crop array on success, None on failure."""
    img = cv2.imread(photo_src)
    if img is None: return None
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
    x2 = max(x1+1, min(x2, w)); y2 = max(y1+1, min(y2, h))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return None
    resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    if dest is not None:
        cv2.imwrite(str(dest), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return resized

# ==============================================================================
# IMAGE CANVAS — handles move/resize/draw
# ==============================================================================

class ImageCanvas(QWidget):
    """
    Displays the original image with a draggable, resizable bounding box.
    Box coordinates are always in IMAGE space.
    """
    boxChanged = pyqtSignal(int, int, int, int)  # emitted whenever box moves

    # Handle names (8 + move)
    HANDLES = ["tl","tm","tr","rm","br","bm","bl","lm","move"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(500, 400)
        self.setMouseTracking(True)

        self._pixmap  = None
        self._img_rect = QRect()

        # Box in IMAGE coordinates
        self._bx1 = self._by1 = 0
        self._bx2 = self._by2 = 100
        self._orig_box = (0, 0, 100, 100)  # for reset

        # Interaction state
        self._draw_mode  = False   # D key toggled draw mode
        self._drag_handle = None   # which handle is being dragged
        self._drag_start_w = None  # QPoint in widget space at drag start
        self._drag_start_box = None  # box at drag start
        self._drawing     = False  # currently drawing a new box
        self._draw_p1     = None   # QPoint in widget space

        self.setCursor(Qt.ArrowCursor)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_image_and_box(self, img_bgr: np.ndarray, box: tuple):
        """img_bgr: OpenCV BGR. box: (x1,y1,x2,y2) in image coords."""
        rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.set_box(*box)
        self._orig_box  = tuple(box)
        self._draw_mode = False
        self._drawing   = False
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_box(self, x1, y1, x2, y2):
        if not self._pixmap: return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        self._bx1 = max(0, min(int(x1), pw-1))
        self._by1 = max(0, min(int(y1), ph-1))
        self._bx2 = max(self._bx1+1, min(int(x2), pw))
        self._by2 = max(self._by1+1, min(int(y2), ph))
        self.boxChanged.emit(self._bx1, self._by1, self._bx2, self._by2)
        self.update()

    def get_box(self):
        return (self._bx1, self._by1, self._bx2, self._by2)

    def reset_box(self):
        self.set_box(*self._orig_box)

    def set_draw_mode(self, enabled: bool):
        self._draw_mode = enabled
        self._drawing   = False
        self._draw_p1   = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def clear(self):
        self._pixmap = None
        self._draw_mode = False
        self._drawing   = False
        self.update()

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _fit_rect(self) -> QRect:
        if not self._pixmap: return QRect()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        s  = min(ww / pw, wh / ph)
        nw, nh = int(pw * s), int(ph * s)
        return QRect((ww - nw) // 2, (wh - nh) // 2, nw, nh)

    def _to_widget(self, ix, iy, r: QRect) -> QPoint:
        if not self._pixmap or r.isEmpty(): return QPoint(int(ix), int(iy))
        sx = r.width()  / self._pixmap.width()
        sy = r.height() / self._pixmap.height()
        return QPoint(r.x() + int(ix * sx), r.y() + int(iy * sy))

    def _to_image(self, wx, wy, r: QRect) -> tuple:
        if not self._pixmap or r.isEmpty(): return (wx, wy)
        sx = self._pixmap.width()  / r.width()
        sy = self._pixmap.height() / r.height()
        ix = (wx - r.x()) * sx
        iy = (wy - r.y()) * sy
        pw, ph = self._pixmap.width(), self._pixmap.height()
        return (max(0, min(ix, pw)), max(0, min(iy, ph)))

    def _handle_points(self, r: QRect) -> dict:
        """Returns handle positions in WIDGET space."""
        p = lambda x, y: self._to_widget(x, y, r)
        x1,y1,x2,y2 = self._bx1,self._by1,self._bx2,self._by2
        mx,my = (x1+x2)/2, (y1+y2)/2
        return {
            "tl": p(x1,y1), "tm": p(mx,y1), "tr": p(x2,y1),
            "rm": p(x2,my), "br": p(x2,y2), "bm": p(mx,y2),
            "bl": p(x1,y2), "lm": p(x1,my),
        }

    def _hit_handle(self, wx, wy, r: QRect) -> str | None:
        """Returns handle name if (wx,wy) is near a handle, else None."""
        for name, pt in self._handle_points(r).items():
            if abs(wx - pt.x()) <= HANDLE_RADIUS and abs(wy - pt.y()) <= HANDLE_RADIUS:
                return name
        # Check inside box
        p1 = self._to_widget(self._bx1, self._by1, r)
        p2 = self._to_widget(self._bx2, self._by2, r)
        box_r = QRect(p1, p2).normalized()
        if box_r.contains(QPoint(wx, wy)):
            return "move"
        return None

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, e: QMouseEvent):
        if not self._pixmap: return
        r  = self._fit_rect()
        wx, wy = e.x(), e.y()

        # Right-click → center box on cursor
        if e.button() == Qt.RightButton:
            ix, iy = self._to_image(wx, wy, r)
            hw = (self._bx2 - self._bx1) / 2
            hh = (self._by2 - self._by1) / 2
            self.set_box(ix - hw, iy - hh, ix + hw, iy + hh)
            return

        if e.button() != Qt.LeftButton: return

        if self._draw_mode:
            # Start drawing a new box
            self._drawing  = True
            self._draw_p1  = (wx, wy)
            return

        # Check handles
        h = self._hit_handle(wx, wy, r)
        if h:
            self._drag_handle    = h
            self._drag_start_w   = (wx, wy)
            self._drag_start_box = self.get_box()
            cur = {
                "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
                "tm": Qt.SizeVerCursor,  "bm": Qt.SizeVerCursor,
                "lm": Qt.SizeHorCursor,  "rm": Qt.SizeHorCursor,
                "move": Qt.SizeAllCursor,
            }.get(h, Qt.ArrowCursor)
            self.setCursor(cur)

    def mouseMoveEvent(self, e: QMouseEvent):
        if not self._pixmap: return
        r  = self._fit_rect()
        wx, wy = e.x(), e.y()

        # Update cursor when hovering
        if not self._drag_handle and not self._drawing and not self._draw_mode:
            h = self._hit_handle(wx, wy, r)
            cur = {
                "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
                "tm": Qt.SizeVerCursor,   "bm": Qt.SizeVerCursor,
                "lm": Qt.SizeHorCursor,   "rm": Qt.SizeHorCursor,
                "move": Qt.SizeAllCursor,
            }.get(h, Qt.CrossCursor if self._draw_mode else Qt.ArrowCursor)
            self.setCursor(cur)

        # Drawing a new box
        if self._drawing and self._draw_p1:
            p1x, p1y = self._draw_p1
            # Convert both corners to image space and update box
            ix1, iy1 = self._to_image(min(p1x,wx), min(p1y,wy), r)
            ix2, iy2 = self._to_image(max(p1x,wx), max(p1y,wy), r)
            self.set_box(ix1, iy1, ix2, iy2)
            return

        # Dragging a handle
        if self._drag_handle and self._drag_start_w:
            sw, sh = self._drag_start_w
            dx_w = wx - sw
            dy_w = wy - sh
            # Convert delta to image space
            if self._pixmap and not r.isEmpty():
                dx = dx_w * self._pixmap.width()  / r.width()
                dy = dy_w * self._pixmap.height() / r.height()
            else:
                dx, dy = dx_w, dy_w

            b  = self._drag_start_box
            x1,y1,x2,y2 = b[0],b[1],b[2],b[3]
            h  = self._drag_handle

            if   h == "move": x1+=dx;y1+=dy;x2+=dx;y2+=dy
            elif h == "tl":   x1+=dx;y1+=dy
            elif h == "tr":   x2+=dx;y1+=dy
            elif h == "bl":   x1+=dx;y2+=dy
            elif h == "br":   x2+=dx;y2+=dy
            elif h == "tm":   y1+=dy
            elif h == "bm":   y2+=dy
            elif h == "lm":   x1+=dx
            elif h == "rm":   x2+=dx

            self.set_box(x1, y1, x2, y2)

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._drag_handle    = None
            self._drag_start_w   = None
            self._drag_start_box = None
            if self._drawing:
                self._drawing   = False
                self._draw_p1   = None
                self._draw_mode = False   # exit draw mode after one draw
                self.setCursor(Qt.ArrowCursor)

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, e: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(10, 10, 10))

        if not self._pixmap:
            self._draw_hint(p)
            return

        r = self._fit_rect()
        p.drawPixmap(r, self._pixmap)

        # Box
        pt1 = self._to_widget(self._bx1, self._by1, r)
        pt2 = self._to_widget(self._bx2, self._by2, r)
        box_r = QRect(pt1, pt2).normalized()

        bw = self._bx2 - self._bx1
        bh = self._by2 - self._by1
        ratio = bw / bh if bh > 0 else 0
        col = QColor(255, 60, 80) if ratio < 0.5 or ratio > 2.0 else QColor(0, 220, 90)

        pen = QPen(col, 2)
        p.setPen(pen)
        p.drawRect(box_r)

        # Handles
        hp = self._handle_points(r)
        p.setBrush(QBrush(col))
        p.setPen(QPen(Qt.white, 1))
        for pt in hp.values():
            p.drawEllipse(pt, HANDLE_RADIUS, HANDLE_RADIUS)

        # Ratio label
        p.setFont(QFont("Consolas", 9))
        p.setPen(QPen(col))
        label = f"{int(bw)}×{int(bh)}px  ratio={ratio:.2f}"
        p.drawText(box_r.left(), box_r.top() - 6, label)

        # Draw mode overlay
        if self._draw_mode:
            p.setFont(QFont("Consolas", 11, QFont.Bold))
            p.setPen(QPen(QColor(255, 160, 0)))
            p.drawText(10, 24, "DRAW MODE — click and drag to draw a new box")

    def _draw_hint(self, p: QPainter):
        p.setPen(QColor(50, 50, 50))
        p.drawRect(self.rect().adjusted(30, 30, -30, -30))
        p.setPen(QColor(70, 70, 70))
        p.setFont(QFont("Consolas", 13))
        p.drawText(self.rect(), Qt.AlignCenter,
                   "Drag & drop crop images here\n\n"
                   "Move box: drag inside or on handles\n"
                   "Resize: drag corner/edge handles\n"
                   "New box: press D then draw\n"
                   "Right-click: center box on cursor\n\n"
                   "Enter/D = save    S = skip    A = back    Z = reset    Q = quit")

# ==============================================================================
# PREVIEW LABEL
# ==============================================================================

class PreviewLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(224, 224)
        self.setStyleSheet("border: 1px solid #333; background: #0a0a0a;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("preview")

    def set_crop(self, crop_bgr: np.ndarray):
        if crop_bgr is None:
            self.setText("error")
            return
        rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

# ==============================================================================
# MAIN WINDOW
# ==============================================================================

class ReviewWindow(QMainWindow):
    def __init__(self, json_path: Path, initial_files: list):
        super().__init__()
        self.json_mgr = JsonManager(json_path)
        self.db       = self.json_mgr.load()
        self.queue    = deque()
        self.history  = []   # list of crop paths for "previous" navigation

        self.current_crop  = None
        self.current_key   = None
        self.current_entry = None
        self.current_img   = None

        self.stats = {"validated": 0, "corrected": 0,
                      "rejected": 0, "skipped": 0, "deleted": 0}

        self._build_ui()
        self.setAcceptDrops(True)
        self.setWindowTitle("Crop Reviewer — CNRS IPHC")
        self.resize(1300, 820)

        if initial_files:
            self._enqueue([Path(f) for f in initial_files])

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Main area: canvas + right panel
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Canvas
        self.canvas = ImageCanvas()
        self.canvas.boxChanged.connect(self._on_box_changed)
        content_layout.addWidget(self.canvas, stretch=1)

        # Right panel
        right = QFrame()
        right.setFixedWidth(260)
        right.setStyleSheet("QFrame{background:#111;border-left:1px solid #222;}")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(8)

        lbl = QLabel("PREVIEW  224×224")
        lbl.setStyleSheet("color:#444;font-family:Consolas;font-size:10px;")
        rl.addWidget(lbl, alignment=Qt.AlignHCenter)

        self.preview = PreviewLabel()
        rl.addWidget(self.preview, alignment=Qt.AlignHCenter)

        self.lbl_box = QLabel("")
        self.lbl_box.setStyleSheet(
            "color:#666;font-family:Consolas;font-size:10px;"
            "background:#0d0d0d;padding:6px;border-radius:3px;"
        )
        self.lbl_box.setWordWrap(True)
        rl.addWidget(self.lbl_box)

        rl.addStretch()

        def btn(text, shortcut, color, slot):
            b = QPushButton(f"{text}  [{shortcut}]")
            b.setFixedHeight(36)
            b.setStyleSheet(
                f"QPushButton{{background:{color};color:#eee;"
                f"border:none;border-radius:4px;"
                f"font-family:Consolas;font-size:12px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{color}cc;}}"
                f"QPushButton:pressed{{background:{color}88;}}"
            )
            b.clicked.connect(slot)
            rl.addWidget(b)
            return b

        btn("Draw new box",  "D",      "#8a5c00", self._action_draw)
        btn("Validate",      "Enter",  "#1a5c2e", self._action_validate)
        btn("Reject",        "R",      "#5c1a1a", self._action_reject)
        btn("Skip",          "S",      "#3a3a3a", self._action_skip)
        btn("Previous",      "A",      "#1a2a5c", self._action_previous)
        btn("Reset box",     "Z",      "#3a3a3a", self._action_reset)
        btn("DELETE ALL",    "Del/X",  "#6e0000", self._action_delete)

        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet(
            "color:#555;font-family:Consolas;font-size:9px;"
        )
        self.lbl_stats.setWordWrap(True)
        rl.addWidget(self.lbl_stats)

        content_layout.addWidget(right)
        root_layout.addWidget(content, stretch=1)

        # Bottom bar
        bot = QFrame()
        bot.setFixedHeight(28)
        bot.setStyleSheet("QFrame{background:#0a0a0a;border-top:1px solid #1e1e1e;}")
        bl = QHBoxLayout(bot)
        bl.setContentsMargins(10, 0, 10, 0)
        self.lbl_status = QLabel("Ready — drag crop images onto the window")
        self.lbl_status.setStyleSheet("color:#555;font-family:Consolas;font-size:10px;")
        bl.addWidget(self.lbl_status)
        root_layout.addWidget(bot)

    def _set_status(self, msg: str):
        self.lbl_status.setText(msg)

    def _update_stats_label(self):
        s = self.stats
        self.lbl_stats.setText(
            f"✓ {s['corrected']} corrected\n"
            f"✔ {s['validated']} validated\n"
            f"✗ {s['rejected']} rejected\n"
            f"→ {s['skipped']} skipped\n"
            f"🗑 {s['deleted']} deleted\n"
            f"Queue: {len(self.queue)}"
        )

    # ── Box changed → update preview ─────────────────────────────────────────

    def _on_box_changed(self, x1, y1, x2, y2):
        if self.current_img is None or self.current_entry is None: return
        src = self.current_entry.get("photo_source", "")
        if not src: return
        crop = regen_crop(src, x1, y1, x2, y2, dest=None)
        self.preview.set_crop(crop)
        bw, bh = x2-x1, y2-y1
        ratio = bw/bh if bh>0 else 0
        self.lbl_box.setText(
            f"x1={x1}  y1={y1}\nx2={x2}  y2={y2}\n"
            f"size: {bw}×{bh}px\nratio: {ratio:.2f}"
        )

    # ── Drag & Drop ───────────────────────────────────────────────────────────

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        paths = []
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.exists():
                paths.append(p)
        if paths:
            self._enqueue(paths)

    # ── Queue ─────────────────────────────────────────────────────────────────

    def _enqueue(self, paths: list):
        for p in paths:
            self.queue.append(p)
        self._set_status(f"Added {len(paths)} file(s). Queue: {len(self.queue)}")
        self._update_stats_label()
        if self.current_crop is None:
            self._load_next()

    def _load_crop(self, crop_path: Path):
        self.db = self.json_mgr.get_data()
        key, entry = find_entry(crop_path, self.db)

        if not key:
            self._set_status(f"[WARN] No JSON entry for {crop_path.name} — skipping")
            return False

        src = entry.get("photo_source", "")
        if not src or not Path(src).exists():
            self._set_status(f"[WARN] Source not found: {src} — skipping")
            return False

        img = cv2.imread(src)
        if img is None:
            self._set_status(f"[WARN] Cannot read {src} — skipping")
            return False

        self.current_crop  = crop_path
        self.current_key   = key
        self.current_entry = entry
        self.current_img   = img

        x1 = entry.get("crop_x1", 0)
        y1 = entry.get("crop_y1", 0)
        x2 = entry.get("crop_x2", 100)
        y2 = entry.get("crop_y2", 100)

        self.canvas.set_image_and_box(img, (x1, y1, x2, y2))
        self._on_box_changed(x1, y1, x2, y2)

        self._set_status(
            f"{crop_path.name}   source: {Path(src).name}   "
            f"status: {entry.get('statut','?')}   "
            f"queue: {len(self.queue)}"
        )
        self._update_stats_label()
        return True

    def _load_next(self):
        while self.queue:
            crop_path = self.queue.popleft()
            if self._load_crop(crop_path):
                self.history.append(crop_path)
                return
        # Queue empty
        self.current_crop = None
        self.canvas.clear()
        self.preview.setText("done")
        self._set_status("Queue empty — drop more files to continue")
        self._update_stats_label()

    # ── Actions ───────────────────────────────────────────────────────────────

    def _action_draw(self):
        if self.current_crop is None: return
        self.canvas.set_draw_mode(True)
        self._set_status("Draw mode: left-click + drag to draw a new box.  Escape = cancel")

    def _action_validate(self):
        if self.current_crop is None or self.current_entry is None: return

        x1, y1, x2, y2 = self.canvas.get_box()
        src = self.current_entry.get("photo_source", "")

        # Regenerate crop file
        ok = regen_crop(src, x1, y1, x2, y2, self.current_crop)
        if ok is None:
            self._set_status("[ERROR] Could not regenerate crop — check coordinates")
            return

        # Update JSON
        updated = dict(self.current_entry)
        updated.update({
            "crop_x1":           x1, "crop_y1": y1,
            "crop_x2":           x2, "crop_y2": y2,
            "statut":            "valide",
            "manually_reviewed": True,
            "review_date":       datetime.now().isoformat(),
            "crop_file":         str(self.current_crop),
        })
        data = self.json_mgr.get_data()
        data[self.current_key] = updated
        saved = self.json_mgr.save(data)

        if saved:
            self.stats["corrected"] += 1
            self._set_status(f"✓ Saved: {self.current_crop.name}")
        else:
            self._set_status("[ERROR] JSON save failed")

        self.current_crop = None
        self._load_next()

    def _action_reject(self):
        if self.current_crop is None or self.current_entry is None: return
        updated = dict(self.current_entry)
        updated["statut"]      = "rejete"
        updated["reject_date"] = datetime.now().isoformat()
        data = self.json_mgr.get_data()
        data[self.current_key] = updated
        self.json_mgr.save(data)
        self.stats["rejected"] += 1
        self._set_status(f"✗ Rejected: {self.current_crop.name}")
        self.current_crop = None
        self._load_next()

    def _action_skip(self):
        if self.current_crop is None: return
        self.stats["skipped"] += 1
        self._set_status(f"→ Skipped: {self.current_crop.name}")
        self.current_crop = None
        self._load_next()

    def _action_previous(self):
        if len(self.history) < 2: return
        if self.current_crop:
            self.queue.appendleft(self.current_crop)
        prev = self.history[-2]
        self.history = self.history[:-2]
        self.queue.appendleft(prev)
        self.current_crop = None
        self._load_next()

    def _action_reset(self):
        if self.current_crop is None: return
        self.canvas.reset_box()

    def _action_delete(self):
        """
        Permanently deletes:
          1. The crop file (224×224 JPEG in crops/)
          2. The original source photo (in WILD_ORANGS/raw/)
          3. The JSON entry in boxes_wild.json

        No confirmation dialog — this is intentional for speed during review.
        The .bak JSON backup is always kept as safety net.
        """
        if self.current_crop is None or self.current_entry is None: return

        deleted = []
        failed  = []

        # 1. Delete crop file
        crop_file = Path(self.current_entry.get("crop_file", str(self.current_crop)))
        for f in [crop_file, self.current_crop]:
            if f and f.exists():
                try:
                    f.unlink()
                    deleted.append(f.name)
                except Exception as e:
                    failed.append(f"{f.name}: {e}")

        # 2. Delete original source photo
        src = self.current_entry.get("photo_source", "")
        if src:
            src_path = Path(src)
            if src_path.exists():
                try:
                    src_path.unlink()
                    deleted.append(src_path.name)
                except Exception as e:
                    failed.append(f"{src_path.name}: {e}")

        # 3. Remove JSON entry (atomic save with backup)
        data = self.json_mgr.get_data()
        if self.current_key in data:
            del data[self.current_key]
            self.json_mgr.save(data)

        self.stats["deleted"] += 1
        msg = f"🗑 Deleted: {', '.join(deleted)}"
        if failed:
            msg += f"  [WARN could not delete: {', '.join(failed)}]"
        self._set_status(msg)

        self.current_crop = None
        self._load_next()



    # ── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_D:
            self._action_draw()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._action_validate()
        elif key == Qt.Key_R:
            self._action_reject()
        elif key == Qt.Key_S:
            self._action_skip()
        elif key in (Qt.Key_A, Qt.Key_Left):
            self._action_previous()
        elif key == Qt.Key_Z:
            self._action_reset()
        elif key in (Qt.Key_Delete, Qt.Key_X):
            self._action_delete()
        elif key == Qt.Key_Escape:
            self.canvas.set_draw_mode(False)
            self._set_status("Draw mode cancelled")
        elif key == Qt.Key_Q:
            self.close()

    def closeEvent(self, e):
        s = self.stats
        print(f"\n{'='*50}\n  Session: "
              f"{s['corrected']} corrected  {s['validated']} validated  "
              f"{s['rejected']} rejected  {s['skipped']} skipped  "
              f"{s['deleted']} deleted\n{'='*50}")
        e.accept()

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("files", nargs="*", type=str)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from PyQt5.QtGui import QPalette
    pal = QPalette()
    pal.setColor(QPalette.Window,     QColor(12, 12, 12))
    pal.setColor(QPalette.WindowText, QColor(200, 200, 200))
    pal.setColor(QPalette.Base,       QColor(18, 18, 18))
    pal.setColor(QPalette.Button,     QColor(35, 35, 35))
    pal.setColor(QPalette.ButtonText, QColor(200, 200, 200))
    app.setPalette(pal)

    win = ReviewWindow(args.json, args.files)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()