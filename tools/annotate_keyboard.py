# -*- coding: utf-8 -*-
"""
annotate_keyboard.py — Keyboard-driven annotation tool
D = Validate | S = Skip | A = Previous | Z = Reset | Right-click = Delete box
Automatically moves to the next individual once the target count is reached.
"""

import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import YOLO_DATASET_DIR

IMAGES_DIR = YOLO_DATASET_DIR / "images"
LABELS_DIR = YOLO_DATASET_DIR / "labels"
DONE_FILE  = YOLO_DATASET_DIR / "done.txt"

TARGET = 30   # valid images per individual

# =============================================================================
# done.txt helpers
# =============================================================================

def load_done():
    try: return set(open(DONE_FILE).read().splitlines())
    except: return set()

def mark_done(stem):
    with open(DONE_FILE, 'a') as f:
        f.write(stem + "\n")

# =============================================================================
# COUNT VALID IMAGES PER INDIVIDUAL (from done set only)
# =============================================================================

def count_valid(done):
    labels_dir = Path(LABELS_DIR)
    counts = defaultdict(int)
    for stem in done:
        lbl = labels_dir / (stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            counts[stem.split("_")[0]] += 1
    return counts

# =============================================================================
# ORDERED IMAGE LIST
# Images not yet processed; individuals below target first.
# Once an individual reaches target, their remaining images are skipped.
# =============================================================================

def get_images(done, valid_counts):
    per_ind = defaultdict(list)
    for p in sorted(Path(IMAGES_DIR).glob("*.jpg")):
        per_ind[p.stem.split("_")[0]].append(p)

    # Sort individuals by number of valid images (ascending — least-done first)
    individuals = sorted(per_ind.keys(), key=lambda k: valid_counts.get(k, 0))

    result = []
    for ind in individuals:
        if valid_counts.get(ind, 0) >= TARGET:
            continue
        for img in per_ind[ind]:
            if img.stem not in done:
                result.append(img)

    return result

# =============================================================================
# LABEL HELPERS
# =============================================================================

def read_label(path, W, H):
    boxes = []
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        for line in p.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                _, xc, yc, w, h = map(float, parts)
                boxes.append([
                    int((xc-w/2)*W), int((yc-h/2)*H),
                    int((xc+w/2)*W), int((yc+h/2)*H)
                ])
    return boxes

def write_label(path, boxes, W, H):
    with open(path, 'w') as f:
        for x1, y1, x2, y2 in boxes:
            xc = max(0, min(1, ((x1+x2)/2)/W))
            yc = max(0, min(1, ((y1+y2)/2)/H))
            w  = max(0.01, min(1, (x2-x1)/W))
            h  = max(0.01, min(1, (y2-y1)/H))
            f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

# =============================================================================
# APPLICATION
# =============================================================================

class App:
    def __init__(self, root):
        self.root  = root
        self.root.title("AnnotateTool")
        self.root.configure(bg="#111")
        self.root.state('zoomed')

        self.done         = load_done()
        self.valid_counts = count_valid(self.done)
        self.imgs         = get_images(self.done, self.valid_counts)
        self.idx          = 0

        self.boxes   = []
        self.orig    = []
        self.W = self.H = 1
        self.sc = 1.0
        self.ox = self.oy = 0
        self.drag = self.dstart = self.dbefore = None
        self.bi = 0
        self.drawing = False
        self.dpos = None

        self._build()
        self._binds()
        self.show()

    def _refresh_list(self):
        """Recompute image list after each action."""
        self.valid_counts = count_valid(self.done)
        self.imgs = get_images(self.done, self.valid_counts)
        self.idx  = 0

    # =========================================================================
    # UI
    # =========================================================================

    def _build(self):
        top = tk.Frame(self.root, bg="#1e1e1e", height=52)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)

        self.l_pos = tk.Label(top, text="", bg="#1e1e1e", fg="#ff4466",
                              font=("Consolas", 13, "bold"))
        self.l_pos.pack(side=tk.LEFT, padx=12)

        self.l_ind = tk.Label(top, text="", bg="#1e1e1e", fg="#ffffff",
                              font=("Consolas", 12))
        self.l_ind.pack(side=tk.LEFT, padx=8)

        self.l_obj = tk.Label(top, text="", bg="#1e1e1e", fg="#00cc66",
                              font=("Consolas", 11, "bold"))
        self.l_obj.pack(side=tk.LEFT, padx=16)

        self.l_rat = tk.Label(top, text="", bg="#1e1e1e", fg="#888888",
                              font=("Consolas", 10))
        self.l_rat.pack(side=tk.LEFT, padx=12)

        tk.Label(top,
            text="D Validate   S Skip   A Previous   Z Reset   Right-click Delete box",
            bg="#1e1e1e", fg="#444444", font=("Consolas", 9)
        ).pack(side=tk.RIGHT, padx=12)

        self.cv = tk.Canvas(self.root, bg="#0a0a0a", cursor="crosshair",
                            highlightthickness=0)
        self.cv.pack(fill=tk.BOTH, expand=True)

        bot = tk.Frame(self.root, bg="#1e1e1e", height=36)
        bot.pack(fill=tk.X, side=tk.BOTTOM)
        bot.pack_propagate(False)

        self.l_stat = tk.Label(bot, text="", bg="#1e1e1e", fg="#555555",
                               font=("Consolas", 10))
        self.l_stat.pack(side=tk.LEFT, padx=12)

        self.l_ind_stats = tk.Label(bot, text="", bg="#1e1e1e", fg="#444444",
                                    font=("Consolas", 9))
        self.l_ind_stats.pack(side=tk.RIGHT, padx=12)

        self.cv.bind("<ButtonPress-1>",   self.m_press)
        self.cv.bind("<B1-Motion>",       self.m_drag)
        self.cv.bind("<ButtonRelease-1>", self.m_release)
        self.cv.bind("<ButtonPress-3>",   self.m_right)

    def _binds(self):
        self.root.bind("d",       lambda e: self.validate())
        self.root.bind("D",       lambda e: self.validate())
        self.root.bind("<Right>", lambda e: self.validate())
        self.root.bind("s",       lambda e: self.skip())
        self.root.bind("S",       lambda e: self.skip())
        self.root.bind("a",       lambda e: self.prev())
        self.root.bind("A",       lambda e: self.prev())
        self.root.bind("<Left>",  lambda e: self.prev())
        self.root.bind("z",       lambda e: self.reset())
        self.root.bind("Z",       lambda e: self.reset())
        self.root.bind("<Escape>", lambda e: self.quit())

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def show(self):
        all_inds = set(p.stem.split("_")[0]
                       for p in Path(IMAGES_DIR).glob("*.jpg"))
        all_done = all(self.valid_counts.get(k, 0) >= TARGET for k in all_inds)

        if all_done:
            messagebox.showinfo(
                "Target reached",
                "All individuals have reached 30 valid images.\n"
                "Run 02_train_yolo_nano.py!"
            )
            self.root.quit()
            return

        if self.idx >= len(self.imgs):
            messagebox.showinfo(
                "End of list",
                "No more images to annotate for individuals below target.\n"
                "Run 02_train_yolo_nano.py!"
            )
            self.root.quit()
            return

        p  = self.imgs[self.idx]
        lp = Path(LABELS_DIR) / (p.stem + ".txt")

        img = Image.open(p).convert("RGB")
        self.W, self.H = img.size

        self.root.update()
        cw = self.cv.winfo_width(); ch = self.cv.winfo_height()
        self.sc = min(cw/self.W, ch/self.H) * 0.95
        dw = int(self.W*self.sc); dh = int(self.H*self.sc)
        self.ox = (cw-dw)//2; self.oy = (ch-dh)//2

        self.tk    = ImageTk.PhotoImage(img.resize((dw, dh), Image.LANCZOS))
        self.boxes = read_label(lp, self.W, self.H)
        self.orig  = [list(b) for b in self.boxes]
        self.bi    = 0

        ind     = p.stem.split("_")[0]
        nb_ind  = self.valid_counts.get(ind, 0)
        needed  = max(0, TARGET - nb_ind)
        total_v = sum(self.valid_counts.values())

        self.l_pos.config(text=f"{self.idx+1}/{len(self.imgs)}")
        self.l_ind.config(
            text=f"  {ind}  ({nb_ind}/{TARGET})  need {needed} more",
            fg="#ffaa00" if needed > 0 else "#00cc66"
        )
        self.l_obj.config(
            text=f"  Valid: {total_v}   Processed: {len(self.done)}",
            fg="#00cc66"
        )

        stats = "  ".join(
            f"{k}:{self.valid_counts.get(k,0)}"
            + ("" if self.valid_counts.get(k,0) >= TARGET
               else f"(-{TARGET - self.valid_counts.get(k,0)})")
            for k in sorted(all_inds)
        )
        self.l_stat.config(text=f"Image {self.idx+1}/{len(self.imgs)}")
        self.l_ind_stats.config(text=stats)

        self.redraw()

    def redraw(self):
        self.cv.delete("all")
        self.cv.create_image(self.ox, self.oy, anchor=tk.NW, image=self.tk)

        cw = self.cv.winfo_width(); ch = self.cv.winfo_height()
        for x in range(0, cw, 100):
            self.cv.create_line(x, 0, x, ch, fill="#181818", width=1)
        for y in range(0, ch, 100):
            self.cv.create_line(0, y, cw, y, fill="#181818", width=1)

        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            bx1=int(x1*self.sc)+self.ox; by1=int(y1*self.sc)+self.oy
            bx2=int(x2*self.sc)+self.ox; by2=int(y2*self.sc)+self.oy
            col = "#00ff88" if i == self.bi else "#4499ff"
            self.cv.create_rectangle(bx1, by1, bx2, by2, outline=col, width=2)
            for px, py in [
                (bx1,by1),(bx2,by1),(bx1,by2),(bx2,by2),
                ((bx1+bx2)//2,by1),((bx1+bx2)//2,by2),
                (bx1,(by1+by2)//2),(bx2,(by1+by2)//2)
            ]:
                self.cv.create_rectangle(px-5,py-5,px+5,py+5,fill=col,outline="#fff")
            r = (x2-x1)/(y2-y1) if (y2-y1) > 0 else 0
            self.cv.create_text(bx1+4, by1+4, text=f"{r:.2f}",
                                anchor=tk.NW, fill=col, font=("Consolas", 9))
            self.l_rat.config(text=f"W/H ratio: {r:.2f}  (target 1.0-1.5)")

        if not self.boxes:
            self.cv.create_text(cw//2, ch//2,
                text="No box — drag to create — S to skip",
                fill="#ff4466", font=("Consolas", 13))
            self.l_rat.config(text="No box")

    # =========================================================================
    # MOUSE
    # =========================================================================

    def to_img(self, cx, cy):
        return (cx-self.ox)/self.sc, (cy-self.oy)/self.sc

    def get_handle(self, cx, cy, x1, y1, x2, y2):
        bx1=int(x1*self.sc)+self.ox; by1=int(y1*self.sc)+self.oy
        bx2=int(x2*self.sc)+self.ox; by2=int(y2*self.sc)+self.oy
        mx=(bx1+bx2)//2; my=(by1+by2)//2
        for n,(px,py) in [
            ('tl',(bx1,by1)),('tr',(bx2,by1)),('bl',(bx1,by2)),('br',(bx2,by2)),
            ('tm',(mx,by1)), ('bm',(mx,by2)), ('lm',(bx1,my)), ('rm',(bx2,my))
        ]:
            if abs(cx-px)<12 and abs(cy-py)<12: return n
        return None

    def m_press(self, e):
        cx, cy = e.x, e.y
        for i, (x1,y1,x2,y2) in enumerate(self.boxes):
            h = self.get_handle(cx, cy, x1, y1, x2, y2)
            if h:
                self.drag=h; self.dstart=(cx,cy)
                self.dbefore=list(self.boxes[i]); self.bi=i; return
            bx1=int(x1*self.sc)+self.ox; by1=int(y1*self.sc)+self.oy
            bx2=int(x2*self.sc)+self.ox; by2=int(y2*self.sc)+self.oy
            if bx1<cx<bx2 and by1<cy<by2:
                self.drag='move'; self.dstart=(cx,cy)
                self.dbefore=list(self.boxes[i]); self.bi=i; return
        self.drawing=True; self.dpos=self.to_img(cx, cy)

    def m_drag(self, e):
        cx, cy = e.x, e.y
        if self.drawing and self.dpos:
            ix, iy = self.to_img(cx, cy)
            x1=min(self.dpos[0],ix); y1=min(self.dpos[1],iy)
            x2=max(self.dpos[0],ix); y2=max(self.dpos[1],iy)
            self.cv.delete("all")
            self.cv.create_image(self.ox, self.oy, anchor=tk.NW, image=self.tk)
            for bx1,by1,bx2,by2 in self.boxes:
                self.cv.create_rectangle(
                    int(bx1*self.sc)+self.ox, int(by1*self.sc)+self.oy,
                    int(bx2*self.sc)+self.ox, int(by2*self.sc)+self.oy,
                    outline="#4499ff", width=2)
            self.cv.create_rectangle(
                int(x1*self.sc)+self.ox, int(y1*self.sc)+self.oy,
                int(x2*self.sc)+self.ox, int(y2*self.sc)+self.oy,
                outline="#ffff00", width=2)
            return
        if self.drag and self.dstart:
            dx=(cx-self.dstart[0])/self.sc; dy=(cy-self.dstart[1])/self.sc
            b=list(self.dbefore); i=self.bi; m=self.drag
            if   m=='move': self.boxes[i]=[b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy]
            elif m=='tl':   self.boxes[i]=[b[0]+dx,b[1]+dy,b[2],b[3]]
            elif m=='tr':   self.boxes[i]=[b[0],b[1]+dy,b[2]+dx,b[3]]
            elif m=='bl':   self.boxes[i]=[b[0]+dx,b[1],b[2],b[3]+dy]
            elif m=='br':   self.boxes[i]=[b[0],b[1],b[2]+dx,b[3]+dy]
            elif m=='tm':   self.boxes[i]=[b[0],b[1]+dy,b[2],b[3]]
            elif m=='bm':   self.boxes[i]=[b[0],b[1],b[2],b[3]+dy]
            elif m=='lm':   self.boxes[i]=[b[0]+dx,b[1],b[2],b[3]]
            elif m=='rm':   self.boxes[i]=[b[0],b[1],b[2]+dx,b[3]]
            self.redraw()

    def m_release(self, e):
        if self.drawing and self.dpos:
            ix, iy = self.to_img(e.x, e.y)
            x1=min(self.dpos[0],ix); y1=min(self.dpos[1],iy)
            x2=max(self.dpos[0],ix); y2=max(self.dpos[1],iy)
            if abs(x2-x1)>10 and abs(y2-y1)>10:
                self.boxes.append([x1,y1,x2,y2]); self.bi=len(self.boxes)-1
            self.drawing=False; self.dpos=None; self.redraw()
        self.drag=None; self.dstart=None

    def m_right(self, e):
        cx, cy = e.x, e.y
        for i, (x1,y1,x2,y2) in enumerate(self.boxes):
            bx1=int(x1*self.sc)+self.ox; by1=int(y1*self.sc)+self.oy
            bx2=int(x2*self.sc)+self.ox; by2=int(y2*self.sc)+self.oy
            if bx1<cx<bx2 and by1<cy<by2:
                self.boxes.pop(i); self.bi=max(0,len(self.boxes)-1)
                self.redraw(); return

    # =========================================================================
    # ACTIONS
    # =========================================================================

    def validate(self):
        p  = self.imgs[self.idx]
        lp = Path(LABELS_DIR) / (p.stem + ".txt")
        write_label(lp, self.boxes, self.W, self.H)
        if p.stem not in self.done:
            self.done.add(p.stem)
            mark_done(p.stem)
        # Recompute list — if individual reached target, list refreshes
        self._refresh_list()
        self.bi = 0
        self.show()

    def skip(self):
        p  = self.imgs[self.idx]
        lp = Path(LABELS_DIR) / (p.stem + ".txt")
        open(lp, 'w').close()
        if p.stem not in self.done:
            self.done.add(p.stem)
            mark_done(p.stem)
        self.idx += 1; self.bi = 0; self.show()

    def prev(self):
        if self.idx > 0:
            self.idx -= 1; self.bi = 0; self.show()

    def reset(self):
        self.boxes = [list(b) for b in self.orig]; self.bi = 0; self.redraw()

    def quit(self):
        if messagebox.askyesno("Quit", "Quit the annotator?"):
            self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
