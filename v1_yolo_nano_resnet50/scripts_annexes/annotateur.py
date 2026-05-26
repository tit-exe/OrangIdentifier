# -*- coding: utf-8 -*-
"""
OrangAnnot - Outil d'annotation
D = Valider | A = Precedent | S = Skip | Z = Reset boites | Clic droit = Supprimer boite
"""

import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk

IMAGES_DIR    = r"D:\OrangIdentifier\DATASET_YOLO\images"
LABELS_DIR    = r"D:\OrangIdentifier\DATASET_YOLO\labels"
PROGRESS_FILE = r"D:\OrangIdentifier\DATASET_YOLO\progress.txt"
VALIDEES_FILE = r"D:\OrangIdentifier\DATASET_YOLO\validees.txt"
PRIORITE      = 30
OBJECTIF      = 300

# ── Persistance ────────────────────────────────────────────────

def save_progress(n):
    open(PROGRESS_FILE,'w').write(str(n))

def load_progress():
    try: return int(open(PROGRESS_FILE).read().strip())
    except: return 0

def load_validees():
    try: return int(open(VALIDEES_FILE).read().strip())
    except: return 0

def inc_validees():
    open(VALIDEES_FILE,'w').write(str(load_validees()+1))

# ── Images ──────────────────────────────────────────────────────

def get_images():
    inds = {}
    for p in sorted(Path(IMAGES_DIR).glob("*.jpg")):
        k = p.stem.split("_")[0]
        inds.setdefault(k,[]).append(p)
    prio, reste = [], []
    for k,v in sorted(inds.items()):
        prio.extend(v[:PRIORITE]); reste.extend(v[PRIORITE:])
    return prio + reste

# ── Labels ──────────────────────────────────────────────────────

def read_label(path, W, H):
    boxes = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            p = line.strip().split()
            if len(p)==5:
                _,xc,yc,w,h = map(float,p)
                boxes.append([int((xc-w/2)*W), int((yc-h/2)*H),
                               int((xc+w/2)*W), int((yc+h/2)*H)])
    return boxes

def write_label(path, boxes, W, H):
    with open(path,'w') as f:
        for x1,y1,x2,y2 in boxes:
            xc = max(0,min(1,((x1+x2)/2)/W))
            yc = max(0,min(1,((y1+y2)/2)/H))
            w  = max(0.01,min(1,(x2-x1)/W))
            h  = max(0.01,min(1,(y2-y1)/H))
            f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

# ── App ─────────────────────────────────────────────────────────

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("OrangAnnot")
        self.root.configure(bg="#111")
        self.root.state('zoomed')

        self.imgs  = get_images()
        self.total = len(self.imgs)
        self.idx   = min(load_progress(), self.total-1)
        self.val   = load_validees()

        self.boxes = []
        self.orig  = []
        self.W = self.H = 1
        self.sc = 1.0
        self.ox = self.oy = 0

        self.drag   = None
        self.dstart = None
        self.dbefore= None
        self.bi     = 0
        self.drawing= False
        self.dpos   = None

        self._build()
        self._binds()
        self.show()

    def _build(self):
        # TOP BAR
        top = tk.Frame(self.root, bg="#222", height=48)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)

        self.l_pos = tk.Label(top, text="", bg="#222", fg="#ff4466", font=("Consolas",13,"bold"))
        self.l_pos.pack(side=tk.LEFT, padx=12)

        self.l_ind = tk.Label(top, text="", bg="#222", fg="#ffffff", font=("Consolas",12))
        self.l_ind.pack(side=tk.LEFT, padx=8)

        self.l_obj = tk.Label(top, text="", bg="#222", fg="#00cc66", font=("Consolas",11,"bold"))
        self.l_obj.pack(side=tk.LEFT, padx=12)

        self.l_rat = tk.Label(top, text="", bg="#222", fg="#aaaaaa", font=("Consolas",10))
        self.l_rat.pack(side=tk.LEFT, padx=12)

        tk.Label(top,
            text="D=Valider  A=Precedent  S=Skip  Z=Reset  ClikDroit=SupprBoite",
            bg="#222", fg="#555555", font=("Consolas",9)
        ).pack(side=tk.RIGHT, padx=12)

        # CANVAS
        self.cv = tk.Canvas(self.root, bg="#0a0a0a", cursor="crosshair", highlightthickness=0)
        self.cv.pack(fill=tk.BOTH, expand=True)

        # BOTTOM BAR
        bot = tk.Frame(self.root, bg="#222", height=32)
        bot.pack(fill=tk.X, side=tk.BOTTOM)
        bot.pack_propagate(False)

        self.l_stat = tk.Label(bot, text="", bg="#222", fg="#666", font=("Consolas",10))
        self.l_stat.pack(side=tk.LEFT, padx=12)

        # barre objectif verte
        tk.Label(bot, text="Objectif:", bg="#222", fg="#555", font=("Consolas",9)).pack(side=tk.RIGHT, padx=4)
        f2 = tk.Frame(bot, bg="#222"); f2.pack(side=tk.RIGHT, padx=4, pady=10)
        self.ob_bg = tk.Frame(f2, bg="#333", width=160, height=8); self.ob_bg.pack()
        self.ob    = tk.Frame(self.ob_bg, bg="#00cc66", height=8)
        self.ob.place(x=0, y=0, width=0, height=8)

        # barre globale rouge
        tk.Label(bot, text="Global:", bg="#222", fg="#555", font=("Consolas",9)).pack(side=tk.RIGHT, padx=4)
        f1 = tk.Frame(bot, bg="#222"); f1.pack(side=tk.RIGHT, padx=4, pady=10)
        self.pb_bg = tk.Frame(f1, bg="#333", width=200, height=8); self.pb_bg.pack()
        self.pb    = tk.Frame(self.pb_bg, bg="#ff4466", height=8)
        self.pb.place(x=0, y=0, width=0, height=8)

        # mouse
        self.cv.bind("<ButtonPress-1>",   self.m_press)
        self.cv.bind("<B1-Motion>",       self.m_drag)
        self.cv.bind("<ButtonRelease-1>", self.m_release)
        self.cv.bind("<ButtonPress-3>",   self.m_right)

    def _binds(self):
        self.root.bind("d", lambda e: self.valider())
        self.root.bind("D", lambda e: self.valider())
        self.root.bind("<Right>", lambda e: self.valider())
        self.root.bind("a", lambda e: self.prev())
        self.root.bind("A", lambda e: self.prev())
        self.root.bind("<Left>", lambda e: self.prev())
        self.root.bind("s", lambda e: self.skip())
        self.root.bind("S", lambda e: self.skip())
        self.root.bind("z", lambda e: self.reset())
        self.root.bind("Z", lambda e: self.reset())
        self.root.bind("<Escape>", lambda e: self.quit())

    # ── load image ─────────────────────────────────────────────

    def show(self):
        if self.idx >= self.total:
            messagebox.showinfo("Termine", "Toutes les images traitees !")
            self.root.quit(); return

        p = self.imgs[self.idx]
        lp = Path(LABELS_DIR) / (p.stem + ".txt")

        img = Image.open(p).convert("RGB")
        self.W, self.H = img.size

        self.root.update()
        cw = self.cv.winfo_width()
        ch = self.cv.winfo_height()
        self.sc = min(cw/self.W, ch/self.H) * 0.95
        dw = int(self.W*self.sc); dh = int(self.H*self.sc)
        self.ox = (cw-dw)//2; self.oy = (ch-dh)//2

        self.tk = ImageTk.PhotoImage(img.resize((dw,dh), Image.LANCZOS))
        self.boxes = read_label(lp, self.W, self.H)
        self.orig  = [list(b) for b in self.boxes]
        self.bi    = 0

        ind = p.stem.split("_")[0]
        self.val = load_validees()
        restant  = max(0, OBJECTIF - self.val)

        self.l_pos.config(text=f"{self.idx+1}/{self.total}")
        self.l_ind.config(text=f"  {ind}  |  {p.name}")

        if self.val >= OBJECTIF:
            self.l_obj.config(text=f"  OBJECTIF ATTEINT ({self.val}) → Lance 2_train_yolo.py !", fg="#ffff00")
        else:
            self.l_obj.config(text=f"  {self.val}/{OBJECTIF} validees  (encore {restant})", fg="#00cc66")

        self.pb.place(x=0, y=0, width=int((self.idx/self.total)*200), height=8)
        self.ob.place(x=0, y=0, width=int(min(self.val/OBJECTIF,1)*160), height=8)
        self.l_stat.config(text=f"Image {self.idx+1}  |  Validees par toi: {self.val}  |  Restantes: {self.total-self.idx}")

        self.redraw()

    def redraw(self):
        self.cv.delete("all")
        self.cv.create_image(self.ox, self.oy, anchor=tk.NW, image=self.tk)

        # grille
        cw = self.cv.winfo_width(); ch = self.cv.winfo_height()
        for x in range(0, cw, 100): self.cv.create_line(x,0,x,ch, fill="#181818", width=1)
        for y in range(0, ch, 100): self.cv.create_line(0,y,cw,y, fill="#181818", width=1)

        # boxes
        for i,(x1,y1,x2,y2) in enumerate(self.boxes):
            bx1=int(x1*self.sc)+self.ox; by1=int(y1*self.sc)+self.oy
            bx2=int(x2*self.sc)+self.ox; by2=int(y2*self.sc)+self.oy
            col = "#00ff88" if i==self.bi else "#4499ff"
            self.cv.create_rectangle(bx1,by1,bx2,by2, outline=col, width=2)
            for px,py in [(bx1,by1),(bx2,by1),(bx1,by2),(bx2,by2),
                          ((bx1+bx2)//2,by1),((bx1+bx2)//2,by2),
                          (bx1,(by1+by2)//2),(bx2,(by1+by2)//2)]:
                self.cv.create_rectangle(px-5,py-5,px+5,py+5, fill=col, outline="#fff")
            r = (x2-x1)/(y2-y1) if (y2-y1)>0 else 0
            self.cv.create_text(bx1+4, by1+4, text=f"{r:.2f}", anchor=tk.NW, fill=col, font=("Consolas",9))
            self.l_rat.config(text=f"Ratio L/H: {r:.2f}  (vise 1.2-1.5)")

        if not self.boxes:
            self.cv.create_text(cw//2, ch//2,
                text="Aucune boite — Glisse pour creer — S pour skipper",
                fill="#ff4466", font=("Consolas",13))
            self.l_rat.config(text="Aucune boite")

    # ── coords ─────────────────────────────────────────────────

    def to_img(self, cx, cy):
        return (cx-self.ox)/self.sc, (cy-self.oy)/self.sc

    def get_handle(self, cx, cy, x1, y1, x2, y2):
        bx1=int(x1*self.sc)+self.ox; by1=int(y1*self.sc)+self.oy
        bx2=int(x2*self.sc)+self.ox; by2=int(y2*self.sc)+self.oy
        mx=(bx1+bx2)//2; my=(by1+by2)//2
        for n,(px,py) in [('tl',(bx1,by1)),('tr',(bx2,by1)),('bl',(bx1,by2)),('br',(bx2,by2)),
                           ('tm',(mx,by1)), ('bm',(mx,by2)), ('lm',(bx1,my)), ('rm',(bx2,my))]:
            if abs(cx-px)<12 and abs(cy-py)<12: return n
        return None

    # ── mouse ──────────────────────────────────────────────────

    def m_press(self, e):
        cx,cy = e.x, e.y
        for i,(x1,y1,x2,y2) in enumerate(self.boxes):
            h = self.get_handle(cx,cy,x1,y1,x2,y2)
            if h:
                self.drag=h; self.dstart=(cx,cy); self.dbefore=list(self.boxes[i]); self.bi=i; return
            bx1=int(x1*self.sc)+self.ox; by1=int(y1*self.sc)+self.oy
            bx2=int(x2*self.sc)+self.ox; by2=int(y2*self.sc)+self.oy
            if bx1<cx<bx2 and by1<cy<by2:
                self.drag='move'; self.dstart=(cx,cy); self.dbefore=list(self.boxes[i]); self.bi=i; return
        self.drawing=True; self.dpos=self.to_img(cx,cy)

    def m_drag(self, e):
        cx,cy = e.x, e.y
        if self.drawing and self.dpos:
            ix,iy = self.to_img(cx,cy)
            x1=min(self.dpos[0],ix); y1=min(self.dpos[1],iy)
            x2=max(self.dpos[0],ix); y2=max(self.dpos[1],iy)
            self.cv.delete("all")
            self.cv.create_image(self.ox,self.oy,anchor=tk.NW,image=self.tk)
            for bx1,by1,bx2,by2 in self.boxes:
                self.cv.create_rectangle(int(bx1*self.sc)+self.ox,int(by1*self.sc)+self.oy,
                                          int(bx2*self.sc)+self.ox,int(by2*self.sc)+self.oy,outline="#4499ff",width=2)
            self.cv.create_rectangle(int(x1*self.sc)+self.ox,int(y1*self.sc)+self.oy,
                                      int(x2*self.sc)+self.ox,int(y2*self.sc)+self.oy,outline="#ffff00",width=2)
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
            ix,iy = self.to_img(e.x,e.y)
            x1=min(self.dpos[0],ix); y1=min(self.dpos[1],iy)
            x2=max(self.dpos[0],ix); y2=max(self.dpos[1],iy)
            if abs(x2-x1)>10 and abs(y2-y1)>10:
                self.boxes.append([x1,y1,x2,y2]); self.bi=len(self.boxes)-1
            self.drawing=False; self.dpos=None; self.redraw()
        self.drag=None; self.dstart=None

    def m_right(self, e):
        cx,cy = e.x, e.y
        for i,(x1,y1,x2,y2) in enumerate(self.boxes):
            bx1=int(x1*self.sc)+self.ox; by1=int(y1*self.sc)+self.oy
            bx2=int(x2*self.sc)+self.ox; by2=int(y2*self.sc)+self.oy
            if bx1<cx<bx2 and by1<cy<by2:
                self.boxes.pop(i); self.bi=max(0,len(self.boxes)-1); self.redraw(); return

    # ── actions ────────────────────────────────────────────────

    def valider(self):
        lp = Path(LABELS_DIR)/(self.imgs[self.idx].stem+".txt")
        write_label(lp, self.boxes, self.W, self.H)
        inc_validees()
        self.idx+=1; save_progress(self.idx); self.bi=0; self.show()

    def skip(self):
        lp = Path(LABELS_DIR)/(self.imgs[self.idx].stem+".txt")
        open(lp,'w').close()
        self.idx+=1; save_progress(self.idx); self.bi=0; self.show()

    def prev(self):
        if self.idx>0: self.idx-=1; save_progress(self.idx); self.bi=0; self.show()

    def reset(self):
        self.boxes=[list(b) for b in self.orig]; self.bi=0; self.redraw()

    def quit(self):
        save_progress(self.idx)
        if messagebox.askyesno("Quitter", f"Quitter ? Reprise depuis image {self.idx+1}."):
            self.root.quit()

if __name__=="__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()