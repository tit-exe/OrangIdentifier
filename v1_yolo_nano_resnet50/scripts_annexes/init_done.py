# =============================================================================
# init_done.py
# Initialise done.txt avec les 500 premieres images de la liste originale
# qui ont au moins une boite (label non vide).
# =============================================================================

from pathlib import Path
from collections import defaultdict

IMAGES_DIR = r"D:\OrangIdentifier\DATASET_YOLO\images"
LABELS_DIR = r"D:\OrangIdentifier\DATASET_YOLO\labels"
DONE_FILE  = r"D:\OrangIdentifier\DATASET_YOLO\done.txt"

PRIORITE   = 30
NB_TRAITES = 500

# Reconstituer la liste dans le meme ordre que l'ancien annotateur
par_ind = defaultdict(list)
for p in sorted(Path(IMAGES_DIR).glob("*.jpg")):
    par_ind[p.stem.split("_")[0]].append(p)

prio, reste = [], []
for k, v in sorted(par_ind.items()):
    prio.extend(v[:PRIORITE])
    reste.extend(v[PRIORITE:])

liste    = prio + reste
traitees = liste[:NB_TRAITES]

print("=" * 55)
print("INITIALISATION DE done.txt")
print("=" * 55)
print(f"Premiere image : {traitees[0].stem}")
print(f"Derniere image : {traitees[-1].stem}")

labels_dir = Path(LABELS_DIR)
avec_boite = []
sans_boite = []

for img in traitees:
    lbl = labels_dir / (img.stem + ".txt")
    if lbl.exists() and lbl.stat().st_size > 0:
        avec_boite.append(img.stem)
    else:
        sans_boite.append(img.stem)

print(f"\nParmi les {NB_TRAITES} images traitees :")
print(f"  Avec boite (valides) : {len(avec_boite)}")
print(f"  Sans boite (skips)   : {len(sans_boite)}")

print("\nDistribution valides par individu :")
par_ind_v = defaultdict(int)
for stem in avec_boite:
    par_ind_v[stem.split("_")[0]] += 1

for ind, nb in sorted(par_ind_v.items()):
    statut = "OK" if nb >= PRIORITE else f"manque {PRIORITE - nb}"
    print(f"  {ind:<30} {nb:>4}  {statut}")

done_path = Path(DONE_FILE)
if done_path.exists():
    print(f"\ndone.txt existe deja.")
    confirm = input("Ecraser ? (o/n) : ").strip().lower()
    if confirm != 'o':
        print("Annule.")
        exit(0)

with open(done_path, 'w') as f:
    for img in traitees:
        f.write(img.stem + "\n")

print(f"\ndone.txt ecrit : {NB_TRAITES} stems.")
print("Lance maintenant l'annotateur.")
print("=" * 55)