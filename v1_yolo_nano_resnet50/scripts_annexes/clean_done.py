# =============================================================================
# clean_done.py
# Verifie toutes les images de done.txt, retire celles sans boite,
# et affiche les stats completes. Aucune action manuelle requise.
# =============================================================================

from pathlib import Path
from collections import defaultdict

LABELS_DIR = r"D:\OrangIdentifier\DATASET_YOLO\labels"
DONE_FILE  = r"D:\OrangIdentifier\DATASET_YOLO\done.txt"

# Lecture de done.txt
done_path = Path(DONE_FILE)
if not done_path.exists():
    print("ERREUR : done.txt introuvable.")
    exit(1)

stems_originaux = [s.strip() for s in done_path.read_text().splitlines() if s.strip()]
labels_dir = Path(LABELS_DIR)

# Classification de chaque stem
avec_boite    = []
sans_boite    = []
sans_label    = []

for stem in stems_originaux:
    lbl = labels_dir / (stem + ".txt")
    if not lbl.exists():
        sans_label.append(stem)
    elif lbl.stat().st_size == 0:
        sans_boite.append(stem)
    else:
        # Verifier que le label est valide (au moins une ligne correcte)
        lignes_valides = []
        for ligne in lbl.read_text().splitlines():
            parts = ligne.strip().split()
            if len(parts) == 5:
                try:
                    vals = list(map(float, parts))
                    if all(0 <= v <= 1 for v in vals[1:]):
                        lignes_valides.append(ligne)
                except ValueError:
                    pass
        if lignes_valides:
            avec_boite.append(stem)
        else:
            sans_boite.append(stem)

# Stats par individu
valides_par_ind = defaultdict(int)
retires_par_ind = defaultdict(int)

for stem in avec_boite:
    valides_par_ind[stem.split("_")[0]] += 1
for stem in sans_boite + sans_label:
    retires_par_ind[stem.split("_")[0]] += 1

print("=" * 60)
print("NETTOYAGE DE done.txt")
print("=" * 60)
print(f"  Stems originaux          : {len(stems_originaux)}")
print(f"  Avec boite (conserves)   : {len(avec_boite)}")
print(f"  Sans boite (retires)     : {len(sans_boite)}")
print(f"  Sans label (retires)     : {len(sans_label)}")
print(f"  Total retire             : {len(sans_boite) + len(sans_label)}")

print("\nDistribution valides par individu apres nettoyage :")
print(f"  {'Individu':<30} {'Valides':>8} {'Retires':>8}")
print(f"  {'-'*30} {'-'*8} {'-'*8}")
tous_individus = sorted(set(list(valides_par_ind.keys()) + list(retires_par_ind.keys())))
for ind in tous_individus:
    v = valides_par_ind.get(ind, 0)
    r = retires_par_ind.get(ind, 0)
    statut = "OK" if v >= 30 else f"manque {30 - v}"
    print(f"  {ind:<30} {v:>8} {r:>8}  {statut}")

print(f"\n  Total valides final      : {len(avec_boite)}")

# Ecriture du done.txt nettoye
done_path.write_text("\n".join(avec_boite) + "\n")
print(f"\ndone.txt mis a jour : {len(avec_boite)} stems conserves.")
print("=" * 60)