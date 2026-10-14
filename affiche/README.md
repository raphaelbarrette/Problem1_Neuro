# Affiche — Problème 1 (GEI723, automne 2026)

## Compiler

Il faut **XeLaTeX** (ou LuaLaTeX), à cause des polices.

```
./build.sh
```

ou à la main :

```
xelatex affiche && bibtex affiche && xelatex affiche && xelatex affiche
xelatex annexe
xelatex affiche_complete
```

Sur **Overleaf** : téléverser tout le dossier, puis *Menu → Compiler → XeLaTeX*. Compiler `affiche.tex`
et `annexe.tex`, télécharger les deux PDF (ou compiler `affiche_complete.tex` après les avoir ajoutés).

## Régénérer les figures

Depuis le dossier parent (`Code/`) :

```
python poster_figs.py      # figures de l'affiche (Partie I et II)
python poster_figs.py arene
python annexe_figs.py      # figures de l'annexe
```

Les simulations sont mises en cache dans `cache_affiche/` ; effacer ce dossier pour tout recalculer. Les figures sont écrites directement dans `affiche/figures/`, puis `cd affiche && ./build.sh`.

