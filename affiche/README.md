# Affiche — Problème 1 (GEI723, automne 2026)

Affiche scientifique A0 portrait en LaTeX (`tikzposter`), page 1 + annexe (page 2).

## Fichiers

| Fichier | Rôle |
|---|---|
| `affiche.tex` | Page 1 : l'affiche |
| `annexe.tex` | Page 2 : paramètres, justifications, expériences complémentaires |
| `affiche_complete.tex` | Réunit les deux pages dans `affiche_complete.pdf` (le PDF à remettre) |
| `strategie.tex` | Notes pour la défense : démarche étape par étape et sources (document lettre, 4 pages) |
| `style.tex` | Style commun : polices, couleurs, commandes (`\titre`, `\legende`, `\encadre`…) |
| `schema_partie1.tex`, `schema_partie2.tex` | Schémas du réseau en TikZ |
| `references.bib` | Références (BibTeX, style `unsrt`) |
| `figures/` | Figures en PDF vectoriel (et PNG), produites par `poster_figs.py` et `annexe_figs.py` (dossier parent) |
| `fonts/` | Polices IBM Plex (licence SIL OFL, voir `fonts/OFL-license.txt`) |
| `build.sh` | Compile tout |

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

Les simulations sont mises en cache dans `cache_affiche/` ; effacer ce dossier pour tout recalculer
(compter environ une heure). Les figures sont écrites directement dans `affiche/figures/`, puis `cd affiche && ./build.sh`.

## À compléter

- Nom d'Émile : chercher `[NOM]` dans `affiche.tex` et `annexe.tex`.
- Remerciements : chercher `[À compléter ou retirer]` dans `affiche.tex`.
