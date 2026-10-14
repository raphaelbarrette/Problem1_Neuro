Repo for the first problem of GEI723

## Structure

- `cpg/config.py` : modele LIF commun et parametres (Partie I et `PARAM` de la Partie II)
- `cpg/network.py` : populations, synapses, pattes ; `simuler` (Partie I, commandes a la main)
- `cpg/sensors.py` : `simuler_capteurs` (Partie II), `recrutement`, `arene` (boucle fermee)
- `cpg/body_model.py` : decodage des decharges en trajectoire, mesures (frequence de pas, etc.)
- `plotting.py` : figures communes, sauvegarde dans `figures/`
- `experiments/` : une experience par fichier
- `poster_figs.py`, `annexe_figs.py` : figures de l'affiche et de l'annexe (PDF + PNG dans `affiche/figures/`)
- `affiche/` : affiche A0 en LaTeX (tikzposter, XeLaTeX) ; voir `affiche/README.md`

## Utilisation (depuis ce dossier)

```
python -m experiments.exp_forward_backward
python -m experiments.exp_part2_stimuli 3 5 arene   # etapes choisies (toutes par defaut)
python run_all.py                                    # tout, figures dans figures/ sans affichage
```

## Lien biologique (references a verifier avant de les citer)

- **Reflexe de fuite de la blatte** : des poils filiformes sur les cerques (appendices a l'arriere
  de l'abdomen) detectent les mouvements d'air et, via les interneurones geants, declenchent une
  course rapide. Analogue de la voie SENS_AR -> CPG (acceleration).
  Camhi, J. M. (1980), *The escape system of the cockroach*, Scientific American 243(6) ;
  Ritzmann, R. E. & Eaton, R. C. (1997), *Neural substrates for initiation of startle responses*.
- **Antennes et suivi de mur** : *Periplaneta americana* suit les murs au contact de ses antennes
  et tourne selon la distance percue. Analogue des voies laterales SENS_G/D -> compteurs (virage
  gradue selon l'intensite).
  Camhi, J. M. & Johnson, E. N. (1999), *High-frequency steering maneuvers mediated by tactile
  cues: antennal wall-following in the cockroach*, J. Exp. Biol. 202 ;
  Cowan, N. J. et al. (2006), *Task-level control of rapid wall following in the American
  cockroach*, J. Exp. Biol. 209.
- **Selection d'action par inhibition** : le choix par inhibition laterale / winner-take-all est
  un motif classique des circuits moteurs des invertebres (ici entre capteurs, et CMD_AV <-> CMD_AR).

## Affiche

```
python poster_figs.py && python poster_figs.py arene && python annexe_figs.py   # figures (cache dans cache_affiche/)
cd affiche && ./build.sh                                                            # affiche_complete.pdf (2 pages A0)
```
