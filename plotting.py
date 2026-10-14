'''Figures communes : raster + demarche + trajectoire (Partie I), raster complet d'un scenario
(Partie II), et sauvegarde des figures dans figures/.'''
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms

from cpg.config import CAPTEURS, CMD_NOMS
from cpg.body_model import decoder, pos_a

DOSSIER = Path(__file__).parent / 'figures'
COUL = dict(G='C0', D='C1', AV='C3', AR='C2')


def sauver(fig, nom):
    '''Enregistre la figure dans figures/<nom>.png.'''
    DOSSIER.mkdir(exist_ok=True)
    fig.savefig(DOSSIER / f'{nom}.png', dpi=120, bbox_inches='tight')


def demarche(ax, contact, pattes):
    '''Diagramme de demarche (barre = patte au sol), couleur = groupe du CPG (k + s) % 2.'''
    for r, p in enumerate(pattes):
        ax.broken_barh([(a, b - a) for a, b in contact[p]], (r - 0.4, 0.8), color=f'C{(p // 2 + p % 2) % 2}')


def afficher_marche(R, titre):
    '''Partie I : raster du CPG, diagramme de demarche et trajectoire. Renvoie (fig, traj).'''
    traj, contact = decoder(R)
    n = R['n']
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5), gridspec_kw={'width_ratios': [1, 2, 1]})
    t, i = R['CPG']
    ax[0].plot(t, i, '|', ms=20, c='k')
    ax[0].set(yticks=[0, 1], yticklabels=['A', 'B'], xlabel='Temps (ms)', title='CPG', xlim=(0, 1000))
    pattes = range(n) if n <= 12 else range(0, n, 2)       # grand N : cote gauche seulement
    demarche(ax[1], contact, pattes)
    ax[1].set(yticks=range(n) if n <= 12 else [], ylabel='Patte' if n <= 12 else 'Pattes gauches (avant -> arriere)',
              xlabel='Temps (ms)', title='Diagramme de demarche (barre = patte au sol)')
    if n <= 12: ax[1].set_yticklabels([f"{'GD'[p % 2]}{p // 2 + 1}" for p in pattes])
    ax[1].invert_yaxis()
    ax[2].plot(traj[:, 1], traj[:, 2]); ax[2].plot(0, 0, 'ko')
    ax[2].set_aspect('equal', adjustable='datalim'); ax[2].set(xlabel='x', ylabel='y', title='Trajectoire')
    fig.suptitle(titre); fig.tight_layout()
    return fig, traj


def afficher_scenario(R, titre, stimuli=(), traj=True):
    '''Partie II : raster complet (capteurs -> commandes -> compteurs -> CPG -> pattes) avec la
    trajectoire a droite. Les bandes colorees marquent les stimuli. Renvoie la figure.'''
    M, n = R['M'], R['n']
    fig = plt.figure(figsize=(16, 7))
    g = fig.add_gridspec(5, 2, width_ratios=[2.3, 1], height_ratios=[3, 1.6, 1, 1, 2.6], hspace=0.08, wspace=0.15)
    ax = [fig.add_subplot(g[0, 0])]
    ax += [fig.add_subplot(g[r, 0], sharex=ax[0]) for r in range(1, 5 if R['pattes'] else 4)]
    for c, cap in enumerate(CAPTEURS):
        t, i = R['S_' + cap]
        ax[0].plot(t, i + c * M, '|', c=COUL[cap], ms=3)
    ax[0].set(yticks=[M / 2 + c * M for c in range(4)], yticklabels=['SENS_' + c for c in CAPTEURS], ylim=(-1, 4 * M))
    t, i = R['CMD']
    ax[1].plot(t, i, '|', c='k', ms=7)
    ax[1].set(yticks=range(4), yticklabels=CMD_NOMS, ylim=(-0.6, 3.6))
    for (nom, col) in [('CPT', 'C4'), ('INH', 'C5')]:
        t, i = R[nom]
        ax[2].plot(t, i + (2 if nom == 'INH' else 0), '|', c=col, ms=7)
    ax[2].set(yticks=range(4), yticklabels=['CPT_G', 'CPT_D', 'INH_G', 'INH_D'], ylim=(-0.6, 3.6))
    t, i = R['CPG']
    ax[3].plot(t, i, '|', c='k', ms=10)
    ax[3].set(yticks=[0, 1], yticklabels=['A', 'B'], ylim=(-0.6, 1.6))
    if R['pattes']:
        pattes = range(n) if n <= 12 else range(0, n, 2 * max(1, n // 24))
        for r, p in enumerate(pattes):
            for nom, mk, col in [('AVANT', '|', 'C2'), ('ARRIERE', '|', 'C3')]:
                t, i = R[nom]
                ax[4].plot(t[i == p], np.full(np.sum(i == p), r), mk, c=col, ms=5)
        ax[4].set(yticks=range(len(pattes)) if n <= 12 else [], ylabel='' if n <= 12 else 'pattes G',
                  ylim=(len(pattes) - 0.5, -0.5))
        if n <= 12: ax[4].set_yticklabels([f"{'GD'[p % 2]}{p // 2 + 1}" for p in pattes])
        ax[4].plot([], [], '|', c='C2', label='AVANT'); ax[4].plot([], [], '|', c='C3', label='ARRIERE')
        ax[4].legend(loc='upper right', fontsize=8, ncol=2)
    for a in ax:
        for cap, t0, inten, dur in stimuli:
            a.axvspan(t0 / ms, (t0 + dur) / ms, color=COUL[cap], alpha=0.15)
        if a is not ax[-1]: plt.setp(a.get_xticklabels(), visible=False)
    ax[-1].set(xlabel='Temps (ms)', xlim=(0, R['duree']))
    ax[0].set_title(titre)
    if R['pattes'] and traj:
        tr = decoder(R)[0]
        at = fig.add_subplot(g[:, 1])
        at.plot(tr[:, 1], tr[:, 2], c='0.7', lw=1)
        sc = at.scatter(tr[:, 1], tr[:, 2], c=tr[:, 0] / 1000, s=6, cmap='viridis')
        at.plot(0, 0, 'ko')
        for cap, t0, inten, dur in stimuli:
            at.plot(*pos_a(tr, t0 / ms), '*', c=COUL[cap], ms=14, mec='k', label=f'contact_{cap} ({inten})')
        at.set_aspect('equal', adjustable='datalim'); at.set(xlabel='x', ylabel='y', title='Trajectoire')
        fig.colorbar(sc, ax=at, label='temps (s)', shrink=0.6)
        if stimuli: at.legend(fontsize=8)
    return fig
