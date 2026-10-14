'''Figures de l'affiche (style uniforme, lisibles en A0), en PNG et PDF dans affiche/figures/.

Usage (depuis Code/) : python poster_figs.py [p1] [p2] [arene]   (p1 et p2 par defaut)
Les simulations sont mises en cache dans cache_affiche/ ; effacer ce dossier pour tout recalculer.
'''
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from brian2 import ms, second

from cpg import decoder, simuler, simuler_capteurs, freq_pas, activite, duree_active, pos_a, recrutement, arene
from cpg.config import PARAM

OUT = Path(__file__).parent / 'affiche' / 'figures'   # utilisees directement par affiche/affiche.tex
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(__file__).parent / 'cache_affiche'
CACHE.mkdir(exist_ok=True)

# Police IBM Plex Sans (fournie dans affiche/fonts) ; sinon matplotlib garde sa police par defaut.
for dossier in [Path(__file__).parent / 'affiche' / 'fonts']:
    for f in dossier.glob('*.ttf'):
        font_manager.fontManager.addfont(str(f))
POLICE = 'IBM Plex Sans' if any('IBM Plex Sans' == f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'

INK, INK2, MUTED, GRID, BG = '#16161a', '#4a4a52', '#8a8a92', '#e4e2dc', '#ffffff'
BLEU, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
SERIES = [BLEU, ORANGE, AQUA, VIOLET]
COUL_CAP = dict(G=BLEU, D=ORANGE, AV=AQUA, AR=VIOLET)
W = 6.28            # largeur d'une colonne de l'affiche (pouces, affichee a 150 px/po)

plt.rcParams.update({
    'font.family': POLICE, 'font.size': 14, 'axes.titlesize': 15, 'axes.labelsize': 14,
    'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 12.5,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left', 'axes.titlepad': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 1, 'axes.axisbelow': True, 'lines.linewidth': 2.2, 'legend.frameon': False,
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
})


def sauver(fig, nom):
    '''PNG (300 dpi) et PDF vectoriel (pour l'affiche LaTeX).'''
    fig.savefig(OUT / f'{nom}.png', dpi=300, bbox_inches='tight', pad_inches=0.06)
    fig.savefig(OUT / f'{nom}.pdf', bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print('  ->', nom)


def cache(nom, fn):
    p = CACHE / f'{nom}.pkl'
    if p.exists(): return pickle.loads(p.read_bytes())
    r = fn(); p.write_bytes(pickle.dumps(r)); return r


def label_patte(p): return f"{'GD'[p % 2]}{p // 2 + 1}"


# ======================================================================= Partie I
def fig_cpg_demarche():
    R = cache('avant', lambda: simuler(mode_G='avant', mode_D='avant'))
    traj, contact = decoder(R)
    fig, ax = plt.subplots(2, 1, figsize=(W, 4.1), sharex=True, gridspec_kw={'height_ratios': [1, 3.2], 'hspace': 0.36})
    t, i = R['CPG']
    for k, c in [(0, BLEU), (1, ORANGE)]:
        ax[0].vlines(t[i == k], k - 0.38, k + 0.38, color=c, lw=3)
    ax[0].set(yticks=[0, 1], yticklabels=['A', 'B'], ylim=(-0.6, 1.6))
    ax[0].set_title('CPG : une décharge par phase, A et B en alternance')
    ax[0].grid(False)
    for r, p in enumerate(range(6)):
        g = (p // 2 + p % 2) % 2
        ax[1].broken_barh([(a, b - a) for a, b in contact[p]], (r - 0.36, 0.72), color=[BLEU, ORANGE][g], lw=0)
    ax[1].set(yticks=range(6), yticklabels=[label_patte(p) for p in range(6)], xlim=(0, 1000), xlabel='Temps (ms)')
    ax[1].invert_yaxis(); ax[1].grid(axis='y', visible=False)
    ax[1].set_title('Démarche tripode (barre = patte au sol)')
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color=BLEU, label='tripode X : G1, D2, G3'), Patch(color=ORANGE, label='tripode Y : D1, G2, D3')],
                 loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.2, columnspacing=1.4, fontsize=12.5)
    sauver(fig, 'p1_cpg_demarche')


def fig_trajectoires():
    Ra = cache('avant', lambda: simuler(mode_G='avant', mode_D='avant'))
    Rr = cache('arriere', lambda: simuler(mode_G='arriere', mode_D='arriere'))
    Rt = cache('rotation', lambda: simuler(mode_G='avant', mode_D='arriere'))
    ta, tr, tt = decoder(Ra)[0], decoder(Rr)[0], decoder(Rt)[0]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={'wspace': 0.42})
    ax[0].step(ta[:, 0] / 1000, ta[:, 1], where='post', c=BLEU)
    ax[0].step(tr[:, 0] / 1000, tr[:, 1], where='post', c=ORANGE)
    ax[0].text(2.05, ta[-1, 1], 'avant', color=BLEU, va='center', fontweight='semibold')
    ax[0].text(2.05, tr[-1, 1], 'arrière', color=ORANGE, va='center', fontweight='semibold')
    ax[0].set(xlabel='Temps (s)', ylabel='Position x', xlim=(0, 2.55), yticks=[-2, -1, 0, 1, 2])
    ax[0].set_title('Marche avant / arrière')
    ax[1].step(tt[:, 0] / 1000, np.degrees(tt[:, 3]), where='post', c=AQUA)
    ax[1].set(xlabel='Temps (s)', ylabel='Orientation θ (°)', xlim=(0, 2))
    ax[1].set_title('Rotation sur place')
    ax[1].text(0.06, -225, f"|x|, |y| < 0,04\nθ final = {np.degrees(tt[-1, 3]):.0f}°", fontsize=12, color=INK2)
    sauver(fig, 'p1_trajectoires')
    return ta[-1], tr[-1], tt[-1]


def fig_cercles():
    res = {}
    for n_s in [1, 2, 3]:
        R = cache(f'cercle{n_s}', lambda n_s=n_s: simuler(virage_D=True, n_saut=n_s, duree=8*second))
        tr = decoder(R)[0]
        dist = np.sum(np.hypot(np.diff(tr[:, 1]), np.diff(tr[:, 2])))
        res[n_s] = (tr, dist / abs(tr[-1, 3]))
    fig, ax = plt.subplots(figsize=(W * 0.47, W * 0.5))
    for c, (n_s, (tr, Rm)) in zip(SERIES, res.items()):
        Rt = 0.5 * (2 * n_s - 1)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(Rt * np.cos(th), -Rt + Rt * np.sin(th), ls=(0, (3, 3)), c=MUTED, lw=1.2)
        ax.plot(tr[:, 1], tr[:, 2], c=c, lw=2.6)
        lx, ly = {1: (-2.35, -0.55), 2: (-0.45, -1.75), 3: (2.2, -4.05)}[n_s]
        ax.text(lx, ly, f'n = {n_s}', color=c, fontweight='semibold', va='center')
    ax.plot(0, 0, 'o', c=INK, ms=7)
    ax.set_aspect('equal'); ax.set(xlabel='x (L)', ylabel='y (L)', xlim=(-2.8, 3.3))
    ax.set_title('Virage à droite, 8 s', fontsize=14)
    ax.plot([], [], ls=(0, (3, 3)), c=MUTED, lw=1.2, label='théorie')
    ax.legend(loc='lower left', fontsize=12)
    sauver(fig, 'p1_cercles')
    return {n: r for n, (_, r) in res.items()}


def fig_vitesse():
    I_vals = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    def run():
        out = []
        for I_d in I_vals:
            R = simuler(I_drive=I_d)
            tr = decoder(R)[0]
            t, i = R['CPG']; ta = t[i == 0]
            out.append((1000 / np.mean(np.diff(ta[1:])), tr[-1, 1] / (R['duree'] / 1000)))
        return out
    res = np.array(cache('vitesse2', run))
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.1), gridspec_kw={'wspace': 0.42})
    ax[0].plot(I_vals, res[:, 0], 'o-', c=BLEU, ms=8)
    ax[0].set(xlabel='I_drive', ylabel='Cycles par seconde (Hz)', xticks=[1.5, 2, 2.5, 3]); ax[0].set_title('Fréquence du CPG')
    ax[1].plot(I_vals, res[:, 1], 'o-', c=BLEU, ms=8)
    ax[1].set(xlabel='I_drive', ylabel='Vitesse (unités / s)', xticks=[1.5, 2, 2.5, 3]); ax[1].set_title('Vitesse du corps')
    sauver(fig, 'p1_vitesse')
    return dict(zip(I_vals, res.tolist()))


def fig_N():
    R = cache('N200', lambda: simuler(N_PAIRES=200, delta=0.5*ms))
    contact = decoder(R)[1]
    n = R['n']
    pattes = list(range(0, n, 2))[::5]       # pattes gauches, une sur 5 (groupes alternes)
    fig, ax = plt.subplots(figsize=(W, 2.0))
    for r, p in enumerate(pattes):
        g = (p // 2 + p % 2) % 2
        ax.broken_barh([(a, b - a) for a, b in contact[p]], (r - 0.42, 0.84), color=[BLEU, ORANGE][g], lw=0)
    ax.set(xlim=(0, 1000), xlabel='Temps (ms)', yticks=[0, len(pattes) - 1], yticklabels=['G1', f'G{pattes[-1] // 2 + 1}'])
    ax.invert_yaxis(); ax.grid(axis='y', visible=False)
    ax.set_title('200 paires (pattes gauches, 1 sur 5), Δ = 0,5 ms')
    sauver(fig, 'p1_N200')


# ======================================================================= Partie II
def fig_raster_recul():
    stim = [('AV', 1000*ms, 0.8, 100*ms)]
    R = cache('p2_recul08', lambda: simuler_capteurs(stim, duree=4*second))
    tr = decoder(R)[0]
    M = R['M']
    fig, ax = plt.subplots(5, 1, figsize=(W, 3.95), sharex=True,
                           gridspec_kw={'height_ratios': [1.3, 1, 0.8, 1.3, 1.6], 'hspace': 0.18})
    for a in ax: a.axvspan(1, 1.1, color=AQUA, alpha=0.18, lw=0); a.grid(False)
    t, i = R['S_AV']; ax[0].vlines(t / 1000, i - 0.4, i + 0.4, color=AQUA, lw=1.6)
    ax[0].set(yticks=[M / 2 - 0.5], yticklabels=['SENS_AV\n(10 n.)'], ylim=(-1, M))
    t, i = R['CMD']
    for k, (nom, c) in enumerate([('CMD_AV_G', BLEU), ('CMD_AR_G', ORANGE)]):
        ax[1].vlines(t[i == k] / 1000, k - 0.38, k + 0.38, color=c, lw=0.9)
    ax[1].set(yticks=[0, 1], yticklabels=['CMD_AV_G', 'CMD_AR_G'], ylim=(-0.6, 1.6))
    t, i = R['CPG']
    for k, c in [(0, INK2), (1, MUTED)]: ax[2].vlines(t[i == k] / 1000, k - 0.38, k + 0.38, color=c, lw=2)
    ax[2].set(yticks=[0, 1], yticklabels=['A', 'B'], ylim=(-0.6, 1.6))
    for nom, k, c in [('AVANT', 0, BLEU), ('ARRIERE', 1, ORANGE)]:
        t, i = R[nom]; m = i == 0
        ax[3].vlines(t[m] / 1000, k - 0.38, k + 0.38, color=c, lw=2)
    ax[3].set(yticks=[0, 1], yticklabels=['AVANT G1', 'ARRIÈRE G1'], ylim=(-0.6, 1.6))
    ax[4].step(tr[:, 0] / 1000, tr[:, 1], where='post', c=INK)
    ax[4].set(ylabel='x', xlabel='Temps (s)', xlim=(0, 4)); ax[4].grid(True)
    ax[0].set_title('Contact de face (intensité 0,8 ; 100 ms)')
    sauver(fig, 'p2_raster_recul')
    t, i = R['CMD']
    tar = t[i == 1]
    return dict(debut=tar.min() - 1000 if len(tar) else None, duree=(tar.max() - tar.min()) if len(tar) else 0, xmax=tr[:, 1].max(), xf=tr[-1, 1])


def fig_recul_intensites():
    I_vals = [0.3, 0.5, 0.8, 1.0]
    res = {I: cache(f'p2_recul_{I}', lambda I=I: simuler_capteurs([('AV', 1000*ms, I, 100*ms)], duree=4*second)) for I in I_vals}
    Iv = np.round(np.linspace(0.1, 1.0, 10), 2)
    dur = cache('p2_duree_recul', lambda: [duree_active(simuler_capteurs([('AV', 200*ms, I, 100*ms)], duree=3*second, pattes=False), 'CMD', 1, 200) for I in Iv])
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={'wspace': 0.42, 'width_ratios': [1.35, 1]})
    for c, (I, R) in zip(SERIES, res.items()):
        tr = decoder(R)[0]
        ax[0].step(tr[:, 0] / 1000, tr[:, 1], where='post', c=c, label=str(I).replace('.', ','))
    ax[0].axvspan(1, 1.1, color=AQUA, alpha=0.18, lw=0)
    ax[0].set(xlabel='Temps (s)', ylabel='Position x')
    ax[0].legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), fontsize=11.5, ncol=4, handlelength=1.0, columnspacing=0.7, handletextpad=0.3)
    ax[1].plot(Iv, np.array(dur) / 1000, 'o-', c=AQUA, ms=7)
    ax[1].set(xlabel='Intensité', ylabel='Durée du recul (s)'); ax[1].set_title('Durée du recul')
    sauver(fig, 'p2_recul_intensites')
    return dict(zip(Iv.tolist(), (np.array(dur) / 1000).tolist()))


def fig_virages():
    res = {I: cache(f'p2_virD_{I}', lambda I=I: simuler_capteurs([('D', 500*ms, I, 100*ms)], duree=5*second)) for I in [0.3, 0.6, 1.0]}
    RG = cache('p2_virG_0.6', lambda: simuler_capteurs([('G', 500*ms, 0.6, 100*ms)], duree=5*second))
    fig, ax = plt.subplots(figsize=(W * 0.62, W * 0.62))
    noms = {0.3: 'douce', 0.6: 'serrée', 1.0: 'pivot'}
    out = {}
    for c, (I, R) in zip(SERIES, res.items()):
        tr = decoder(R)[0]
        out[I] = np.degrees(tr[-1, 3])
        ax.plot(tr[:, 1], tr[:, 2], c=c, lw=2.6, label=f"D {str(I).replace('.', ',')} ({noms[I]})")
        ax.plot(*pos_a(tr, 500), '*', c=c, ms=15, mec=INK, mew=0.8)
    trg = decoder(RG)[0]
    out['G0.6'] = np.degrees(trg[-1, 3])
    ax.plot(trg[:, 1], trg[:, 2], c=VIOLET, lw=2, ls=(0, (4, 2)), label='G 0,6 (symétrie)')
    ax.plot(0, 0, 'o', c=INK, ms=7)
    ax.set_aspect('equal', adjustable='datalim'); ax.set(xlabel='x', ylabel='y')
    ax.set_title('Contact sur un flanc → tourner')
    ax.legend(loc='best', fontsize=11.5)
    sauver(fig, 'p2_virages')
    return out


def fig_acceleration():
    Iv = [0.3, 0.6, 1.0]
    def run():
        out = {}
        for I in Iv:
            R = simuler_capteurs([('AR', 1000*ms, I, 100*ms)], duree=3.5*second, pattes=False)
            out[I] = freq_pas(R)
        return out
    res = cache('p2_accel', run)
    fig, ax = plt.subplots(figsize=(W, 2.15))
    for c, (I, (t, f)) in zip(SERIES, res.items()):
        m = t > 250; t, f = t[m], f[m]
        ax.plot(t / 1000, f, 'o-', c=c, ms=4, lw=2, label=str(I).replace('.', ','))
    ax.axvspan(1, 1.1, color=VIOLET, alpha=0.18, lw=0)
    ax.set(xlabel='Temps (s)', ylabel='Cycles par seconde (Hz)', xlim=(0.2, 3.5))
    ax.set_title('Heurté par derrière → accélérer')
    ax.legend(title='intensité', ncol=3, loc='upper right', fontsize=11.5, title_fontsize=11.5)
    sauver(fig, 'p2_acceleration')
    return {I: (float(f[(t > 300) & (t < 1000)].mean()), float(f[t > 1000].max())) for I, (t, f) in res.items()}


def fig_arene():
    R, traj, contacts = cache('p2_arene', lambda: arene(duree=30*second, W_AV_CPT=0.12))
    fig, ax = plt.subplots(figsize=(W * 0.62, W * 0.62))
    ax.add_patch(plt.Rectangle((0, 0), 8, 8, fill=False, lw=3, ec=INK))
    ax.plot(traj[:, 1], traj[:, 2], c=MUTED, lw=1.4)
    for cap in ['G', 'D', 'AV', 'AR']:
        pts = [(x, y) for (_, c, _, x, y) in contacts if c == cap]
        if pts: ax.plot(*np.array(pts).T, '*', c=COUL_CAP[cap], ms=14, mec=INK, mew=0.6, ls='none', label=f'contact {cap} ({len(pts)})')
    ax.plot(*traj[0, 1:3], 'o', c=INK, ms=7)
    ax.set_aspect('equal'); ax.set(xlim=(-0.3, 8.3), ylim=(-0.3, 8.3)); ax.grid(False)
    ax.set_title('Boucle fermée : arène, 30 s')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=10.5, handletextpad=0.2, columnspacing=0.8)
    sauver(fig, 'p2_arene')
    return len(contacts)


if __name__ == '__main__':
    quoi = sys.argv[1:] or ['p1', 'p2']
    vals = {}
    if 'p1' in quoi:
        fig_cpg_demarche(); vals['traj'] = fig_trajectoires(); vals['cercles'] = fig_cercles()
        vals['vitesse'] = fig_vitesse(); fig_N()
    if 'p2' in quoi:
        vals['recul08'] = fig_raster_recul(); vals['recul'] = fig_recul_intensites()
        vals['virages'] = fig_virages(); vals['accel'] = fig_acceleration()
    if 'arene' in quoi:
        vals['arene'] = fig_arene()
    for k, v in vals.items(): print(k, v)
