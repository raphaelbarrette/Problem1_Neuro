'''Figures de l'annexe de l'affiche, meme style que poster_figs (PNG et PDF dans affiche/figures/).

Usage (depuis Code/) : python annexe_figs.py [a] [b] [pivot]   (tout par defaut)
'''
import sys

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, second

from poster_figs import (AQUA, BLEU, COUL_CAP, INK, MUTED, ORANGE, SERIES, VIOLET, W, cache, sauver)
from cpg import activite, decoder, duree_active, freq_pas, pos_a, recrutement, simuler_capteurs


def virg(x): return str(x).replace('.', ',')


def fig_recrutement():
    I_vals = np.linspace(0, 1, 21)
    tot, rec, _ = recrutement(I_vals)
    _, _, (t, i) = recrutement([0.2, 0.5, 1.0])
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={'wspace': 0.45, 'width_ratios': [1.2, 1]})
    for b, (inten, c) in enumerate(zip([0.2, 0.5, 1.0], SERIES)):
        m = i // 10 == b
        ax[0].vlines(t[m], i[m] % 10 + b * 12 - 0.4, i[m] % 10 + b * 12 + 0.4, color=c, lw=1.4)
    ax[0].set(xlabel='Temps (ms)', yticks=[4.5, 16.5, 28.5], yticklabels=['0,2', '0,5', '1,0'], ylabel='Intensité', xlim=(0, 100)); ax[0].grid(False)
    ax[0].set_title('Contact de 100 ms')
    ax[1].plot(I_vals, rec, 'o-', c=AQUA, ms=6)
    ax[1].set(xlabel='Intensité', ylabel='Neurones recrutés'); ax[1].set_title('Recrutement (sur 10)')
    sauver(fig, 'a_recrutement')


def fig_selection():
    conflits = {'AV 0,6 + D 0,6': [('AV', 0.6), ('D', 0.6)], 'G 0,4 + D 0,8': [('G', 0.4), ('D', 0.8)],
                'D 0,6 + AR 0,8': [('D', 0.6), ('AR', 0.8)]}
    st = lambda lst: [(c, 500*ms, i, 100*ms) for c, i in lst]
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.6), gridspec_kw={'wspace': 0.35})
    for a, (nom, lst) in zip(ax, conflits.items()):
        Rc = cache('sel_' + nom, lambda lst=lst: simuler_capteurs(st(lst), duree=3.5*second))
        tr = decoder(Rc)[0]
        for cap, inten in lst:
            Rs = cache(f'seul_{cap}_{inten}', lambda cap=cap, inten=inten: simuler_capteurs(st([(cap, inten)]), duree=3.5*second))
            ts = decoder(Rs)[0]
            a.plot(ts[:, 1], ts[:, 2], ls=(0, (3, 2)), c=COUL_CAP[cap], lw=1.8)
        a.plot(tr[:, 1], tr[:, 2], c=INK, lw=2.4)
        a.plot(0, 0, 'o', c=INK, ms=5)
        a.set_aspect('equal', adjustable='datalim'); a.set_title(nom, fontsize=13); a.tick_params(labelsize=10.5)
    ax[0].plot([], [], c=INK, lw=2.4, label='simultanés'); ax[0].plot([], [], c=MUTED, ls=(0, (3, 2)), label='chacun seul')
    fig.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
    sauver(fig, 'a_selection')


def fig_sensibilite():
    I_vals = np.round(np.linspace(0.05, 0.5, 10), 3)
    def run():
        out = {}
        for w in [0.03, 0.05, 0.08]:
            out[w] = [len(activite(simuler_capteurs([('AV', 200*ms, I, 100*ms)], duree=1.5*second, pattes=False,
                                                    W_AV_AR=w, W_AV_AV=0.8 * w), 'CMD', 1)) > 0 for I in I_vals]
        return out
    res = cache('a_sensib', run)
    fig, ax = plt.subplots(figsize=(W / 2, 2.3))
    for k, (c, (w, rep)) in enumerate(zip(SERIES, res.items())):
        ax.step(I_vals, np.array(rep) + 0.04 * k, where='mid', c=c, lw=2.2, label=virg(w))
    ax.set(yticks=[0, 1], yticklabels=['non', 'oui'], xlabel='Intensité (face)'); ax.set_title('W_AV_AR : sensibilité', fontsize=13)
    ax.legend(fontsize=10.5, loc='center right'); ax.tick_params(labelsize=11)
    sauver(fig, 'a_sensibilite')
    return {w: (I_vals[np.argmax(r)] if any(r) else None) for w, r in res.items()}


def fig_persistance():
    I_vals = [0.3, 0.5, 0.8, 1.0]
    def run():
        return {tp: [duree_active(simuler_capteurs([('AV', 200*ms, I, 100*ms)], duree=5*second, pattes=False,
                                                   TAU_PERSIST=tp*ms), 'CMD', 1, 200) / 1000 for I in I_vals] for tp in [400, 800, 1600]}
    res = cache('a_persist', run)
    fig, ax = plt.subplots(figsize=(W / 2, 2.3))
    for c, (tp, d) in zip(SERIES, res.items()):
        ax.plot(I_vals, d, 'o-', c=c, ms=5, label=f'{tp} ms')
    ax.set(xlabel='Intensité (face)', ylabel='Recul (s)'); ax.set_title('τ_persist : durée', fontsize=13)
    ax.legend(fontsize=10.5); ax.tick_params(labelsize=11)
    sauver(fig, 'a_persistance')
    return res


def fig_latence():
    delais = [0, 100, 200, 400]
    def run():
        out = []
        for dl in delais:
            R = simuler_capteurs([('AV', 1000*ms, 0.8, 100*ms)], duree=3*second, DELAI_SENS=dl*ms)
            tr = decoder(R)[0]
            out.append(tr[:, 1].max() - pos_a(tr, 1000)[0])
        return out
    dist = cache('a_latence', run)
    fig, ax = plt.subplots(figsize=(W / 2, 2.3))
    ax.plot(delais, dist, 'o-', c=AQUA, ms=6)
    ax.set(xlabel='Délai capteur (ms)', ylabel='Distance'); ax.set_title('Délai : temps de réaction', fontsize=13)
    ax.tick_params(labelsize=11)
    sauver(fig, 'a_latence')
    return dict(zip(delais, dist))


def fig_pivot():
    I_vals = np.round(np.linspace(0.3, 1.0, 8), 2)
    def run():
        return {w: [len(activite(simuler_capteurs([('D', 200*ms, I, 100*ms)], duree=1.5*second, pattes=False, W_LAT_AR=w), 'CMD', 1)) > 0
                    for I in I_vals] for w in [0.02, 0.025, 0.035]}
    res = cache('a_pivot', run)
    fig, ax = plt.subplots(figsize=(W / 2, 2.3))
    for k, (c, (w, piv)) in enumerate(zip(SERIES, res.items())):
        ax.step(I_vals, np.array(piv) + 0.04 * k, where='mid', c=c, lw=2.2, label=virg(w))
    ax.set(yticks=[0, 1], yticklabels=['virage', 'pivot'], xlabel='Intensité (flanc)'); ax.set_title('W_LAT_AR : pivot', fontsize=13)
    ax.legend(fontsize=10.5, loc='center right'); ax.tick_params(labelsize=11)
    sauver(fig, 'a_pivot')
    return {w: (I_vals[np.argmax(r)] if any(r) else None) for w, r in res.items()}


def fig_seuils():
    I_vals = np.linspace(0, 1.5, 31)
    fig, ax = plt.subplots(figsize=(W / 2, 2.3))
    for c, (a, b) in zip(SERIES, [(0.9, 1.1), (0.2, 1.8), (0.2, 3.0)]):
        ax.plot(I_vals, recrutement(I_vals, a, b)[1], 'o-', c=c, ms=3, lw=1.8, label=f'{virg(a)} à {virg(b)}')
    ax.set(xlabel='Intensité', ylabel='Recrutés'); ax.set_title('Seuils des capteurs', fontsize=13)
    ax.legend(fontsize=10.5); ax.tick_params(labelsize=11)
    sauver(fig, 'a_seuils')


def fig_habituation():
    pousses = [('AR', t0 * ms, 0.8, 100*ms) for t0 in [500, 2000, 3500, 5000, 6500]]
    def run():
        return {hab: freq_pas(simuler_capteurs(pousses, duree=8*second, pattes=False, HABITUATION=hab)) for hab in [False, True]}
    res = cache('a_habit', run)
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for c, (hab, (t, f)) in zip([MUTED, VIOLET], res.items()):
        m = t > 250; t, f = t[m], f[m]
        ax.plot(t / 1000, f, 'o-', c=c, ms=3, lw=1.8, label='avec habituation' if hab else 'sans habituation')
    for p in pousses: ax.axvspan(p[1] / second, p[1] / second + 0.1, color=VIOLET, alpha=0.2, lw=0)
    ax.set(xlabel='Temps (s)', ylabel='Cycles / s (Hz)'); ax.set_title('Poussées répétées par derrière')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=2)
    sauver(fig, 'a_habituation')


def fig_N2():
    scen = [('D', 500*ms, 0.6, 100*ms), ('AV', 3000*ms, 0.8, 100*ms)]
    fig, ax = plt.subplots(figsize=(W * 0.55, W * 0.55))
    for c, (N, dl) in zip(SERIES, [(3, 0), (20, 5), (200, 0.5)]):
        R = cache(f'a_N{N}', lambda N=N, dl=dl: simuler_capteurs(scen, N_PAIRES=N, delta=dl*ms, duree=5*second))
        tr = decoder(R)[0]
        ax.plot(tr[:, 1], tr[:, 2], c=c, lw=2.4, label=f'N = {N}')
    ax.plot(0, 0, 'o', c=INK, ms=6)
    ax.set_aspect('equal', adjustable='datalim'); ax.set(xlabel='x', ylabel='y')
    ax.set_title('Partie II : même scénario, N = 3, 20, 200', fontsize=13)
    ax.legend(fontsize=11)
    sauver(fig, 'a_N_partie2')



def fig_raster_pivot():
    R = cache('p2_virD_1.0', lambda: simuler_capteurs([('D', 500*ms, 1.0, 100*ms)], duree=5*second))
    tr = decoder(R)[0]
    M = R['M']
    fig, ax = plt.subplots(5, 1, figsize=(W, 4.6), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.2, 1, 1.3, 1.5], 'hspace': 0.2})
    for a in ax: a.axvspan(0.5, 0.6, color=ORANGE, alpha=0.18, lw=0); a.grid(False)
    t, i = R['S_D']; ax[0].vlines(t / 1000, i - 0.4, i + 0.4, color=ORANGE, lw=1.4)
    ax[0].set(yticks=[M / 2 - 0.5], yticklabels=['SENS_D'], ylim=(-1, M))
    t, i = R['CPT']; ax[1].vlines(t[i == 0] / 1000, -0.38, 0.38, color=VIOLET, lw=2)
    t, i = R['INH']; ax[1].vlines(t[i == 0] / 1000, 0.62, 1.38, color=INK, lw=2)
    ax[1].set(yticks=[0, 1], yticklabels=['COMPTEUR_G', 'INHIB_G'], ylim=(-0.6, 1.6))
    t, i = R['CMD']
    for k, (lab, c) in enumerate([('CMD_AV_G', BLEU), ('CMD_AR_G', ORANGE)]):
        ax[2].vlines(t[i == k] / 1000, k - 0.38, k + 0.38, color=c, lw=0.8)
    ax[2].set(yticks=[0, 1], yticklabels=['CMD_AV_G', 'CMD_AR_G'], ylim=(-0.6, 1.6))
    for nom, k, c in [('AVANT', 0, BLEU), ('ARRIERE', 1, ORANGE)]:
        t, i = R[nom]; ax[3].vlines(t[i == 0] / 1000, k - 0.38, k + 0.38, color=c, lw=2)
    ax[3].set(yticks=[0, 1], yticklabels=['AVANT G1', 'ARRIÈRE G1'], ylim=(-0.6, 1.6))
    ax[4].step(tr[:, 0] / 1000, np.degrees(tr[:, 3]), where='post', c=INK)
    ax[4].set(ylabel='θ (°)', xlabel='Temps (s)', xlim=(0, 5)); ax[4].grid(True)
    ax[0].set_title('Contact fort à droite (1,0) : pivot puis virage à gauche')
    sauver(fig, 'a_raster_pivot')


GROUPES = {'a': [fig_recrutement, fig_seuils, fig_sensibilite, fig_pivot, fig_persistance],
           'b': [fig_selection, fig_latence, fig_habituation, fig_N2], 'pivot': [fig_raster_pivot]}

if __name__ == '__main__':
    for g in sys.argv[1:] or GROUPES:
        for f in GROUPES[g]:
            r = f()
            if r is not None: print(f.__name__, r)

