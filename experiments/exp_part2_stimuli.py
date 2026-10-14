'''Partie II : systeme pseudo-autonome, etapes 1 a 9.

Une patte ne bouge que d'avant en arriere, l'hexapode ne peut pas faire de pas de cote.
"Deplacement a gauche" veut donc dire tourner vers la gauche.

Usage : python -m experiments.exp_part2_stimuli [etape ...]   (ex. : 3 5 arene ; tout par defaut)
'''
import sys

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, second

from cpg import activite, arene, decoder, duree_active, freq_pas, pos_a, recrutement, simuler_capteurs
from cpg.config import CAPTEURS, CMD_NOMS, PARAM
from plotting import COUL, afficher_scenario, demarche, sauver


def etape1():
    '''Sans stimulus, le biais tonique fait decharger CMD_AV_G et CMD_AV_D (~100 Hz) : marche avant.'''
    R = simuler_capteurs(duree=2*second)
    sauver(afficher_scenario(R, 'Sans stimulus : marche avant par defaut'), 'p2_1_defaut')
    traj = decoder(R)[0]
    print('Decharges des commandes :', dict(zip(CMD_NOMS, np.bincount(R['CMD'][1], minlength=4))))
    print(f"Position finale : x = {traj[-1, 1]:.2f}, y = {traj[-1, 2]:.2f}, theta = {traj[-1, 3]:.2f} rad  (Partie I : x = 2.00)")


def etape2():
    '''Recrutement et codage de l'intensite : un contact fort recrute plus de neurones, plus vite.'''
    I_vals = np.linspace(0, 1, 21)
    tot, rec, _ = recrutement(I_vals)
    _, _, (t, i) = recrutement([0.2, 0.5, 1.0])
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
    for b, inten in enumerate([0.2, 0.5, 1.0]):
        m = i // PARAM['M'] == b
        ax[0].plot(t[m], i[m] % PARAM['M'] + b * (PARAM['M'] + 2), '|', ms=6, c=f'C{b}', label=f'intensite {inten}')
    ax[0].set(xlabel='Temps (ms)', ylabel='neurone (seuil croissant)', yticks=[], title='Raster : contact de 100 ms')
    ax[0].legend(fontsize=8)
    ax[1].plot(I_vals, rec, 'o-'); ax[1].set(xlabel='Intensite du contact', ylabel='Neurones recrutes', title='Recrutement')
    ax[2].plot(I_vals, tot, 'o-'); ax[2].set(xlabel='Intensite du contact', ylabel='Decharges totales', title='Nombre total de decharges')
    fig.tight_layout()
    sauver(fig, 'p2_2_recrutement')


def etape3():
    '''Obstacle de face -> reculer, puis reprise de la marche avant (WTA CMD_AV / CMD_AR).'''
    stim = [('AV', 1000*ms, 0.8, 100*ms)]
    R = simuler_capteurs(stim, duree=4*second)
    sauver(afficher_scenario(R, 'Obstacle de face (intensite 0.8) : recul puis reprise de la marche avant', stim),
           'p2_3_recul')

    fig, ax = plt.subplots(1, 2, figsize=(14, 3.5))
    for inten in [0.3, 0.5, 0.8, 1.0]:
        R = simuler_capteurs([('AV', 1000*ms, inten, 100*ms)], duree=4*second)
        traj = decoder(R)[0]
        ax[0].plot(traj[:, 0] / 1000, traj[:, 1], label=f'intensite {inten}')
        ax[1].plot(R['t_gs'] / 1000, R['gs_CMD'][1], label=f'intensite {inten}')
    ax[0].axvspan(1, 1.1, color=COUL['AV'], alpha=0.2)
    ax[0].set(xlabel='Temps (s)', ylabel='x (position avant)', title='Aller-retour : position au cours du temps'); ax[0].legend()
    ax[1].axhline(1, ls='--', c='gray', label='seuil')
    ax[1].set(xlabel='Temps (s)', ylabel='gs de CMD_AR_G', title='Courant lent (persistance) sur CMD_AR_G'); ax[1].legend()
    fig.tight_layout()
    sauver(fig, 'p2_3_recul_intensites')

    # Duree du recul (activite de CMD_AR_G) en fonction de l'intensite, commandes seulement
    I_vals = np.round(np.linspace(0.1, 1.0, 10), 2)
    dur_recul = []
    for inten in I_vals:
        R = simuler_capteurs([('AV', 200*ms, inten, 100*ms)], duree=3*second, pattes=False)
        dur_recul.append(duree_active(R, 'CMD', 1, 200))
    fig = plt.figure(figsize=(5, 3.5))
    plt.plot(I_vals, np.array(dur_recul) / 1000, 'o-')
    plt.xlabel('Intensite du contact avant'); plt.ylabel('Duree du recul (s)'); plt.title('Plus l\'impact est fort, plus le recul dure')
    fig.tight_layout()
    sauver(fig, 'p2_3_duree_recul')


def etape4():
    '''Heurte par derriere -> accelerer : SENS_AR augmente le drive effectif du CPG.'''
    stim = [('AR', 1000*ms, 0.8, 100*ms)]
    R = simuler_capteurs(stim, duree=3*second)
    sauver(afficher_scenario(R, 'Heurte par derriere (intensite 0.8) : acceleration du CPG', stim, traj=False),
           'p2_4_acceleration')

    I_vals = np.round(np.linspace(0.1, 1.0, 10), 2)
    pic = []
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.5))
    for inten in I_vals:
        R = simuler_capteurs([('AR', 1000*ms, inten, 100*ms)], duree=3.5*second, pattes=False)
        t, f = freq_pas(R)
        pic.append(f[t > 1000].max())
        if inten in (0.3, 0.6, 1.0): ax[0].plot(t / 1000, f, 'o-', ms=3, label=f'intensite {inten}')
    ax[0].axvspan(1, 1.1, color=COUL['AR'], alpha=0.2)
    ax[0].set(xlabel='Temps (s)', ylabel='Frequence de pas (Hz)', title='Frequence de pas au cours du temps'); ax[0].legend()
    ax[1].plot(I_vals, pic, 'o-'); ax[1].axhline(f[(t > 300) & (t < 1000)].mean(), ls='--', c='gray', label='croisiere')
    ax[1].set(xlabel='Intensite du contact arriere', ylabel='Pic de frequence (Hz)', title='Pic de frequence vs intensite'); ax[1].legend()
    fig.tight_layout()
    sauver(fig, 'p2_4_frequence')


def etape5():
    '''Obstacle sur un flanc -> tourner : courbe douce, virage serre ou pivot selon l'intensite.'''
    res_virage = {inten: simuler_capteurs([('D', 500*ms, inten, 100*ms)], duree=5*second) for inten in [0.3, 0.6, 1.0]}

    fig, ax = plt.subplots(3, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [2.2, 1]})
    gs_ = ax[0, 1].get_gridspec()
    for a in ax[:, 1]: a.remove()
    at = fig.add_subplot(gs_[:, 1])
    for r, (inten, R) in enumerate(res_virage.items()):
        M = R['M']
        t, i = R['S_D']; ax[r, 0].plot(t, i, '|', c=COUL['D'], ms=4)
        t = activite(R, 'CPT', 0); ax[r, 0].plot(t, np.full(len(t), M + 1.5), '|', c='C4', ms=12)
        t = activite(R, 'CMD', 1); ax[r, 0].plot(t, np.full(len(t), M + 3.5), '|', c='k', ms=12)
        t = activite(R, 'CPG', 0); ax[r, 0].plot(t, np.full(len(t), M + 5.5), '|', c='0.5', ms=12)
        ax[r, 0].set(yticks=[M / 2, M + 1.5, M + 3.5, M + 5.5], yticklabels=['SENS_D', 'COMPTEUR_G', 'CMD_AR_G', 'CPG (A)'],
                     xlim=(0, 5000), ylim=(-1, M + 7))
        ax[r, 0].set_title(f'Contact flanc droit, intensite {inten}', fontsize=10, loc='left')
        traj = decoder(R)[0]
        at.plot(traj[:, 1], traj[:, 2], label=f'intensite {inten}  (theta final {np.degrees(traj[-1, 3]):.0f} deg)')
        at.plot(*pos_a(traj, 500), '*', c=COUL['D'], ms=12, mec='k')
    ax[2, 0].set_xlabel('Temps (ms)')
    at.plot(0, 0, 'ko'); at.set_aspect('equal', adjustable='datalim')
    at.set(xlabel='x', ylabel='y', title='Courbe douce, virage serre, pivot'); at.legend(fontsize=8)
    fig.tight_layout()
    sauver(fig, 'p2_5_virages')

    stim = [('D', 500*ms, 1.0, 100*ms)]
    sauver(afficher_scenario(res_virage[1.0], 'Contact fort sur le flanc droit : pivot puis virage a gauche', stim),
           'p2_5_pivot')

    # Flanc gauche -> tourner a droite : meme mecanisme en symetrique
    fig = plt.figure(figsize=(5, 5))
    for cap, inten in [('D', 0.6), ('G', 0.6)]:
        traj = decoder(res_virage[inten] if cap == 'D' else simuler_capteurs([(cap, 500*ms, inten, 100*ms)], duree=5*second))[0]
        plt.plot(traj[:, 1], traj[:, 2], c=COUL[cap], label=f'contact_{cap} = {inten}')
    plt.plot(0, 0, 'ko'); plt.axis('equal'); plt.xlabel('x'); plt.ylabel('y'); plt.legend()
    plt.title('Symetrie gauche / droite')
    sauver(fig, 'p2_5_symetrie')


def etape6():
    '''Selection d'action (stimuli simultanes) : AVANT > FLANCS > ARRIERE.'''
    conflits = {
        'AV 0.6 + D 0.6':  [('AV', 500*ms, 0.6, 100*ms), ('D', 500*ms, 0.6, 100*ms)],
        'G 0.4 + D 0.8':   [('G', 500*ms, 0.4, 100*ms), ('D', 500*ms, 0.8, 100*ms)],
        'D 0.6 + AR 0.8':  [('D', 500*ms, 0.6, 100*ms), ('AR', 500*ms, 0.8, 100*ms)],
    }
    seuls = {'AV 0.6': [('AV', 500*ms, 0.6, 100*ms)], 'D 0.6': [('D', 500*ms, 0.6, 100*ms)],
             'G 0.4': [('G', 500*ms, 0.4, 100*ms)], 'D 0.8': [('D', 500*ms, 0.8, 100*ms)], 'AR 0.8': [('AR', 500*ms, 0.8, 100*ms)]}
    print(f"{'scenario':<16}" + ''.join(f'{"SENS_" + c:>9}' for c in CAPTEURS) + '   pic freq (Hz)')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, (nom, stim) in zip(ax, conflits.items()):
        Rc = simuler_capteurs(stim, duree=3.5*second)
        print(f'{nom:<16}' + ''.join(f"{len(Rc['S_' + c][0]):>9}" for c in CAPTEURS) + f'   {freq_pas(Rc)[1].max():.1f}')
        traj = decoder(Rc)[0]
        a.plot(traj[:, 1], traj[:, 2], 'k', lw=2.5, label=f'simultanes : {nom}')
        for part in nom.split(' + '):
            traj = decoder(simuler_capteurs(seuls[part], duree=3.5*second))[0]
            a.plot(traj[:, 1], traj[:, 2], '--', c=COUL[part.split()[0]], label=f'seul : {part}')
        a.plot(0, 0, 'ko'); a.set_aspect('equal', adjustable='datalim'); a.legend(fontsize=8); a.set(xlabel='x', ylabel='y')
    fig.suptitle("Selection d'action : AVANT > FLANCS > ARRIERE"); fig.tight_layout()
    sauver(fig, 'p2_6_selection')


def etape7_sensibilite():
    '''7.1 Sensibilite (poids SENS_AV -> CMD) : reaction = CMD_AR_G decharge.'''
    I_vals = np.round(np.linspace(0.05, 0.5, 10), 3)
    fig = plt.figure(figsize=(6, 3.5))
    for w in [0.03, 0.05, 0.08]:
        rep = []
        for inten in I_vals:
            R = simuler_capteurs([('AV', 200*ms, inten, 100*ms)], duree=1.5*second, pattes=False, W_AV_AR=w, W_AV_AV=0.8 * w)
            rep.append(len(activite(R, 'CMD', 1)) > 0)
        plt.step(I_vals, rep, where='mid', lw=2, label=f'W_AV_AR = {w}')
        print(f'W_AV_AR = {w} : intensite minimale pour reculer = {I_vals[np.argmax(rep)] if any(rep) else "> 0.5"}')
    plt.yticks([0, 1], ['non', 'oui']); plt.xlabel('Intensite du contact avant'); plt.ylabel('Recul ?')
    plt.title('Sensibilite'); plt.legend(); fig.tight_layout()
    sauver(fig, 'p2_7_1_sensibilite')


def etape7_persistance():
    '''7.2 Duree de la reaction (tau_persist).'''
    I_vals = [0.3, 0.5, 0.8, 1.0]
    fig = plt.figure(figsize=(6, 3.5))
    for tp in [400, 800, 1600]:
        dur = []
        for inten in I_vals:
            R = simuler_capteurs([('AV', 200*ms, inten, 100*ms)], duree=5*second, pattes=False, TAU_PERSIST=tp*ms)
            dur.append(duree_active(R, 'CMD', 1, 200) / 1000)
        plt.plot(I_vals, dur, 'o-', label=f'tau_persist = {tp} ms')
    plt.xlabel('Intensite du contact avant'); plt.ylabel('Duree du recul (s)'); plt.title('Duree de la reaction')
    plt.legend(); fig.tight_layout()
    sauver(fig, 'p2_7_2_persistance')


def etape7_latence():
    '''7.3 Temps de reaction : distance parcourue entre le contact et le debut du recul.'''
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    delais, dist = [0, 100, 200, 400], []
    for dl in delais:
        R = simuler_capteurs([('AV', 1000*ms, 0.8, 100*ms)], duree=3*second, DELAI_SENS=dl*ms)
        traj = decoder(R)[0]
        x0 = pos_a(traj, 1000)[0]
        dist.append(traj[:, 1].max() - x0)
        ax[0].plot(traj[:, 0] / 1000, traj[:, 1], label=f'delai = {dl} ms')
    ax[0].axvline(1, c=COUL['AV'], ls='--'); ax[0].set(xlabel='Temps (s)', ylabel='x', title='Position (contact a t = 1 s)'); ax[0].legend()
    ax[1].plot(delais, dist, 'o-'); ax[1].set(xlabel='Delai capteur -> commande (ms)', ylabel='Distance avant de reculer',
                                              title='Latence : distance parcourue avant de reagir')
    fig.tight_layout()
    sauver(fig, 'p2_7_3_latence')


def etape7_pivot():
    '''7.4 Seuil de passage virage -> rotation (poids faible SENS_D -> CMD_AR_G).'''
    I_vals = np.round(np.linspace(0.3, 1.0, 8), 2)
    fig = plt.figure(figsize=(6, 3.5))
    for w in [0.02, 0.025, 0.035]:
        piv = []
        for inten in I_vals:
            R = simuler_capteurs([('D', 200*ms, inten, 100*ms)], duree=1.5*second, pattes=False, W_LAT_AR=w)
            piv.append(len(activite(R, 'CMD', 1)) > 0)
        plt.step(I_vals, piv, where='mid', lw=2, label=f'W_LAT_AR = {w}')
        print(f'W_LAT_AR = {w} : bascule virage -> pivot a l\'intensite {I_vals[np.argmax(piv)] if any(piv) else "> 1"}')
    plt.yticks([0, 1], ['virage', 'pivot']); plt.xlabel('Intensite du contact lateral')
    plt.title('Seuil de passage virage -> rotation'); plt.legend(); fig.tight_layout()
    sauver(fig, 'p2_7_4_pivot')


def etape7_seuils():
    '''7.5 Seuils des neurones sensoriels -> plage d'intensites discriminees.'''
    I_vals = np.linspace(0, 1.5, 31)
    fig = plt.figure(figsize=(6, 3.5))
    for smin, smax in [(0.9, 1.1), (0.2, 1.8), (0.2, 3.0)]:
        plt.plot(I_vals, recrutement(I_vals, smin, smax)[1], 'o-', ms=3, label=f'seuils {smin} a {smax}')
    plt.xlabel('Intensite du contact'); plt.ylabel('Neurones recrutes (sur 10)'); plt.title('Plage d\'intensites discriminees')
    plt.legend(); fig.tight_layout()
    sauver(fig, 'p2_7_5_seuils')


def etape7():
    etape7_sensibilite()
    etape7_persistance()
    etape7_latence()
    etape7_pivot()
    etape7_seuils()


def etape8():
    '''Generalisation a N paires : capteurs et commandes sont par cote, on change seulement N_PAIRES.'''
    scenario = [('D', 500*ms, 0.6, 100*ms), ('AV', 3000*ms, 0.8, 100*ms)]
    res_N = {N: simuler_capteurs(scenario, N_PAIRES=N, delta=dl, duree=5*second)
             for N, dl in [(3, 0*ms), (20, 5*ms), (200, 0.5*ms)]}
    fig = plt.figure(figsize=(5, 5))
    for N, R in res_N.items():
        traj = decoder(R)[0]
        plt.plot(traj[:, 1], traj[:, 2], label=f'N_PAIRES = {N}')
    plt.plot(0, 0, 'ko'); plt.axis('equal'); plt.xlabel('x'); plt.ylabel('y'); plt.legend()
    plt.title('Meme scenario, N = 3, 20, 200')
    sauver(fig, 'p2_8_trajectoires_N')

    for N in [20, 200]:
        sauver(afficher_scenario(res_N[N], f'N_PAIRES = {N} : raster complet (pattes gauches, echantillon)', scenario),
               f'p2_8_raster_N{N}')
        contact = decoder(res_N[N])[1]
        n = res_N[N]['n']
        fig = plt.figure(figsize=(12, 3))
        demarche(plt.gca(), contact, range(0, n, 2 * max(1, n // 40)))
        plt.gca().invert_yaxis(); plt.yticks([])
        plt.xlabel('Temps (ms)'); plt.ylabel('Pattes gauches (avant -> arriere)')
        plt.title(f'Diagramme de demarche, N_PAIRES = {N} (barre = patte au sol)')
        sauver(fig, f'p2_8_demarche_N{N}')


def etape9_habituation():
    '''Bonus : depression a court terme SENS_AR -> CPG, l'acceleration diminue a chaque poussee.'''
    pousses = [('AR', t0 * ms, 0.8, 100*ms) for t0 in [500, 2000, 3500, 5000, 6500]]
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.5), gridspec_kw={'width_ratios': [2.5, 1]})
    for hab, lab in [(False, 'sans habituation'), (True, 'avec habituation')]:
        R = simuler_capteurs(pousses, duree=8*second, pattes=False, HABITUATION=hab)
        t, f = freq_pas(R)
        ax[0].plot(t / 1000, f, 'o-', ms=3, label=lab)
        pics = [f[(t > p[1] / ms) & (t < p[1] / ms + 1500)].max() for p in pousses]
        ax[1].plot(range(1, len(pousses) + 1), pics, 'o-', label=lab)
    for p in pousses: ax[0].axvspan(p[1] / second, p[1] / second + 0.1, color=COUL['AR'], alpha=0.3)
    ax[0].set(xlabel='Temps (s)', ylabel='Frequence de pas (Hz)', title='Poussees repetees par derriere'); ax[0].legend()
    ax[1].set(xlabel='Numero de la poussee', ylabel='Pic de frequence (Hz)', title='Habituation'); ax[1].legend()
    fig.tight_layout()
    sauver(fig, 'p2_9_habituation')


def etape9_arene():
    '''Bonus : boucle fermee dans une arene, l'hexapode explore son environnement tout seul.'''
    R, traj, contacts = arene(duree=30*second, W_AV_CPT=0.12)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.4]})
    ax[0].add_patch(plt.Rectangle((0, 0), 8, 8, fill=False, lw=3))
    ax[0].plot(traj[:, 1], traj[:, 2], c='0.75', lw=1)
    sc = ax[0].scatter(traj[:, 1], traj[:, 2], c=traj[:, 0] / 1000, s=5, cmap='viridis')
    for cap in CAPTEURS:
        pts = [(x, y) for (_, c, _, x, y) in contacts if c == cap]
        if pts: ax[0].plot(*np.array(pts).T, '*', c=COUL[cap], ms=13, mec='k', label=f'contact_{cap} ({len(pts)})')
    ax[0].plot(*traj[0, 1:3], 'ko'); ax[0].set_aspect('equal'); ax[0].legend(fontsize=8, loc='upper right')
    ax[0].set(xlim=(-0.3, 8.3), ylim=(-0.3, 8.3), title='Exploration autonome (30 s)')
    fig.colorbar(sc, ax=ax[0], label='temps (s)', shrink=0.7)
    for c, cap in enumerate(CAPTEURS):
        t, i = R['S_' + cap]; ax[1].plot(t / 1000, i + c * R['M'], '|', c=COUL[cap], ms=3)
    t, i = R['CMD']; ax[1].plot(t / 1000, 4 * R['M'] + 2 + 2 * i, '|', c='k', ms=5)
    t, i = R['CPT']; ax[1].plot(t / 1000, 4 * R['M'] + 11 + 2 * i, '|', c='C4', ms=8)
    ax[1].set(yticks=[R['M'] / 2 + c * R['M'] for c in range(4)] + [4 * R['M'] + 2 + 2 * j for j in range(4)] + [4 * R['M'] + 11, 4 * R['M'] + 13],
              yticklabels=['SENS_' + c for c in CAPTEURS] + CMD_NOMS + ['CPT_G', 'CPT_D'], xlabel='Temps (s)', title='Capteurs -> commandes')
    fig.tight_layout()
    sauver(fig, 'p2_9_arene')
    print(f'{len(contacts)} contacts :', ', '.join(f'{c}@{t / 1000:.1f}s({i:.2f})' for t, c, i, _, _ in contacts))


ETAPES = {'1': etape1, '2': etape2, '3': etape3, '4': etape4, '5': etape5, '6': etape6, '7': etape7,
          '8': etape8, 'habituation': etape9_habituation, 'arene': etape9_arene}


def main(etapes=None):
    for e in etapes or ETAPES:
        print(f'--- Etape {e}')
        ETAPES[e]()


if __name__ == '__main__':
    main(sys.argv[1:])
    plt.show()
