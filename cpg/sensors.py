'''Partie II : systeme pseudo-autonome.

Les commandes fixees a la main de la Partie I (mode_G, mode_D, virage_G, virage_D, I_drive)
sont remplacees par des neurones de commande pilotes par des neurones sensoriels :

    Stimuli (contact + intensite) : flanc G . flanc D . avant . arriere
            |
    Neurones sensoriels SENS_G/D/AV/AR (M neurones, seuils differents)  -- selection d'action
            |  synapses lentes (tau_persist)
    Neurones de commande CMD_AV_G . CMD_AR_G . CMD_AV_D . CMD_AR_D . COMPTEUR_G/D . drive du CPG
            |
    Reseau de la Partie I : CPG -> interneurones de direction -> motoneurones des pattes

Un stimulus est un tuple (capteur, t0, intensite, duree).

Modele : meme LIF qu'en Partie I, avec deux ajouts (voir config.eqs) :
- gs : courant synaptique lent (signe) qui decroit avec tau_s. Une synapse lente fait gs += w
  (excitatrice) ou gs -= w (inhibitrice). C'est le mecanisme de persistance : un contact de
  100 ms produit une reaction qui dure ~tau_persist.
- seuil : seuil de decharge propre a chaque neurone (1 partout, sauf chez les capteurs ou il
  va de SEUIL_MIN a SEUIL_MAX -> codage de l'intensite par recrutement et par frequence).

Biais de mode (CMD -> interneurones). En Partie I, le biais B_ET = 0.6 des IN etait fixe a la
main. Ici chaque decharge du neurone de commande ajoute du courant lent sur les IN de son cote,
par une synapse saturante gs += W_MODE (G_MODE - gs), qui decroit avec TAU_MODE = 50 ms. En
regime periodique, gs oscille entre ~0.45 et ~0.85 pour une commande de 50 a 300 Hz (~0.62 a
0.76 a 100 Hz, la frequence de CMD_AV au repos). Il faut > 0.4 pour que le ET (CPG, W_ET = 0.6,
+ biais) franchisse le seuil, et < 1 pour que l'IN ne decharge pas seul.

Compteur. En Partie I, c'etait un integrateur parfait (fuite 100 s) qui ne recevait le CPG que
si le virage etait active. Ici il recoit toujours les salves de A (W_CPT = 0.35) mais fuit
(TAU_CPT = 150 ms) : avec un cycle de ~195 ms a I_drive = 2, son pic plafonne a ~0.48, donc le
CPG seul ne le fait jamais decharger. Le courant sensoriel lent s'ajoute et le compteur atteint
son seuil plus souvent quand le contact est fort.

Persistance. Chaque decharge sensorielle ajoute w au gs du neurone de commande, qui decroit avec
TAU_PERSIST = 800 ms (le plan suggerait 500 ms ; 800 ms allonge les reactions, voir l'etape
7.2). Le pic de gs est proportionnel au nombre de decharges, et la reaction dure tant que gs
reste au-dessus du seuil : t ~ TAU_PERSIST ln(g_pic / g_seuil). Plus le contact est fort, plus
la reaction monte haut et dure longtemps.
'''
import numpy as np
from brian2 import (Network, SpikeMonitor, StateMonitor, TimedArray, ms, network_operation,
                    second, start_scope)

from .config import CAPTEURS, NOMS, PARAM, W_AB, W_MOT
from .body_model import Decodeur
from .network import construire_pattes, cpg_pop, indices, pop, syn


def simuler_capteurs(stimuli=(), N_PAIRES=3, I_drive=2.0, duree=2*second, delta=0*ms, pattes=True,
                     extra=None, **p):
    '''stimuli : liste de (capteur, t0, intensite, duree), ou None si le courant sensoriel est
    impose de l'exterieur (boucle fermee). pattes=False construit seulement capteurs + commandes
    + CPG + compteurs (plus rapide). p : remplace des valeurs de PARAM.

    4 neurones de commande, indice 2s + ar : 0 = CMD_AV_G, 1 = CMD_AR_G, 2 = CMD_AV_D,
    3 = CMD_AR_D. CMD_AV a un biais tonique (I_CMD, ~100 Hz) -> marche avant par defaut.
    Capteurs et commandes sont par cote, pas par patte : on change toujours seulement N_PAIRES.'''
    P_ = PARAM | p
    start_scope()
    M = P_['M']
    tous = lambda cibles: tuple(x.ravel() for x in np.meshgrid(range(M), cibles))
    n, k, s, *_ = indices(N_PAIRES, delta)

    # ------------------------------------------------ Partie I : CPG, compteurs, INHIB
    cpg = cpg_pop(I_drive, tau_s=P_['TAU_PERSIST'])
    cpt = pop(2, tau=P_['TAU_CPT'], tau_s=P_['TAU_PERSIST'])   # COMPTEUR_G, COMPTEUR_D (fuyants)
    inh = pop(2)                              # INHIB_G, INHIB_D
    S = [syn(cpg, cpg, W_AB, [0, 1], [1, 0], inhib=True),
         syn(cpg, cpt, P_['W_CPT'], [0, 0], [0, 1]),          # chaque salve de A -> compteurs
         syn(cpt, inh, W_MOT, [0, 1], [0, 1])]

    # ------------------------------------------------ Partie II : capteurs et commandes
    if stimuli is not None:                   # stimuli scriptes -> courant k x intensite
        stim = np.zeros((int(duree / ms) + 1, 4))
        for cap, t0, inten, dur in stimuli:
            stim[int(t0 / ms):int((t0 + dur) / ms), CAPTEURS.index(cap)] = inten
        stim_ta = TimedArray(stim, dt=1*ms)
    SENS = {}
    for c, cap in enumerate(CAPTEURS):
        SENS[cap] = pop(M, tau=10*ms, seuil=np.linspace(P_['SEUIL_MIN'], P_['SEUIL_MAX'], M))
        if stimuli is not None:
            SENS[cap].run_regularly(f"I = {P_['K_SENS']} * stim_ta(t, {c})", dt=1*ms)
    cmd = pop(4, tau=10*ms, tau_s=P_['TAU_PERSIST'], Ib=[P_['I_CMD'], 0, P_['I_CMD'], 0])

    ds = P_['DELAI_SENS']
    S += [syn(cmd, cmd, P_['W_WTA'], [0, 1, 2, 3], [1, 0, 3, 2], inhib=True),       # AV <-> AR
          syn(cmd, cpt, P_['W_AR_CPT'], [1, 3], [0, 1], inhib=True),                  # pivot > virage
          # recul : SENS_AV excite CMD_AR_G/D et inhibe CMD_AV_G/D (lentes)
          syn(SENS['AV'], cmd, P_['W_AV_AR'], *tous([1, 3]), lente=True, delay=ds),
          syn(SENS['AV'], cmd, P_['W_AV_AV'], *tous([0, 2]), lente=True, inhib=True, delay=ds),
          # virages : flanc D -> COMPTEUR_G (+ CMD_AR_G, poids faible) ; flanc G symetrique
          syn(SENS['D'], cpt, P_['W_LAT_CPT'], *tous([0]), lente=True, delay=ds),
          syn(SENS['G'], cpt, P_['W_LAT_CPT'], *tous([1]), lente=True, delay=ds),
          syn(SENS['D'], cmd, P_['W_LAT_AR'], *tous([1]), lente=True, delay=ds),
          syn(SENS['G'], cmd, P_['W_LAT_AR'], *tous([3]), lente=True, delay=ds)]
    if P_['W_AV_CPT'] > 0:
        S.append(syn(SENS['AV'], cpt, P_['W_AV_CPT'], *tous([0]), lente=True, delay=ds))
    # acceleration : SENS_AR -> A et B du CPG (lente), avec habituation en option
    if P_['HABITUATION']:
        S.append(syn(SENS['AR'], cpg, None, *tous([0, 1]), delay=ds,
                     model=f"dx/dt = (1 - x) / ({P_['TAU_REC'] / ms} * ms) : 1 (event-driven)",
                     on_pre=f"gs_post += {P_['W_AR_CPG']} * x; x -= {P_['U_STD']} * x"))
        S[-1].x = 1
    else:
        S.append(syn(SENS['AR'], cpg, P_['W_AR_CPG'], *tous([0, 1]), lente=True, delay=ds))
    # selection d'action : AVANT > FLANCS > ARRIERE (inhibition rapide entre capteurs)
    S += [syn(SENS['AV'], SENS[c], P_['W_S_AV'], inhib=True) for c in ['G', 'D', 'AR']]
    S += [syn(SENS[c], SENS['AR'], P_['W_S_LAT'], inhib=True) for c in ['G', 'D']]
    S += [syn(SENS['G'], SENS['D'], P_['W_S_GD'], inhib=True), syn(SENS['D'], SENS['G'], P_['W_S_GD'], inhib=True)]

    groupes = [cpg, cpt, inh, cmd, *SENS.values()]
    Mon = {'CPG': SpikeMonitor(cpg), 'CPT': SpikeMonitor(cpt), 'INH': SpikeMonitor(inh), 'CMD': SpikeMonitor(cmd)}
    Mon |= {'S_' + c: SpikeMonitor(SENS[c]) for c in CAPTEURS}
    gsM = [StateMonitor(g, 'gs', record=True, dt=1*ms) for g in (cmd, cpt, cpg)]

    # ------------------------------------------------ Partie I : pattes (inchange)
    if pattes:
        P, S_p = construire_pattes(cpg, inh, N_PAIRES, delta)
        S += S_p
        for nom in NOMS[4:]: P[nom].tau_s = P_['TAU_MODE']     # decroissance du biais de mode
        for nom in ['IN_sw_av', 'IN_st_ar']: P[nom].gs = 0.7   # depart en regime de marche avant
        # NOUVEAU : le biais de mode vient des neurones de commande (remplace mode_G / mode_D)
        for nom, ar in [('IN_sw_av', 0), ('IN_st_ar', 0), ('IN_sw_ar', 1), ('IN_st_av', 1)]:
            S.append(syn(cmd, P[nom], None, 2 * s + ar, range(n), delay=k * delta,
                         on_pre=f"gs_post += {P_['W_MODE']} * ({P_['G_MODE']} - gs_post)"))
        groupes += list(P.values())
        Mon |= {nom: SpikeMonitor(P[nom]) for nom in NOMS}

    objets = extra(Mon | {'n': n, 'SENS': SENS}) if extra else []
    Network(*groupes, *S, *Mon.values(), *gsM, *objets).run(duree)
    R = {nom: (m.t / ms, np.array(m.i)) for nom, m in Mon.items()}
    R |= {'t_gs': gsM[0].t / ms, 'gs_CMD': gsM[0].gs.copy(), 'gs_CPT': gsM[1].gs.copy(), 'gs_CPG': gsM[2].gs.copy()}
    return R | {'n': n, 'duree': duree / ms, 'M': M, 'pattes': pattes}


def recrutement(intensites, seuil_min=PARAM['SEUIL_MIN'], seuil_max=PARAM['SEUIL_MAX'], M=PARAM['M'],
                K=PARAM['K_SENS'], duree=100*ms):
    '''Une population sensorielle par intensite (contact de 100 ms). Renvoie decharges et recrutes.'''
    start_scope()
    nI = len(intensites)
    G = pop(nI * M, tau=10*ms, seuil=np.tile(np.linspace(seuil_min, seuil_max, M), nI))
    G.I = K * np.repeat(intensites, M)
    S = SpikeMonitor(G)
    Network(G, S).run(duree)
    cnt = np.array(S.count).reshape(nI, M)
    return cnt.sum(1), (cnt > 0).sum(1), (S.t / ms, np.array(S.i))


def arene(duree=20*second, taille=8.0, rayon=0.5, depart=(4.0, 4.0, 0.4), N_PAIRES=3, **p):
    '''Boucle fermee : un network_operation lit les decharges motrices toutes les 10 ms, met a
    jour la position (decodeur en ligne) et injecte un contact quand le corps (cercle de rayon
    `rayon`) touche un mur. Personne ne fixe les stimuli : l'hexapode explore tout seul.
    - L'angle phi du mur par rapport au cap choisit le capteur : |phi| < 45 deg -> avant ;
      45 a 135 deg -> flanc gauche ; -135 a -45 deg -> flanc droit ; sinon arriere.
    - Intensite = composante normale de l'impact, max(0.3, |cos phi|) : un mur de face donne un
      contact fort, un mur frole un contact faible.
    - Le courant K_SENS x intensite est injecte 100 ms, puis le capteur est refractaire 400 ms.
    - Avec W_AV_CPT > 0 (0.12 dans les experiences), SENS_AV excite aussi COMPTEUR_G :
      l'hexapode recule en tournant au lieu de repartir droit vers le meme mur (dans l'essai
      de 30 s, il lui faut quand meme deux reculs pour quitter le premier mur).
    Renvoie R, la trajectoire et la liste des contacts (t, capteur, intensite, x, y).'''
    P_ = PARAM | p
    journal = {'contacts': []}

    def extra(G):
        D = Decodeur(G['n'], x=depart[0], y=depart[1], th=depart[2],
                     limites=(rayon, taille - rayon, rayon, taille - rayon))
        lus = dict.fromkeys(['LEVER', 'POSER', 'AVANT', 'ARRIERE'], 0)
        fin, pret = dict.fromkeys(CAPTEURS, -1.0), dict.fromkeys(CAPTEURS, 0.0)
        murs = [(0.0, lambda x, y: x >= taille - rayon - 1e-9), (np.pi, lambda x, y: x <= rayon + 1e-9),
                (np.pi / 2, lambda x, y: y >= taille - rayon - 1e-9), (-np.pi / 2, lambda x, y: y <= rayon + 1e-9)]
        journal['D'] = D

        @network_operation(dt=10*ms)
        def monde(t):
            T = float(t / ms)
            ev = []
            for c in lus:                                      # nouvelles decharges motrices
                m, k0 = G[c], lus[c]
                k1 = m.num_spikes
                if k1 > k0:
                    ev += list(zip(np.asarray(m.t[k0:k1] / ms), np.asarray(m.i[k0:k1]), [c] * (k1 - k0)))
                    lus[c] = k1
            for e in sorted(ev): D.evenement(*e)
            for psi, touche in murs:                           # contact avec un mur
                if not touche(D.x, D.y): continue
                phi = (psi - D.th + np.pi) % (2 * np.pi) - np.pi
                cap = 'AV' if abs(phi) < np.pi / 4 else 'AR' if abs(phi) > 3 * np.pi / 4 else 'G' if phi > 0 else 'D'
                if T >= pret[cap]:
                    inten = max(0.3, abs(np.cos(phi)))
                    fin[cap], pret[cap] = T + 100, T + 500
                    G['SENS'][cap].I = P_['K_SENS'] * inten
                    journal['contacts'].append((T, cap, inten, D.x, D.y))
            for cap in CAPTEURS:                               # fin de l'impulsion de 100 ms
                if T >= fin[cap]: G['SENS'][cap].I = 0
        return [monde]

    R = simuler_capteurs(None, N_PAIRES=N_PAIRES, duree=duree, extra=extra, **p)
    return R, np.array(journal['D'].traj), journal['contacts']
