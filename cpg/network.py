'''Briques du reseau (populations, synapses, pattes) et Partie I : systeme non autonome.

Patte i = 2k + s (k = segment, s = 0 gauche / 1 droite), groupe (k + s) % 2 -> source
swing = A (groupe 0) ou B (groupe 1). Le CPG (adaptation forte) produit une decharge par
phase : A et B alternent, et la frequence augmente avec I_drive.
'''
import numpy as np
from brian2 import NeuronGroup, Network, SpikeMonitor, Synapses, ms, second, start_scope

from .config import (B_ET, DELAI, NOMS, TAU_BLOC, W_AB, W_CONTRA, W_ET, W_INHIB, W_MOT, eqs)


def pop(n, tau=2*ms, tau_i=10*ms, tau_a=1*second, tau_s=50*ms, inc_a=0, Ib=0, I=0, seuil=1):
    '''Population LIF au modele commun.'''
    G = NeuronGroup(n, eqs, threshold='v > seuil', reset='v = 0; a += inc_a',
                    refractory=2*ms, method='euler')
    G.tau, G.tau_i, G.tau_a, G.tau_s = tau, tau_i, tau_a, tau_s
    G.inc_a, G.Ib, G.I, G.seuil = inc_a, Ib, I, seuil
    return G


def syn(src, tgt, w, i=None, j=None, inhib=False, lente=False, delay=0*ms, on_pre=None, model=''):
    '''Rapide : excitatrice (v += w) ou inhibitrice (gi += w). Lente : gs += w ou gs -= w.
    i = j = None : connexion tout-a-tout.'''
    if on_pre is None:
        var = 'gs_post' if lente else ('gi_post' if inhib else 'v_post')
        on_pre = f"{var} {'-' if lente and inhib else '+'}= {w}"
    S = Synapses(src, tgt, model=model, on_pre=on_pre)
    if i is None: S.connect()
    else: S.connect(i=np.asarray(i), j=np.asarray(j))
    S.delay = delay
    return S


def indices(N_PAIRES, delta=0*ms):
    '''Indices des pattes : n, segment k, cote s, sources swing / stance, delais CPG -> patte.'''
    n = 2 * N_PAIRES                          # patte i = 2k + s
    k, s = np.arange(n) // 2, np.arange(n) % 2
    grp = (k + s) % 2                         # tripode / alternance
    sw, st = grp, 1 - grp                     # source swing / stance (0 = A, 1 = B)
    d = DELAI + k * delta                     # onde metachronale (option)
    return n, k, s, sw, st, d


def construire_pattes(cpg, inh, N_PAIRES, delta=0*ms):
    '''8 neurones par patte (LEVER, POSER, AVANT, ARRIERE + 4 interneurones ET) et leurs
    synapses. Le biais de mode des interneurones est ajoute par l'appelant.'''
    n, k, s, sw, st, d = indices(N_PAIRES, delta)
    P = {nom: pop(n, tau_i=TAU_BLOC if nom.startswith('IN') else 10*ms) for nom in NOMS}
    S = []

    # LEVER ET POSER ne dependent pas de la direction forward ou backward
    # donc directement liees au CPG et recoivent du signal avant ARRIERE et AVANT.
    # C'est parce que la connection de passent pas par les interneurones qui,
    # avec le biais B_ET decide si on fait AVANT ou ARRIERE selong le mode.
    for nom, src in [('LEVER', sw), ('POSER', st)]:
        S.append(syn(cpg, P[nom], W_MOT, src, range(n), delay=d))

    # Synapses des interneurones avec le CPG avec un poids de W_ET = 0.6
    # comme ca les interneurones ne dechargent pas s'il n'ont pas le biais B_ET = 0.6.
    # On peut gerer la direction avec ce mecanisme.
    for nom, src in [('IN_sw_av', sw), ('IN_sw_ar', sw), ('IN_st_av', st), ('IN_st_ar', st)]:
        S.append(syn(cpg, P[nom], W_ET, src, range(n), delay=d))

    # Synapses des interneurones avec les neurones motrices avec poids de W_MOT = 1.5
    # pour s'assurer que si une interneurone decharge, la neurone AVANT/ARRIERE connecte
    # va depasser son seuil et decharger.
    for a_, b_ in [('IN_sw_av', 'AVANT'), ('IN_st_av', 'AVANT'), ('IN_sw_ar', 'ARRIERE'), ('IN_st_ar', 'ARRIERE')]:
        S.append(syn(P[a_], P[b_], W_MOT, range(n), range(n)))

    # Synapses d'inhibation entre les neurones de mouvements contraires
    # ARRIERE/AVANT et LEVER/POSER
    for a_, b_ in [('LEVER', 'POSER'), ('POSER', 'LEVER'), ('AVANT', 'ARRIERE'), ('ARRIERE', 'AVANT')]:
        S.append(syn(P[a_], P[b_], W_CONTRA, range(n), range(n), inhib=True))

    # Synapses de neurone d'inhibation INHIB_G et INHIB_D vers les interneuones
    # lorsqu'on veut tourner le compteur du cote choisi va decharger sur INHIB_G
    # ou INHIB_D (dependament du cote) certains cycle selon une frequence choisi.
    # Les interneurones pour le tripode seront bloquer et ne dechargerons pas
    # et un cote avancera plus que l'autre ce qui causera le virage.
    for nom in NOMS[4:]:
        S.append(syn(inh, P[nom], W_INHIB, s, range(n), inhib=True, delay=k * delta))
    return P, S


def cpg_pop(I_drive, **kw):
    '''CPG (A, B) : adaptation forte, B part adapte -> demarre en antiphase.'''
    cpg = pop(2, tau=10*ms, tau_i=20*ms, tau_a=200*ms, inc_a=1.5, I=I_drive, **kw)
    # A en index 0 commence avec v = 0.5 et a = 0 donc decharge avant
    # B en index 1 qui a v = 0 et a = 1.
    cpg.v, cpg.a = [0.5, 0], [0, 1]
    return cpg


def simuler(N_PAIRES=3, I_drive=2.0, mode_G='avant', mode_D='avant',
            virage_G=False, virage_D=False, n_saut=2, delta=0*ms, duree=2*second):
    '''Partie I : commandes fixees a la main (mode par cote, virage, I_drive).'''
    start_scope()
    n, _, s, *_ = indices(N_PAIRES, delta)
    avant = np.where(s == 0, mode_G == 'avant', mode_D == 'avant')

    cpg = cpg_pop(I_drive)
    cpt = pop(2, tau=100*second)              # COMPTEUR_G, COMPTEUR_D (fuite tres lente)
    inh = pop(2)                              # INHIB_G, INHIB_D
    P, S = construire_pattes(cpg, inh, N_PAIRES, delta)

    # On donne le biais au bonnes interneurones selon le mode. 
    # Ces neurones seront ceux les seules qui pourront decharger.
    for nom in ['IN_sw_av', 'IN_st_ar']: P[nom].Ib = B_ET * avant
    for nom in ['IN_sw_ar', 'IN_st_av']: P[nom].Ib = B_ET * ~avant
    S.append(syn(cpg, cpg, W_AB, [0, 1], [1, 0], inhib=True))

    # Virage : chaque salve de A ajoute 1/n_saut au compteur -> 1 decharge sur n_saut
    for cote, actif in enumerate([virage_G, virage_D]):
        if actif: S.append(syn(cpg, cpt, 1 / n_saut + 0.01, [0], [cote]))
    S.append(syn(cpt, inh, W_MOT, [0, 1], [0, 1]))

    M = {nom: SpikeMonitor(P[nom]) for nom in NOMS}
    M['CPG'], M['CPT'] = SpikeMonitor(cpg), SpikeMonitor(cpt)
    Network(cpg, cpt, inh, *P.values(), *S, *M.values()).run(duree)
    return {nom: (m.t / ms, np.array(m.i)) for nom, m in M.items()} | {'n': n, 'duree': duree / ms}
