'''Decodage des decharges motrices en mouvement, et mesures sur les resultats de simulation.

Contact entre POSER et LEVER ; AVANT/ARRIERE fixent la position ; une poussee = passage
avant -> arriere au sol (+d) ou arriere -> avant (-d). Chaque patte porte d / N_PAIRES pour que
la vitesse ne depende pas du nombre de pattes. Chaque poussee d'un cote fait avancer le corps
de ds/2 et tourner de +-ds/L.
'''
import numpy as np


class Decodeur:
    '''Contact (POSER/LEVER), position (AVANT/ARRIERE), poussees -> trajectoire (x, y, theta).
    Utilisable hors ligne (decoder) ou en ligne (boucle fermee, sensors.arene).'''
    def __init__(self, n, d=0.2, L=1.0, x=0.0, y=0.0, th=0.0, limites=None):
        self.n, self.d, self.L, self.limites = n, d, L, limites   # limites : (xmin, xmax, ymin, ymax)
        self.sol, self.pos = np.zeros(n, bool), np.zeros(n, int)  # pos : +1 avant, -1 arriere
        self.x, self.y, self.th = x, y, th
        self.traj, self.contact = [(0, x, y, th)], [[] for _ in range(n)]

    def evenement(self, t, i, c):
        if c == 'POSER' and not self.sol[i]: self.sol[i] = True; self.contact[i].append([t, None])
        elif c == 'LEVER' and self.sol[i]: self.sol[i] = False; self.contact[i][-1][1] = t
        elif c in ('AVANT', 'ARRIERE'):
            new = 1 if c == 'AVANT' else -1
            if self.sol[i] and self.pos[i] == -new:                 # poussee au sol
                ds = self.d / (self.n // 2) * (-new)                # +d si avant -> arriere
                self.th += ds / self.L * (1 if i % 2 else -1)       # cote D -> +theta (gauche)
                self.x += ds / 2 * np.cos(self.th); self.y += ds / 2 * np.sin(self.th)
                if self.limites:                                     # un mur bloque le corps
                    x0, x1, y0, y1 = self.limites
                    self.x, self.y = np.clip(self.x, x0, x1), np.clip(self.y, y0, y1)
                self.traj.append((t, self.x, self.y, self.th))
            self.pos[i] = new


def decoder(R, d=0.2, L=1.0):
    '''Decodage hors reseau -> (trajectoire [t, x, y, theta], intervalles de contact par patte).'''
    ev = []
    for c in ['LEVER', 'POSER', 'AVANT', 'ARRIERE']:
        t, i = R[c]
        ev += [(tt, ii, c) for tt, ii in zip(t, i)]
    D = Decodeur(R['n'], d, L)
    for e in sorted(ev): D.evenement(*e)
    for c in D.contact:
        if c and c[-1][1] is None: c[-1][1] = R['duree']
    return np.array(D.traj), D.contact


def pos_a(traj, t):
    '''Position (x, y) interpolee au temps t (ms).'''
    return np.interp(t, traj[:, 0], traj[:, 1]), np.interp(t, traj[:, 0], traj[:, 2])


def freq_pas(R):
    '''Frequence de pas instantanee (Hz) a partir des salves de A du CPG.'''
    t, i = R['CPG']
    ta = t[i == 0]
    return ta[1:], 1000 / np.diff(ta)


def activite(R, nom, idx, apres=0):
    '''Temps des decharges du neurone idx de la population nom apres 'apres' ms.'''
    t, i = R[nom]
    return t[(i == idx) & (t >= apres)]


def duree_active(R, nom, idx, apres=0):
    '''Duree (ms) entre la premiere et la derniere decharge d'un neurone (0 s'il ne decharge pas).'''
    t = activite(R, nom, idx, apres)
    return t[-1] - t[0] if len(t) > 1 else 0.0
