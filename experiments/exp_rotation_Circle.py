'''Partie I, 1.3.5 et 1.3.6 : rotation sur place et deplacement en cercle.

Rotation : mode_G = avant, mode_D = arriere -> v_G = -v_D, donc v = 0 et le corps tourne.
Cercle : virage_D actif, COMPTEUR_D decharge une salve de A sur n_saut, INHIB_D bloque les
interneurones de direction droits pendant ~un cycle. Rayon theorique R = (L/2)(2n - 1),
rayon mesure = distance parcourue / |angle total|.
'''
import matplotlib.pyplot as plt
import numpy as np
from brian2 import second

from cpg import decoder, simuler
from plotting import afficher_marche, sauver


def rotation():
    fig, traj = afficher_marche(simuler(mode_G='avant', mode_D='arriere'), 'Rotation sur place')
    sauver(fig, 'p1_rotation')
    print(f"Rotation, position finale : x = {traj[-1, 1]:.2f}, y = {traj[-1, 2]:.2f}, theta = {traj[-1, 3]:.2f} rad")


def cercle():
    fig = plt.figure(figsize=(5, 5))
    for n_s in [1, 2, 3]:
        R = simuler(virage_D=True, n_saut=n_s, duree=8*second)
        traj = decoder(R)[0]
        dist = np.sum(np.hypot(np.diff(traj[:, 1]), np.diff(traj[:, 2])))
        print(f"n_saut = {n_s} : R theorique = {0.5 * (2 * n_s - 1):.2f} L, R mesure = {dist / abs(traj[-1, 3]):.2f} L")
        plt.plot(traj[:, 1], traj[:, 2], label=f'n_saut = {n_s}')
    plt.plot(0, 0, 'ko'); plt.axis('equal'); plt.xlabel('x'); plt.ylabel('y')
    plt.title('Virage a droite (L = 1)'); plt.legend()
    sauver(fig, 'p1_cercle')


def main():
    rotation()
    cercle()


if __name__ == '__main__':
    main()
    plt.show()
