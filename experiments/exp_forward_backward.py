'''Partie I, 1.3.1 a 1.3.3 : marche avant, marche arriere, ligne droite.

Tripode : G1, D2, G3 alternent avec D1, G2, D3. Autant de poussees a gauche qu'a droite ->
theta revient a 0 (seulement un leger dandinement du tripode). En marche arriere, meme rythme
du CPG ; seul le biais des interneurones change (IN_sw_ar et IN_st_av).
'''
import matplotlib.pyplot as plt

from cpg import simuler
from plotting import afficher_marche, sauver


def main():
    fig, traj = afficher_marche(simuler(mode_G='avant', mode_D='avant'), 'Marche avant (I_drive = 2)')
    sauver(fig, 'p1_marche_avant')
    print(f"Marche avant, position finale : x = {traj[-1, 1]:.2f}, y = {traj[-1, 2]:.2f}, theta = {traj[-1, 3]:.2f} rad")

    fig, traj = afficher_marche(simuler(mode_G='arriere', mode_D='arriere'), 'Marche arriere (I_drive = 2)')
    sauver(fig, 'p1_marche_arriere')
    print(f"Marche arriere, position finale : x = {traj[-1, 1]:.2f}, y = {traj[-1, 2]:.2f}, theta = {traj[-1, 3]:.2f} rad")


if __name__ == '__main__':
    main()
    plt.show()
