'''Partie I, 1.3.4 : variation de la vitesse.

On change seulement I_drive : la frequence du CPG, donc le nombre de poussees par seconde, augmente.
'''
import matplotlib.pyplot as plt
import numpy as np

from cpg import decoder, simuler
from plotting import sauver


def main():
    I_vals = [1.5, 2.0, 2.5, 3.0]
    freq, vit = [], []
    for I_d in I_vals:
        R = simuler(I_drive=I_d)
        traj = decoder(R)[0]
        freq.append(np.sum(R['CPG'][1] == 0) / (R['duree'] / 1000))   # salves de A par seconde
        vit.append(traj[-1, 1] / (R['duree'] / 1000))
        print(f'I_drive = {I_d} : frequence = {freq[-1]:.1f} Hz, vitesse = {vit[-1]:.2f} unites/s')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(I_vals, freq, 'o-'); ax[0].set(xlabel='I_drive', ylabel='Frequence de pas (Hz)')
    ax[1].plot(I_vals, vit, 'o-'); ax[1].set(xlabel='I_drive', ylabel='Vitesse (unites/s)')
    fig.tight_layout()
    sauver(fig, 'p1_vitesse')


if __name__ == '__main__':
    main()
    plt.show()
