'''Partie I, 1.3.7 : generalisation a N paires de pattes.

On change seulement N_PAIRES. Le delai optionnel delta (k * delta par segment) cree une onde
metachronale le long du corps.
'''
import matplotlib.pyplot as plt
from brian2 import ms

from cpg import simuler
from plotting import afficher_marche, sauver


def main():
    for N, dl in [(20, 5*ms), (200, 0.5*ms)]:
        fig, _ = afficher_marche(simuler(N_PAIRES=N, delta=dl), f'N_PAIRES = {N}, delta = {dl / ms:g} ms')
        sauver(fig, f'p1_N{N}')


if __name__ == '__main__':
    main()
    plt.show()
