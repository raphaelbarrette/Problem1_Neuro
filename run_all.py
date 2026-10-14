'''Lance toutes les experiences et enregistre les figures dans figures/ (sans les afficher).

Usage : python run_all.py
'''
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments import (exp_forward_backward, exp_part2_stimuli, exp_rotation_Circle, exp_scaling,
                         exp_speed)

EXPERIENCES = [exp_forward_backward, exp_speed, exp_rotation_Circle, exp_scaling, exp_part2_stimuli]

if __name__ == '__main__':
    for exp in EXPERIENCES:
        print(f'===== {exp.__name__}')
        exp.main()
        plt.close('all')
