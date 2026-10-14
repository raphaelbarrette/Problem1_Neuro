'''Reseau CPG d'un hexapode (Brian2) : Partie I (network) et Partie II (sensors).'''
from . import config
from .body_model import activite, decoder, duree_active, freq_pas, pos_a
from .network import simuler
from .sensors import arene, recrutement, simuler_capteurs
