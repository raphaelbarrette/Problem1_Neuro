'''Modele LIF commun et parametres des Parties I et II.'''
from brian2 import BrianLogger, ms, prefs, second

prefs.codegen.target = 'numpy'
BrianLogger.log_level_error()   # masque les avertissements 'tau'

# Modele de la Partie II. Avec gs = 0 et seuil = 1 (valeurs par defaut), c'est exactement
# le modele de la Partie I : tau dv/dt = I + Ib - v - a - gi.
eqs = '''
dv/dt  = (I + Ib + gs - v - a - gi) / tau : 1 (unless refractory)
da/dt  = -a / tau_a                       : 1   # adaptation (CPG seulement)
dgi/dt = -gi / tau_i                      : 1   # inhibition synaptique rapide
dgs/dt = -gs / tau_s                      : 1   # courant synaptique lent (persistance)
I : 1                                           # I_drive (CPG) ou courant sensoriel k x intensite
Ib : 1                                          # biais tonique (mode Partie I, CMD_AV Partie II)
seuil : 1 (constant)                            # seuil de decharge (varie chez les capteurs)
inc_a : 1 (constant)                            # saut d'adaptation par decharge
tau : second (constant)
tau_a : second (constant)
tau_i : second (constant)
tau_s : second (constant)
'''

# ---------------------------------------------------------------- Partie I
W_AB = 4.0     # inhibition mutuelle A <-> B
W_MOT = 1.5     # source -> LEVER/POSER, IN -> AVANT/ARRIERE (suffit seul)
W_ET = 0.6     # CPG -> IN (insuffisant seul : principe du ET)
B_ET = 0.6     # biais de mode sur les IN (Partie I seulement)
W_CONTRA = 3.0     # LEVER <-> POSER, AVANT <-> ARRIERE
W_INHIB = 3.0     # INHIB_s -> IN du cote s
TAU_BLOC = 55*ms   # inhibition lente des IN : bloque ~ un cycle
DELAI = 3*ms    # latence CPG -> pattes

# ---------------------------------------------------------------- Partie II
PARAM = dict(
    M=10, K_SENS=2.0, SEUIL_MIN=0.2, SEUIL_MAX=1.8, # capteurs : courant = K_SENS x intensite
    TAU_PERSIST=800*ms, # synapses lentes capteur -> commande
    DELAI_SENS=0*ms, # latence capteur -> commande
    I_CMD=1.8, # biais tonique de CMD_AV (~100 Hz)
    W_WTA=1.0, # CMD_AV_s <-> CMD_AR_s
    W_MODE=0.5, G_MODE=0.9, TAU_MODE=50*ms, # CMD -> IN (synapse saturante)
    W_AV_AR=0.05, W_AV_AV=0.04, # SENS_AV -> CMD_AR (+), CMD_AV (-)
    W_AV_CPT=0.0, # option : SENS_AV -> COMPTEUR_G (recul en tournant)
    W_LAT_CPT=0.04, # SENS_D -> COMPTEUR_G (et symetrique)
    W_LAT_AR=0.025, # SENS_D -> CMD_AR_G, poids faible (pivot)
    W_AR_CPG=0.01, # SENS_AR -> A et B du CPG
    W_CPT=0.35, TAU_CPT=150*ms, # CPG(A) -> COMPTEUR, fuite du compteur
    W_AR_CPT=3.0, # CMD_AR_s -| COMPTEUR_s (le pivot remplace le virage)
    W_S_AV=2.0, W_S_LAT=2.0, W_S_GD=1.0, # selection d'action (inhibition rapide)
    HABITUATION=False, U_STD=0.05, TAU_REC=4*second, # bonus : depression SENS_AR -> CPG
)

NOMS = ['LEVER', 'POSER', 'AVANT', 'ARRIERE', 'IN_sw_av', 'IN_st_ar', 'IN_sw_ar', 'IN_st_av']
CAPTEURS = ['G', 'D', 'AV', 'AR']
CMD_NOMS = ['CMD_AV_G', 'CMD_AR_G', 'CMD_AV_D', 'CMD_AR_D']
