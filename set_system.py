import jax.numpy as jnp
from lattice import *

#############################
### Set system parameters ###
#############################
t_hub = -1.
omega = 1.
alpha = 0.4 
U_hub = 0. 

N_e_up = 32 
N_e_do = L - N_e_up 
N_e = N_e_up + N_e_do

name_output_simulations = "TEST"
SR_run = True # True: optimize WF using SR; False: measure

if SR_run==True:
    dt_step = 0.01
    n_SR_steps = 2
    n_sweeps = 400
else:
    N_blocks = 500
    L_each_block = 540


MAX_time = 41000 #Maximum time for the code to run (in seconds)
read_params_from_out = False #If set to True, tries to restart the system from old parameters
step_to_read_params = -1 #If set to -10, it will start the system from the -10th step of the last file of parameters


#Parameters
p_spin_flip = 0.25
p_moving_electrons = 0.5 # The probability of choosing to try to move a phonon at each Metropolis step is 1-p_moving_electrons
displ_phon_move = 0.5 # Each time the Metropolis tries to move a phonon, the move is extracted with uniform probability in the interval [-0.5*displ_phon_move,0.5*displ_phon_move]
sparse_ave_length = L #How many attentped Metropolis steps between two measures


#Phonons patterns allowed
ALLOW_BOW_X_PHON = True 
ALLOW_BOW_Y_PHON = False

Q_x = jnp.pi
Q_y = jnp.pi

stagg_x = jnp.zeros((L), dtype=float)
stagg_y = jnp.zeros((L), dtype=float)

for i in range(Lx):
    for j in range(Ly):
        site = j + i * Ly
        stagg_x = stagg_x.at[site].set(ALLOW_BOW_X_PHON * jnp.cos(Q_x * i) * jnp.cos(Q_y * j))
        stagg_y = stagg_y.at[site].set(ALLOW_BOW_Y_PHON * jnp.cos(Q_x * i) * jnp.cos(Q_y * j))

##################################
### Set variational parameters ###
##################################

hopping_list = jnp.array([0,2,]) # 0 is chemical potential, 1 is NN hopping, 2 is NNN hopping and so on
sWave_pair_list  = jnp.array([0,1,2]) # 0 is onsite pairing, 1 is NN sWave pairing, 2 is NNN sWave pairing and so on
dWave_pair_list  = jnp.array([0,]) # 0 is NN dWave pairing, 1 is NNN dWave pairing and 2 is NNNN dWave pairing. NO MORE!!!
hopping_values = jnp.array([-0.00341175,  0.00442574])
sWave_pair_values = jnp.array([0.06658055, 0.00332171, 0.02865038])
dWave_pair_values = jnp.array([0.04115974])

jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-1)

phonons_XX_jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-1)
phonons_XY_jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-1)
phonons_YY_jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-1)

e_XPh_jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-2)
e_YPh_jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-2)


#bZ = 0.2 
g_phonons = 0.4
f_phonons_Swave = 0.
f_phonons_dwave = 0.


z_phonons_X = 0.1
z_phonons_Y = 0.
rescaledX_omega = 1.
rescaledY_omega = 1.



#######################################################################
#Patterns possible for x and y displacements (Shown for Half-filling) #
#######################################################################

# X (pi,0) columnar pattern
#
# ALLOW_BOW_X_PHON = True
# ALLOW_BOW_Y_PHON = False
# Q_x = jnp.pi
# Q_y = 0
#
#
# - - - -
# - - - -
# - - - -
# - - - -
#

# X (pi,pi) pattern
#
# ALLOW_BOW_X_PHON = True
# ALLOW_BOW_Y_PHON = False
# Q_x = jnp.pi
# Q_y = jnp.pi
#
#
# - - - -
#  - - - -
# - - - -
#  - - - -
#

# Y (pi,0) columnar pattern
#
# ALLOW_BOW_X_PHON = False
# ALLOW_BOW_Y_PHON = True
# Q_x = jnp.pi
# Q_y = 0
# 
#
# |   |   |   |
#
# |   |   |   |
#
# |   |   |   |
#
# |   |   |   |
#

# Y (pi,pi) pattern
#
# ALLOW_BOW_X_PHON = False
# ALLOW_BOW_Y_PHON = True
# Q_x = jnp.pi
# Q_y = jnp.pi
# 
# 
# |   |   |   |
#   |   |   |   |
# |   |   |   |
#   |   |   |   |
# |   |   |   |
#   |   |   |   |
# |   |   |   |
#   |   |   |   |
#
