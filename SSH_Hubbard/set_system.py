import jax.numpy as jnp
import os
import yaml
from SSH_Hubbard.lattice import *

def load_yaml_config(config_path='config.yaml'):
    """Load configuration from YAML file and return as dictionary"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_yaml_config()

#############################
### Set system parameters ###
#############################
# System parameters
t_hub = config['system']['t_hub']
omega = config['system']['omega']
alpha = config['system']['alpha']
U_hub = config['system']['U_hub']

N_e_up = config['system']['N_e_up']
N_e_do = L - N_e_up
N_e = N_e_up + N_e_do

# Create output directory
output_dir = "output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print(f"Directory '{output_dir}' created.")

# Simulation settings
name_output_simulations = config['simulation']['name_output_simulations']
SR_run = config['simulation']['SR_run']

# SR or measurement parameters
if SR_run:
    dt_step = config['sr_parameters']['dt_step']
    n_SR_steps = config['sr_parameters']['n_SR_steps']
    n_sweeps = config['sr_parameters']['n_sweeps']
else:
    N_blocks = config['measurement']['N_blocks']
    L_each_block = config['measurement']['L_each_block']

# Simulation runtime settings
MAX_time = config['simulation']['MAX_time']
read_params_from_out = config['simulation']['read_params_from_out']
step_to_read_params = config['simulation']['step_to_read_params']

# Monte Carlo parameters
p_spin_flip = config['monte_carlo']['p_spin_flip']
p_moving_electrons = config['monte_carlo']['p_moving_electrons']
displ_phon_move = config['monte_carlo']['displ_phon_move']
sparse_ave_length = config['monte_carlo'].get('sparse_ave_length', L)

# Phonon patterns
ALLOW_BOW_X_PHON = config['phonons']['ALLOW_BOW_X_PHON']
ALLOW_BOW_Y_PHON = config['phonons']['ALLOW_BOW_Y_PHON']

# Process special values
Q_x = jnp.pi if config['phonons']['Q_x'] == "pi" else float(config['phonons']['Q_x'])
Q_y = jnp.pi if config['phonons']['Q_y'] == "pi" else float(config['phonons']['Q_y'])

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
var_params = config['variational_parameters']

hopping_list = jnp.array(var_params['hopping_list'])
sWave_pair_list = jnp.array(var_params['sWave_pair_list'])
dWave_pair_list = jnp.array(var_params['dWave_pair_list'])
hopping_values = jnp.array(var_params['hopping_values'])
sWave_pair_values = jnp.array(var_params['sWave_pair_values'])
dWave_pair_values = jnp.array(var_params['dWave_pair_values'])

# Initialize jastrow arrays with proper sizes
jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-1)
phonons_XX_jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-1)
phonons_XY_jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-1)
phonons_YY_jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-1)
e_XPh_jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-2)
e_YPh_jastrow_values = jnp.zeros((1+Lx//2)*(1+Ly//2)-2)

g_phonons = var_params.get('g_phonons', 0.4)
f_phonons_Swave = var_params.get('f_phonons_Swave', 0.0)
f_phonons_dwave = var_params.get('f_phonons_dwave', 0.0)
z_phonons_X = var_params.get('z_phonons_X', 0.1)
z_phonons_Y = var_params.get('z_phonons_Y', 0.0)
rescaledX_omega = var_params.get('rescaledX_omega', 1.0)
rescaledY_omega = var_params.get('rescaledY_omega', 1.0)

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