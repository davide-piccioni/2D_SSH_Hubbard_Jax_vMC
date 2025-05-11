from SSH_Hubbard.lattice import *
from SSH_Hubbard.set_system import *
from SSH_Hubbard.wave_function import *
from SSH_Hubbard.observables import *
from SSH_Hubbard.metropolis import *
from jax import jit
from jax.flatten_util import ravel_pytree
import time

####################################################
### Functions to optimize or measure observables ###
####################################################

#Change parameters according to Stochastic Reconfiguration (Natural gradient)
@jit
def SR_one_step_change_parameters(i, vals, epsilon=10e-4): 
    
    params, gamma, energy_array, param_array, mean_values = vals 

    # Average between chains
    S_matrix_TOT = mean_values['OO']
    O_x_TOT      = mean_values['O']
    E_O_x_TOT    = mean_values['eO']
    E_TOT        = mean_values['e']

    par_flat, unravel_params = ravel_pytree(params)

    S_matrix_TOT = S_matrix_TOT - jnp.outer(O_x_TOT,O_x_TOT) 
    E_O_x_TOT = -E_O_x_TOT + O_x_TOT*E_TOT 
    S_matrix_TOT = S_matrix_TOT + epsilon*jnp.diag(jnp.ones(S_matrix_TOT.shape[0]))

    # S_matrix_TOT is the Fisher information matrix
    # E_O_x_TOT is the gradient of the energy with respect to the parameters
    # If we took S_matrix_TOT = Identity, we would have the steepest descent

    alpha_dot=jnp.linalg.solve(S_matrix_TOT,E_O_x_TOT)

    norm_ = lax.max(jnp.linalg.norm(alpha_dot),2.) # Regularization of the change of parameters

    param_array = param_array.at[i,:].set(par_flat)

    par_flat += alpha_dot*gamma*(2./norm_)

    energy_array = energy_array.at[i].set(E_TOT)


    params = unravel_params(par_flat)
        
    return [params, gamma, energy_array, param_array]
    

# Perform N_iterations steps of Stochastic Reconfiguration (Natural gradient) optimization
def SR_n_iterations(N_iterations, nsweeps, state, params, get_U_mat, log_amplitude_NOT_determinant, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a, gamma, energy_array, params_array, MAX_time, epsilon=10e-4):

    start_time = time.time()

    state = state.replace(U = get_U_mat(params, state.X_Phonons, state.Y_Phonons) )
    state = state.replace(log_amp = wf(params, state.occupied_sites, state.xloc, state.S_z, state.X_Phonons, state.Y_Phonons) )

    state, key, a, _ = mc_sweeps_SR(nsweeps, state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

    print("\n", flush=True)
    for step in range(N_iterations):

        # Thermalization
        state, key, a = mc_sweeps_Thermaliz(int(nsweeps/20), state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons,  key, a)

        # Measure observables
        state, key, a, mean_values = mc_sweeps_SR(nsweeps, state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

        # Chage parameters and save them
        params, gamma, energy_array, params_array = SR_one_step_change_parameters(step, [params, gamma, energy_array, params_array, mean_values], epsilon)

        
        print(step, flush=True)
        print(a.acc_call_hop_NN/a.tot_call_hop_NN,a.acc_call_spin_flip/a.tot_call_spin_flip,a.acc_call_phonon_step/a.tot_call_phonon_step, flush=True)
        a = acceptance(0, 0, 0, 0, 0, 0)
        print("\n", flush=True)

        energy_array = energy_array.at[step].set(energy_array[step]/L)

        if (time.time() - start_time) > MAX_time:
            break

    return state, params, key, a, energy_array, params_array

# Save observables at fixed variational parameters
# To be used in the measurement simulation after the parameters have been optimized
@jit
def save_observables(step, mean_values, energy_array, observables):

    corr_n_array, Nq_array, corr_s_array, Sq_array, corr_bx_array, Bx_q_array, corr_by_array, By_q_array, order_par_bond_x_array, order_par_bond_y_array, bond_amplitudes_array_X, bond_amplitudes_array_Y, X_diff_array, Y_diff_array, corr_dx_array, corr_dy_array, corr_dxy_array, corr_d_onsite_array = observables

    # Average between chains        
    energy = mean_values['e']
    corrn = mean_values['corr_n']
    Nq = mean_values['N_q']
    corrs = mean_values['corr_s']
    Sq = mean_values['S_q']
    corrbx = mean_values['corr_bx']
    Bxq = mean_values['Bx_q']
    corrby = mean_values['corr_by']
    Byq = mean_values['By_q'] 
    ordbondx = mean_values['order_bonds_x']
    ordbondy = mean_values['order_bonds_y']
    bondampx = mean_values['bond_amplitudes_x']
    bondampy = mean_values['bond_amplitudes_y']
    Xdiff = mean_values['X_diff']
    Ydiff = mean_values['Y_diff']
    corrdx = mean_values['corr_dx']
    corrdy = mean_values['corr_dy']
    corrdxy = mean_values['corr_dxy']
    corrd_onsite = mean_values['corr_d_onsite']


    
    energy_array  =  energy_array.at[step].set(energy/(L))
    corr_n_array  =  corr_n_array.at[step,:].set(corrn)
    Nq_array      =  Nq_array.at[step,:].set(Nq)
    corr_s_array  =  corr_s_array.at[step,:].set(corrs)
    Sq_array      =  Sq_array.at[step,:].set(Sq)
    corr_bx_array =  corr_bx_array.at[step,:].set(corrbx)
    Bx_q_array    =  Bx_q_array.at[step,:].set(Bxq)
    corr_by_array =  corr_by_array.at[step,:].set(corrby)
    By_q_array    =  By_q_array.at[step,:].set(Byq)
    order_par_bond_x_array  =  order_par_bond_x_array.at[step].set(ordbondx)
    order_par_bond_y_array  =  order_par_bond_y_array.at[step].set(ordbondy)
    bond_amplitudes_array_X =  bond_amplitudes_array_X.at[step,:].set(bondampx)
    bond_amplitudes_array_Y =  bond_amplitudes_array_Y.at[step,:].set(bondampy)
    X_diff_array  =  X_diff_array.at[step,:].set(Xdiff)
    Y_diff_array  =  Y_diff_array.at[step,:].set(Ydiff)
    corr_dx_array =  corr_dx_array.at[step,:].set(corrdx)
    corr_dy_array =  corr_dy_array.at[step,:].set(corrdy)
    corr_dxy_array=  corr_dxy_array.at[step,:].set(corrdxy)
    corr_d_onsite_array=  corr_d_onsite_array.at[step,:].set(corrd_onsite)

    return energy_array, [ corr_n_array, Nq_array, corr_s_array, Sq_array, corr_bx_array, Bx_q_array, corr_by_array, By_q_array, order_par_bond_x_array, order_par_bond_y_array, bond_amplitudes_array_X, bond_amplitudes_array_Y, X_diff_array, Y_diff_array, corr_dx_array, corr_dy_array, corr_dxy_array, corr_d_onsite_array]



# Perform N_blocks steps of measurement
# To be used in the measurement simulation after the parameters have been optimized                    
def block_average(N_blocks, L_each_block, state, params, get_U_mat, log_amplitude_NOT_determinant, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a, energy_array, observables, MAX_time):

    start_time = time.time()

    state = state.replace(U = get_U_mat(params, state.X_Phonons, state.Y_Phonons) )
    state = state.replace(log_amp = wf(params, state.occupied_sites, state.xloc, state.S_z, state.X_Phonons, state.Y_Phonons) )

    state, key, a, _ = mc_sweeps_Measure(L_each_block, state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

    print("\n", flush=True)
    for step in range(N_blocks):

        state, key, a, mean_values = mc_sweeps_Measure(L_each_block, state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

        energy_array, observables = save_observables(step, mean_values, energy_array, observables)

        print(step, flush=True)
        print(a.acc_call_hop_NN/a.tot_call_hop_NN,a.acc_call_spin_flip/a.tot_call_spin_flip,a.acc_call_phonon_step/a.tot_call_phonon_step, flush=True)
        a = acceptance(0, 0, 0, 0, 0, 0)
        print("\n", flush=True)
#        if (time.time() - start_time) > MAX_time:
#            break

    return state, energy_array, observables 
