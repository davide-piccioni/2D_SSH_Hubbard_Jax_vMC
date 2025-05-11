########################
### Double precision ###
########################
import os
os.environ["OMP_NUM_THREADS"]='1'
from jax.config import config
config.update("jax_enable_x64", True)

###############
### MPI4jax ###
###############
from mpi4py import MPI
import jax.numpy as jnp
import mpi4jax
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_chains = comm.Get_size()

from SSH_Hubbard.lattice import *
from SSH_Hubbard.set_system import *
from SSH_Hubbard.wave_function import *
from SSH_Hubbard.observables import *
from SSH_Hubbard.metropolis import *
from jax import random, jit, grad, hessian
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
    S_matrix_TOT, tok1 = mpi4jax.reduce(mean_values['OO'], op=MPI.SUM, root=0, comm=comm)
    O_x_TOT , tok2 = mpi4jax.reduce(mean_values['O'], op=MPI.SUM, root=0, comm=comm, token=tok1)
    E_O_x_TOT , tok3 = mpi4jax.reduce(mean_values['eO'], op=MPI.SUM, root=0, comm=comm, token=tok2)
    E_TOT, tok4 = mpi4jax.reduce(mean_values['e'], op=MPI.SUM, root=0, comm=comm, token=tok3)

    par_flat, unravel_params = ravel_pytree(params)

    if(rank ==0):
        S_matrix_TOT = S_matrix_TOT/n_chains;
        O_x_TOT= O_x_TOT/n_chains;
        E_O_x_TOT = E_O_x_TOT/n_chains;
        E_TOT = E_TOT/n_chains;

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


    par_flat, _ = mpi4jax.bcast(par_flat, root=0, comm=comm, token=tok4)

    params = unravel_params(par_flat)
        
    return [params, gamma, energy_array, param_array]
    

# Perform N_iterations steps of Stochastic Reconfiguration (Natural gradient) optimization
def SR_n_iterations(N_iterations, nsweeps, state, params, get_U_mat, log_amplitude_NOT_determinant, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a, gamma, energy_array, params_array, MAX_time, epsilon=10e-4):

    start_time = time.time()

    state = state.replace(U = get_U_mat(params, state.X_Phonons, state.Y_Phonons) )
    state = state.replace(log_amp = wf(params, state.occupied_sites, state.xloc, state.S_z, state.X_Phonons, state.Y_Phonons) )

    state, key, a, _ = mc_sweeps_SR(nsweeps, state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

    if (rank==0):
        print("\n", flush=True)
    for step in range(N_iterations):

        # Thermalization
        state, key, a = mc_sweeps_Thermaliz(int(nsweeps/20), state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons,  key, a)

        # Measure observables
        state, key, a, mean_values = mc_sweeps_SR(nsweeps, state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

        # Chage parameters and save them
        params, gamma, energy_array, params_array = SR_one_step_change_parameters(step, [params, gamma, energy_array, params_array, mean_values], epsilon)

        if (rank==0):
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
    energy, tok1 = mpi4jax.reduce(mean_values['e'], op=MPI.SUM, root=0, comm=comm)
    corrn, tok2 = mpi4jax.reduce(mean_values['corr_n'], op=MPI.SUM, root=0, comm=comm, token=tok1)
    Nq, tok3 = mpi4jax.reduce(mean_values['N_q'], op=MPI.SUM, root=0, comm=comm, token=tok2)
    corrs, tok4 = mpi4jax.reduce(mean_values['corr_s'], op=MPI.SUM, root=0, comm=comm, token=tok3)
    Sq, tok5 = mpi4jax.reduce(mean_values['S_q'], op=MPI.SUM, root=0, comm=comm, token=tok4)
    corrbx, tok6 = mpi4jax.reduce(mean_values['corr_bx'], op=MPI.SUM, root=0, comm=comm, token=tok5)
    Bxq, tok7 = mpi4jax.reduce(mean_values['Bx_q'], op=MPI.SUM, root=0, comm=comm, token=tok6) 
    corrby, tok8 = mpi4jax.reduce(mean_values['corr_by'], op=MPI.SUM, root=0, comm=comm, token=tok7)
    Byq, tok9 = mpi4jax.reduce(mean_values['By_q'], op=MPI.SUM, root=0, comm=comm, token=tok8) 
    ordbondx, tok10 = mpi4jax.reduce(mean_values['order_bonds_x'], op=MPI.SUM, root=0, comm=comm, token=tok9)
    ordbondy, tok11 = mpi4jax.reduce(mean_values['order_bonds_y'], op=MPI.SUM, root=0, comm=comm, token=tok10)
    bondampx, tok12 = mpi4jax.reduce(mean_values['bond_amplitudes_x'], op=MPI.SUM, root=0, comm=comm, token=tok11)
    bondampy, tok13 = mpi4jax.reduce(mean_values['bond_amplitudes_y'], op=MPI.SUM, root=0, comm=comm, token=tok12)
    Xdiff, tok14 = mpi4jax.reduce(mean_values['X_diff'], op=MPI.SUM, root=0, comm=comm, token=tok13)
    Ydiff, tok15 = mpi4jax.reduce(mean_values['Y_diff'], op=MPI.SUM, root=0, comm=comm, token=tok14)
    corrdx, tok16 = mpi4jax.reduce(mean_values['corr_dx'], op=MPI.SUM, root=0, comm=comm, token=tok15)
    corrdy, tok17 = mpi4jax.reduce(mean_values['corr_dy'], op=MPI.SUM, root=0, comm=comm, token=tok16)
    corrdxy, tok18 = mpi4jax.reduce(mean_values['corr_dxy'], op=MPI.SUM, root=0, comm=comm, token=tok17)
    corrd_onsite, tok19 = mpi4jax.reduce(mean_values['corr_d_onsite'], op=MPI.SUM, root=0, comm=comm, token=tok18)


    if(rank ==0):
        energy_array  =  energy_array.at[step].set(energy/(L*n_chains))
        corr_n_array  =  corr_n_array.at[step,:].set(corrn/n_chains)
        Nq_array      =  Nq_array.at[step,:].set(Nq/n_chains)
        corr_s_array  =  corr_s_array.at[step,:].set(corrs/n_chains)
        Sq_array      =  Sq_array.at[step,:].set(Sq/n_chains)
        corr_bx_array =  corr_bx_array.at[step,:].set(corrbx/n_chains)
        Bx_q_array    =  Bx_q_array.at[step,:].set(Bxq/n_chains)
        corr_by_array =  corr_by_array.at[step,:].set(corrby/n_chains)
        By_q_array    =  By_q_array.at[step,:].set(Byq/n_chains)
        order_par_bond_x_array  =  order_par_bond_x_array.at[step].set(ordbondx/n_chains)
        order_par_bond_y_array  =  order_par_bond_y_array.at[step].set(ordbondy/n_chains)
        bond_amplitudes_array_X =  bond_amplitudes_array_X.at[step,:].set(bondampx/n_chains)
        bond_amplitudes_array_Y =  bond_amplitudes_array_Y.at[step,:].set(bondampy/n_chains)
        X_diff_array  =  X_diff_array.at[step,:].set(Xdiff/n_chains)
        Y_diff_array  =  Y_diff_array.at[step,:].set(Ydiff/n_chains)
        corr_dx_array =  corr_dx_array.at[step,:].set(corrdx/n_chains)
        corr_dy_array =  corr_dy_array.at[step,:].set(corrdy/n_chains)
        corr_dxy_array=  corr_dxy_array.at[step,:].set(corrdxy/n_chains)
        corr_d_onsite_array=  corr_d_onsite_array.at[step,:].set(corrd_onsite/n_chains)


    return energy_array, [ corr_n_array, Nq_array, corr_s_array, Sq_array, corr_bx_array, Bx_q_array, corr_by_array, By_q_array, order_par_bond_x_array, order_par_bond_y_array, bond_amplitudes_array_X, bond_amplitudes_array_Y, X_diff_array, Y_diff_array, corr_dx_array, corr_dy_array, corr_dxy_array, corr_d_onsite_array]



# Perform N_blocks steps of measurement
# To be used in the measurement simulation after the parameters have been optimized                    
def block_average(N_blocks, L_each_block, state, params, get_U_mat, log_amplitude_NOT_determinant, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a, energy_array, observables, MAX_time):

    start_time = time.time()

    state = state.replace(U = get_U_mat(params, state.X_Phonons, state.Y_Phonons) )
    state = state.replace(log_amp = wf(params, state.occupied_sites, state.xloc, state.S_z, state.X_Phonons, state.Y_Phonons) )

    state, key, a, _ = mc_sweeps_Measure(L_each_block, state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

    if (rank==0):
        print("\n", flush=True)
    for step in range(N_blocks):

        state, key, a, mean_values = mc_sweeps_Measure(L_each_block, state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

        energy_array, observables = save_observables(step, mean_values, energy_array, observables)

        if (rank==0):
            print(step, flush=True)
            print(a.acc_call_hop_NN/a.tot_call_hop_NN,a.acc_call_spin_flip/a.tot_call_spin_flip,a.acc_call_phonon_step/a.tot_call_phonon_step, flush=True)
            a = acceptance(0, 0, 0, 0, 0, 0)
            print("\n", flush=True)
#        if (time.time() - start_time) > MAX_time:
#            break

    return state, energy_array, observables 



##############################
### Calling model and grad ###
##############################

key = random.PRNGKey(rank)
key, *keys_init = random.split(key, num=5)

######################################
### Generate system configurations ###
######################################

nonzeros_up = jnp.sort(jnp.nonzero(random.permutation(keys_init[0],jnp.concatenate((jnp.zeros(L-N_e_up), jnp.ones(N_e_up)), axis=None)))[0])
nonzeros_do = jnp.sort(jnp.nonzero(random.permutation(keys_init[1],jnp.concatenate((jnp.zeros(L-N_e_do), jnp.ones(N_e_do)), axis=None)))[0])

x_loc = jnp.zeros(two_L)
x_loc = x_loc.at[nonzeros_up].set(jnp.arange(N_e_up)+1)
x_loc = x_loc.at[nonzeros_do+L].set(N_e_up+1+jnp.arange(N_e_do))
x_loc = x_loc.astype(int)

occupied_sites_list = jnp.nonzero(x_loc)[0]
appo_x_first = x_loc.astype(jnp.bool_)
appo_x_second = appo_x_first.astype(jnp.int32)
S_z_ = 1.*(appo_x_second[:L] - appo_x_second[L:2*L])

X_Phonons_ = (random.uniform(keys_init[2], (L,) ) - 0.5)*0.02
Y_Phonons_ = (random.uniform(keys_init[3], (L,) ) - 0.5)*0.02

####################################
### Generate model and gradients ###
####################################

state_system = state_Fermions_and_Bosons(xloc=x_loc, occupied_sites=occupied_sites_list, S_z=S_z_, X_Phonons=X_Phonons_, Y_Phonons=Y_Phonons_, U =jnp.zeros((two_L,two_L)), log_amp=0., sign=1. )

key, sub_key = random.split(key, num=2)
model = slater_det(hopping_list=hopping_list, sWave_pair_list=sWave_pair_list, dWave_pair_list=dWave_pair_list, )
params = model.init(sub_key, state_system.occupied_sites, state_system.xloc, state_system.S_z, state_system.X_Phonons, state_system.Y_Phonons)

@jit
def wf(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons):
    return model.apply(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons)

@jit
def wf_grad(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons):
    return grad(model.apply)(params, occupied_sites, xloc,S_z,X_Phonons,Y_Phonons)

@jit
def wf_grad_X_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons):
    return grad(model.apply, argnums=4)(params, occupied_sites, xloc, S_z, X_Phonons, Y_Phonons)

@jit
def wf_lapl_X_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons):
#    return jnp.diag(jacfwd( wf_grad_X_phonons, argnums=4)(params, occupied_sites, xloc, S_z, X_Phonons, Y_Phonons))
    return jnp.trace(hessian(wf, argnums=4)(params, occupied_sites, xloc, S_z, X_Phonons, Y_Phonons ))

@jit
def wf_grad_Y_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons):
    return grad(model.apply, argnums=5)(params, occupied_sites, xloc, S_z, X_Phonons, Y_Phonons)

@jit
def wf_lapl_Y_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons):
#    return jnp.diag(jacfwd( wf_grad_Y_phonons, argnums=5)(params, occupied_sites, xloc, S_z, X_Phonons, Y_Phonons))
    return jnp.trace(hessian(wf, argnums=5)(params, occupied_sites, xloc, S_z, X_Phonons, Y_Phonons ))

@jit
def log_ph_amplitude(params,X_Phonons,Y_Phonons, xloc):
    return model.get_only_log_phonons_amplitude(params,X_Phonons,Y_Phonons, xloc)

@jit
def log_jastrow_amplitude(params,S_z,xloc):
    return model.get_only_log_Jastrow_amplitude(params,S_z,xloc)

@jit
def log_amplitude_NOT_determinant(params, S_z, xloc, X_Phonons, Y_Phonons):
    return log_ph_amplitude(params,X_Phonons,Y_Phonons, xloc) + log_jastrow_amplitude(params,S_z,xloc)

@jit
def get_U_mat(params,X_Phonons,Y_Phonons):
    return model.get_U(params,X_Phonons,Y_Phonons) 

################################
### Setting model parameters ###
################################

par_flat, unravel_params = ravel_pytree(params)

#par_flat = par_flat.at[0].set(bZ)
par_flat = par_flat.at[0].set(f_phonons_Swave)
par_flat = par_flat.at[1].set(f_phonons_dwave)
par_flat = par_flat.at[2].set(g_phonons)
index_ant = 3

par_flat = par_flat.at[index_ant:index_ant+len(hopping_list)].set(hopping_values)
index_ant += len(hopping_list)

par_flat = par_flat.at[index_ant:index_ant+len(sWave_pair_list)].set(sWave_pair_values)
index_ant += len(sWave_pair_list)

par_flat = par_flat.at[index_ant:index_ant+len(dWave_pair_list)].set(dWave_pair_values)
index_ant += len(dWave_pair_list)

par_flat = par_flat.at[index_ant:index_ant+len(phonons_XX_jastrow_values)].set(phonons_XX_jastrow_values)
index_ant += len(phonons_XX_jastrow_values)

par_flat = par_flat.at[index_ant:index_ant+len(phonons_XY_jastrow_values)].set(phonons_XY_jastrow_values)
index_ant += len(phonons_XY_jastrow_values)

par_flat = par_flat.at[index_ant:index_ant+len(phonons_YY_jastrow_values)].set(phonons_YY_jastrow_values)
index_ant += len(phonons_YY_jastrow_values)

par_flat = par_flat.at[index_ant:index_ant+len(e_XPh_jastrow_values)].set(e_XPh_jastrow_values)
index_ant += len(e_XPh_jastrow_values)

par_flat = par_flat.at[index_ant:index_ant+len(e_YPh_jastrow_values)].set(e_YPh_jastrow_values)
index_ant += len(e_YPh_jastrow_values)

par_flat = par_flat.at[index_ant:index_ant+len(jastrow_values)].set(jastrow_values)
index_ant += len(jastrow_values)


par_flat = par_flat.at[-4].set(rescaledX_omega)
par_flat = par_flat.at[-3].set(rescaledY_omega)
par_flat = par_flat.at[-2].set(z_phonons_X)
par_flat = par_flat.at[-1].set(z_phonons_Y)
index_ant += 4


if read_params_from_out==True:
    file_names = os.listdir("./output/")
    file_names.sort()
    try:
        with open("./output/" + file_names[-1], 'rb') as f:
            energy_array_OLD = jnp.load(f)
            param_array_OLD = jnp.load(f)

        if (SR_run == True):
            if jnp.shape(param_array_OLD)[0]<=step_to_read_params:
                step_to_read_params = -1

            par_flat = param_array_OLD[step_to_read_params,:]
        else:
            if jnp.shape(param_array_OLD)[0]<=step_to_read_params:
                step_to_read_params = -101
            for j in range(len(par_flat)):
                par_flat = par_flat.at[j].set( jnp.mean(param_array_OLD[step_to_read_params:-1,j]) )
    except IOError as e:
        print(f"Couldn't open file ({e}), using the parameters that were set inside the code!")

#par_flat = par_flat.at[-2].set(0.)
#par_flat = par_flat.at[-1].set(0.)

params = unravel_params(par_flat)

X_Phonons_ +=  par_flat[-2]*stagg_x
Y_Phonons_ +=  par_flat[-1]*stagg_y
state_system = state_system.replace(X_Phonons=X_Phonons_)
state_system = state_system.replace(Y_Phonons=Y_Phonons_)


state_system = state_system.replace( log_amp = wf(params, state_system.occupied_sites, state_system.xloc, state_system.S_z, state_system.X_Phonons, state_system.Y_Phonons) )
grad_log_parameters = wf_grad(params, state_system.occupied_sites, state_system.xloc, state_system.S_z, state_system.X_Phonons, state_system.Y_Phonons)
grad_log_X_phonons = wf_grad_X_phonons(params, state_system.occupied_sites, state_system.xloc, state_system.S_z, state_system.X_Phonons, state_system.Y_Phonons)
lapl_log_X_phonons = wf_lapl_X_phonons(params, state_system.occupied_sites, state_system.xloc, state_system.S_z, state_system.X_Phonons, state_system.Y_Phonons)
grad_log_Y_phonons = wf_grad_Y_phonons(params, state_system.occupied_sites, state_system.xloc, state_system.S_z, state_system.X_Phonons, state_system.Y_Phonons)
lapl_log_Y_phonons = wf_lapl_Y_phonons(params, state_system.occupied_sites, state_system.xloc, state_system.S_z, state_system.X_Phonons, state_system.Y_Phonons)
log_phonons_amplitude = log_ph_amplitude(params, state_system.X_Phonons, state_system.Y_Phonons, state_system.xloc)
log_Jstrw_amplitude = log_jastrow_amplitude(params, state_system.S_z, state_system.xloc)
log_no_DET_amplitude = log_amplitude_NOT_determinant(params, state_system.S_z, state_system.xloc, state_system.X_Phonons, state_system.Y_Phonons)
state_system = state_system.replace(U =  get_U_mat(params, state_system.X_Phonons, state_system.Y_Phonons))

if (rank==0):
    print(params)
    print("\n\n")

#####################################
### Running the optimization      ###
### of the variational parameters ###
#####################################

if (SR_run == True): 

    if(rank==0):
        print("Stochastic Reconfiguration simulation")
        print("dt = ",dt_step)
        print("SR steps = ",n_SR_steps)
        print("N_block_per_step = ",n_sweeps)
        print("P_spin_flip = ", p_spin_flip)
        print("P_moving_electrons = ", p_moving_electrons)
        print("Displacement_move_Phonons = ", displ_phon_move)
        print("Sparse averaging step = ",sparse_ave_length)
        print("\nL = ",L)
        print("N_e_up = ",N_e_up)
        print("N_e_do = ",N_e_do)
        print("\nt_hub = ",t_hub)
        print("omega = ",omega)
        print("alpha = ",alpha)
        print("U = ",U_hub)

    a = acceptance(0, 0, 0, 0, 0, 0)	
    earr = jnp.zeros(n_SR_steps)
    parr = jnp.zeros((n_SR_steps,index_ant))

    import time
    st = time.time()

    state_system, params, key, a, earr, parr = SR_n_iterations(n_SR_steps, int(n_sweeps/n_chains), state_system, params, get_U_mat, log_amplitude_NOT_determinant, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a, dt_step, earr, parr, MAX_time)

    et = time.time()
    #### get the execution time
    elapsed_time = et - st

    if(rank==0):
        print('Execution time:', elapsed_time, 'seconds')

        out_name = str(jnp.floor(et))
        out_name = out_name[:len(out_name)-1]

        len_reshaped  = jnp.nonzero(earr)[0][-1] + 1
        earr_reshaped = earr[:len_reshaped]
        parr_reshaped = parr[:len_reshaped,:]
	    
        print(earr_reshaped)
        print(parr_reshaped)
        with open('./output/SR_energy_pars_'+name_output_simulations+'_'+out_name+'npy', 'wb') as f:
           jnp.save(f, earr_reshaped)
           jnp.save(f, parr_reshaped)

        def partial_ave(e):
            N = len(e)
            prog_ave = jnp.zeros(N)
            for i in range(50):
                prog_ave = prog_ave.at[i].set(jnp.mean(e[0:i+1]))
            for i in range(50,N):
                prog_ave = prog_ave.at[i].set(jnp.mean(e[i-50:i]))
            return prog_ave

        max_diff =[]
        mean_diff = []

        for i in range(parr_reshaped.shape[1]):
            max_diff.append(jnp.max(jnp.power(parr_reshaped[:,i]-partial_ave(parr_reshaped[:,i]),2)[-100:]))
            mean_diff.append(jnp.mean(jnp.power(parr_reshaped[:,i]-partial_ave(parr_reshaped[:,i]),2)[-100:]))

        print(max(max_diff)<0.005)
        print(max(mean_diff)<0.0008)
        print(max(max_diff))
        print(max(mean_diff))


        print(params['params'])


##############################
### Measuring correlations ###
### at fixed parameters    ###
##############################

else:
    if(rank==0):
        print("Measurement simulation with fixed parameters")
        print("N_blocs = ",N_blocks)
        print("L_each_block = ",L_each_block)
        print("P_spin_flip = ", p_spin_flip)
        print("P_moving_electrons = ", p_moving_electrons)
        print("Displacement_move_Phonons = ", displ_phon_move)
        print("Sparse averaging step = ",sparse_ave_length)
        print("\nL = ",L)
        print("N_e_up = ",N_e_up)
        print("N_e_do = ",N_e_do)
        print("\nt_hub = ",t_hub)
        print("omega = ",omega)
        print("alpha = ",alpha)
        print("U = ",U_hub)

    a = acceptance(0, 0, 0, 0, 0, 0)

    earr = jnp.zeros(N_blocks)
    corr_n_arr = jnp.zeros((N_blocks,L))
    Nq_arr = jnp.zeros((N_blocks,L))
    corr_s_arr = jnp.zeros((N_blocks,L))
    Sq_arr = jnp.zeros((N_blocks,L))
    corr_bx_arr = jnp.zeros((N_blocks,L))
    Bx_q_arr = jnp.zeros((N_blocks,L))
    corr_by_arr = jnp.zeros((N_blocks,L))
    By_q_arr = jnp.zeros((N_blocks,L))
    order_par_bond_x = jnp.zeros(N_blocks)
    order_par_bond_y = jnp.zeros(N_blocks)
    bond_amplitudes_arr_X = jnp.zeros((N_blocks,L))
    bond_amplitudes_arr_Y = jnp.zeros((N_blocks,L))
    X_diff_arr = jnp.zeros((N_blocks,L))
    Y_diff_arr = jnp.zeros((N_blocks,L))
    corr_dx_arr = jnp.zeros((N_blocks,L))
    corr_dy_arr = jnp.zeros((N_blocks,L))
    corr_dxy_arr = jnp.zeros((N_blocks,L))
    corr_d_onsite_arr = jnp.zeros((N_blocks,L))


    import time
    st = time.time()

    state_system, earr, observables = block_average(N_blocks, int(L_each_block/n_chains), state_system, params, get_U_mat, log_amplitude_NOT_determinant, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a, earr, [corr_n_arr, Nq_arr, corr_s_arr, Sq_arr, corr_bx_arr, Bx_q_arr, corr_by_arr, By_q_arr, order_par_bond_x, order_par_bond_y, bond_amplitudes_arr_X, bond_amplitudes_arr_Y, X_diff_arr, Y_diff_arr, corr_dx_arr, corr_dy_arr, corr_dxy_arr, corr_d_onsite_arr ], MAX_time)

    corr_n_arr, Nq_arr, corr_s_arr, Sq_arr, corr_bx_arr, Bx_q_arr, corr_by_arr, By_q_arr, order_par_bond_x, order_par_bond_y, bond_amplitudes_arr_X, bond_amplitudes_arr_Y, X_diff_arr, Y_diff_arr, corr_dx_arr, corr_dy_arr, corr_dxy_arr, corr_d_onsite_arr = observables


    et = time.time()
    # get the execution time
    elapsed_time = et - st


    if(rank==0):
        print('Execution time:', elapsed_time, 'seconds')

        print(jnp.mean(earr))
        print(jnp.std(earr)/jnp.sqrt(N_blocks))
        print("\n")
        print("\n")

        print("n_r")
        for i in range(L):
            print(jnp.mean(corr_n_arr[:,i])," ± ",jnp.std(corr_n_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("N_q")
        for i in range(L):
            print(jnp.mean(Nq_arr[:,i])," ± ",jnp.std(Nq_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("S_r")
        for i in range(L):
            print(jnp.mean(corr_s_arr[:,i])," ± ",jnp.std(corr_s_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("S_q")
        for i in range(L):
            print(jnp.mean(Sq_arr[:,i])," ± ",jnp.std(Sq_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("Bx_r")
        for i in range(L):
            print(jnp.mean(corr_bx_arr[:,i])," ± ",jnp.std(corr_bx_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("Bx_q")
        for i in range(L):
            print(jnp.mean(Bx_q_arr[:,i])," ± ",jnp.std(Bx_q_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("By_r")
        for i in range(L):
            print(jnp.mean(corr_by_arr[:,i])," ± ",jnp.std(corr_by_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("By_q")
        for i in range(L):
            print(jnp.mean(By_q_arr[:,i])," ± ",jnp.std(By_q_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("Bond_x")
        for i in range(L):
            print(jnp.mean(bond_amplitudes_arr_X[:,i])," ± ",jnp.std(bond_amplitudes_arr_X[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("Bond_y")
        for i in range(L):
            print(jnp.mean(bond_amplitudes_arr_Y[:,i])," ± ",jnp.std(bond_amplitudes_arr_Y[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("X_diff")
        for i in range(L):
            print(jnp.mean(X_diff_arr[:,i])," ± ",jnp.std(X_diff_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("Y_diff")
        for i in range(L):
            print(jnp.mean(Y_diff_arr[:,i])," ± ",jnp.std(Y_diff_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("dx_r")
        for i in range(L):
            print(jnp.mean(corr_dx_arr[:,i])," ± ",jnp.std(corr_dx_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("dy_r")
        for i in range(L):
            print(jnp.mean(corr_dy_arr[:,i])," ± ",jnp.std(corr_dy_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("dxy_r")
        for i in range(L):
            print(jnp.mean(corr_dxy_arr[:,i])," ± ",jnp.std(corr_dxy_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("d_onsite_r")
        for i in range(L):
            print(jnp.mean(corr_d_onsite_arr[:,i])," ± ",jnp.std(corr_d_onsite_arr[:,i])/jnp.sqrt(N_blocks))
        print("\n")

        print("order_par_bond_x")
        print(jnp.mean(order_par_bond_x))
        print(jnp.std(order_par_bond_x)/jnp.sqrt(N_blocks))

        print("order_par_bond_y")
        print(jnp.mean(order_par_bond_y))
        print(jnp.std(order_par_bond_y)/jnp.sqrt(N_blocks))



