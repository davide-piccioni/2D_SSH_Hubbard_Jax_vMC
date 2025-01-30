from lattice import *
from set_system import *
from wave_function import *
from observables import *
from jax import random, jit
from jax.flatten_util import ravel_pytree


@struct.dataclass
class acceptance():
    acc_call_hop_NN: int
    tot_call_hop_NN: int
    acc_call_spin_flip: int
    tot_call_spin_flip: int
    acc_call_phonon_step: int
    tot_call_phonon_step: int


@partial(jit, static_argnums=(2,))
def single_hop_NN_step(i, vals, log_amplitude_NOT_determinant):
    state, params, a, rands = vals

    a = a.replace(tot_call_hop_NN=a.tot_call_hop_NN+1)
    l_index = (rands[i, 0] * N_e).astype(int)
    where_to_move = (rands[i, 1] * imulti[0]).astype(int)
    K_index= state.occupied_sites[l_index]
    K_new_index=ivic[K_index%L,where_to_move,0]+L*((K_index/L).astype(int))

    def metropolis_func(vals, log_amplitude_NOT_determinant):
        state, a, l_index, K_index, K_new_index = vals
        occupied_sites_new = (state.occupied_sites).at[l_index].set(K_new_index)
        xloc_new = (state.xloc).at[K_new_index].set((state.xloc)[K_index])
        xloc_new = xloc_new.at[K_index].set(0)
        S_z_new = (state.S_z).at[K_index%L].set((state.S_z)[K_index%L]-(1.-2.*(K_index/L).astype(int)))
        S_z_new = S_z_new.at[K_new_index%L].set(S_z_new[K_new_index%L]+(1.-2.*(K_new_index/L).astype(int)))
        sign, log_det_new = determinant_fixed_U(state.U,occupied_sites_new)
        log_amp_new = log_amplitude_NOT_determinant(params, S_z_new, xloc_new, state.X_Phonons,state.Y_Phonons)+log_det_new 

        def accepted_func(vals):
            state, occupied_sites_new, xloc_new, S_z_new, new_log_amp, a = vals
            state = state.replace(occupied_sites=occupied_sites_new)
            state = state.replace(xloc=xloc_new)
            state = state.replace(S_z=S_z_new)
            state = state.replace(log_amp=new_log_amp)
            a = a.replace(acc_call_hop_NN=a.acc_call_hop_NN+1)
            return state, a

        def refused_func(vals):
            state, occupied_sites_new, xloc_new, S_z_new, new_log_amp, a = vals
            return state, a

        state, a = lax.cond( jnp.log(1.-rands[i, 2]) < 2.*(log_amp_new-state.log_amp), accepted_func, refused_func, [state, occupied_sites_new, xloc_new, S_z_new, log_amp_new, a,])

        return state, a

    def no_metropolis_func(vals):
        state, a, l_index, K_index, K_new_index = vals
        return state, a

    state, a = lax.cond( (state.xloc)[K_new_index]==0 , partial(metropolis_func, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant), no_metropolis_func, (state, a, l_index, K_index, K_new_index))
    
    return [state, params, a, rands]

@partial(jit, static_argnums=(2,))
def single_spin_flip_step(i, vals, log_amplitude_NOT_determinant):
    state, params, a, rands = vals

    a = a.replace(tot_call_spin_flip=a.tot_call_spin_flip+1)

    l_index = (rands[i, 0] * N_e).astype(int)
    where_to_move = (rands[i, 1] * imulti[0]).astype(int)
    K_index=(state.occupied_sites)[l_index]
    K_new_index=ivic[K_index%L,where_to_move,0]+L*((K_index/L).astype(int))

    I_index=K_index%L+L*(1-(K_index/L).astype(int))
    I_new_index=K_new_index%L+L*(1-(K_new_index/L).astype(int))
    m_index = ((state.xloc)[I_index]).astype(int)-1

    def metropolis_func(vals, log_amplitude_NOT_determinant):
        state, a = vals
        occupied_sites_new = (state.occupied_sites).at[l_index].set(K_new_index)
        occupied_sites_new = occupied_sites_new.at[m_index].set(I_new_index)
        xloc_new = (state.xloc).at[K_new_index].set((state.xloc)[K_index])
        xloc_new = xloc_new.at[K_index].set(0)
        xloc_new = xloc_new.at[I_new_index].set(xloc_new[I_index])
        xloc_new = xloc_new.at[I_index].set(0)
        sign, log_det_new = determinant_fixed_U(state.U,occupied_sites_new)
        log_amp_new = log_amplitude_NOT_determinant(params, state.S_z, xloc_new, state.X_Phonons,state.Y_Phonons)+log_det_new 

        def accepted_func(vals):
            state, occupied_sites_new, xloc_new, new_log_amp, a = vals
            state = state.replace(occupied_sites=occupied_sites_new)
            state = state.replace(xloc=xloc_new)
            state = state.replace(log_amp=new_log_amp)
            a = a.replace(acc_call_spin_flip=a.acc_call_spin_flip+1)
            return state, a

        def refused_func(vals):
            state, occupied_sites_new, xloc_new, new_log_amp, a = vals
            return state, a

        state, a = lax.cond( jnp.log(1.-rands[i, 2]) < 2.*(log_amp_new-state.log_amp), accepted_func, refused_func, [state, occupied_sites_new, xloc_new, log_amp_new, a])

        return state, a

    def no_metropolis_func(vals):
        state, a = vals
        return state, a

    state, a = lax.cond( ((state.xloc)[K_new_index]==0)*((state.xloc)[I_new_index]==0)*(m_index>-1) , partial(metropolis_func, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant), no_metropolis_func, (state, a))

    return [state, params, a, rands]


@partial(jit, static_argnums=(2,3))
def single_phonon_step(i, vals, log_amplitude_NOT_determinant, get_U_mat):
    state, params, a, rands = vals

    a = a.replace(tot_call_phonon_step=a.tot_call_phonon_step+1)
    site_index = (rands[i, 0] * two_L).astype(int)
    displacement_x = jnp.zeros(L)
    displacement_x = displacement_x.at[site_index%L].set((rands[i, 1]-0.5)*displ_phon_move*(1-(site_index/L).astype(int)))
    displacement_y = jnp.zeros(L)
    displacement_y = displacement_y.at[site_index%L].set((rands[i, 1]-0.5)*displ_phon_move*((site_index/L).astype(int)))

    def metropolis_func(vals, log_amplitude_NOT_determinant):
        state, a, site_index, displacement_x, displacement_y = vals
        U_mat_new = get_U_mat(params, state.X_Phonons + displacement_x, state.Y_Phonons + displacement_y)
        sign, log_det_new = determinant_fixed_U(U_mat_new, state.occupied_sites)
        log_amp_new = log_amplitude_NOT_determinant(params, state.S_z, state.xloc, state.X_Phonons + displacement_x,state.Y_Phonons + displacement_y) + log_det_new 

        def accepted_func(vals):
            state, new_log_amp, new_U_mat, a, site_index, displacement_x, displacement_y = vals
            state = state.replace(X_Phonons=state.X_Phonons.at[:].add(displacement_x))
            state = state.replace(Y_Phonons=state.Y_Phonons.at[:].add(displacement_y))
            state = state.replace(U=new_U_mat)
            state = state.replace(log_amp=new_log_amp)
            a = a.replace(acc_call_phonon_step=a.acc_call_phonon_step+1)
            return state, a

        def refused_func(vals):
            state, new_log_amp, new_U_mat, a, site_index, displacement_x, displacement_y = vals
            return state, a

        state, a = lax.cond( jnp.log(1.-rands[i, 2]) < 2.*(log_amp_new-state.log_amp) , accepted_func, refused_func, [state, log_amp_new, U_mat_new, a, site_index, displacement_x, displacement_y])

        return state, a

    state, a = partial(metropolis_func, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant)((state, a, site_index, displacement_x, displacement_y))

    return [state, params, a, rands]


@partial(jit, static_argnums=(2,3))
def single_Metropolis_step(i, vals, log_amplitude_NOT_determinant, get_U_mat):
    state, params, a, rands = vals

    def electron_move(n, vals):
        return lax.cond( rands[n, 3] < p_spin_flip, partial(single_spin_flip_step, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant),  partial(single_hop_NN_step, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant), n, vals)

    vals = lax.cond( rands[3*i, 3] < p_moving_electrons, electron_move, partial(single_phonon_step, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant, get_U_mat=get_U_mat), 3*i, vals)
    vals = lax.cond( rands[3*i+1, 3] < p_moving_electrons, electron_move, partial(single_phonon_step, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant, get_U_mat=get_U_mat), 3*i+1, vals)
    vals = lax.cond( rands[3*i+2, 3] < p_moving_electrons, electron_move, partial(single_phonon_step, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant, get_U_mat=get_U_mat), 3*i+2, vals)

    return vals


@partial(jit, static_argnums=(2,3,4,5,6,7,8))
def sweep(i, var, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons):
    state, params, key, a, mean_values = var

    # Generate all the random numbers
    key, subkey = random.split(key, num=2)
    rand = random.uniform(subkey, (3*sparse_ave_length, 4))

    state, params, a, rand = lax.fori_loop(0, sparse_ave_length, partial(single_Metropolis_step, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant, get_U_mat=get_U_mat), [state, params, a, rand])

    mean_values = jax.tree_util.tree_map(lambda x, y : x + y, mean_values, local_obs(state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons))

    return [state, params, key, a, mean_values]


@partial(jit, static_argnums=(2,3,4,5,6,7,8))
def sweep_SR(i, var, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons):
    state, params, key, a, mean_values = var

    # Generate all the random numbers
    key, subkey = random.split(key, num=2)
    rand = random.uniform(subkey, (3*sparse_ave_length, 4))

    state, params, a, rand = lax.fori_loop(0, sparse_ave_length, partial(single_Metropolis_step, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant, get_U_mat=get_U_mat), [state, params, a, rand])

    mean_values = jax.tree_util.tree_map(lambda x, y : x + y, mean_values, local_obs_SR(state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons))

    return [state, params, key, a, mean_values]
        

##################################
### Stochastic Reconfiguration ###
##################################

@partial(jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))
def mc_sweeps_SR(nsweeps, state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a):

    # inizialize mean_values
    mean_values = local_obs_SR(state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons)
    mean_values = jax.tree_util.tree_map(lambda x: x*0, mean_values)

    state, params, key, a, mean_values = lax.fori_loop(0, nsweeps, partial(sweep_SR, get_U_mat=get_U_mat, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant, wf_grad=wf_grad, wf_grad_X_phonons=wf_grad_X_phonons, wf_lapl_X_phonons=wf_lapl_X_phonons, wf_grad_Y_phonons=wf_grad_Y_phonons, wf_lapl_Y_phonons=wf_lapl_Y_phonons), [state, params, key, a, mean_values])

    mean_values = jax.tree_util.tree_map(lambda x: x/nsweeps, mean_values)

    return state, key, a, mean_values

@partial(jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))
def mc_sweeps_Thermaliz(nsweeps, state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a):

    mean_values = local_obs_SR(state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons)
    state, params, key, a, mean_values = lax.fori_loop(0, nsweeps, partial(sweep_SR, get_U_mat=get_U_mat, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant, wf_grad=wf_grad, wf_grad_X_phonons=wf_grad_X_phonons, wf_lapl_X_phonons=wf_lapl_X_phonons, wf_grad_Y_phonons=wf_grad_Y_phonons, wf_lapl_Y_phonons=wf_lapl_Y_phonons), [state, params, key, a, mean_values])

    return state, key, a


#################
### Measuring ###
#################
             
@partial(jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))
def mc_sweeps_Measure(nsweeps, state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a):

    # inizialize mean_values
    mean_values = local_obs(state, params, get_U_mat, log_amplitude_NOT_determinant, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons)
    mean_values = jax.tree_util.tree_map(lambda x: x*0, mean_values)

    state, params, key, a, mean_values = lax.fori_loop(0, nsweeps, partial(sweep, get_U_mat=get_U_mat, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant, wf_grad=wf_grad, wf_grad_X_phonons=wf_grad_X_phonons, wf_lapl_X_phonons=wf_lapl_X_phonons, wf_grad_Y_phonons=wf_grad_Y_phonons, wf_lapl_Y_phonons=wf_lapl_Y_phonons), [state, params, key, a, mean_values])

    mean_values = jax.tree_util.tree_map(lambda x: x/nsweeps, mean_values)

    return state, key, a, mean_values

