from SSH_Hubbard.lattice import *
from SSH_Hubbard.set_system import *
from SSH_Hubbard.wave_function import *
from jax import random, jit
from jax.flatten_util import ravel_pytree

############################
### Compute local energy ###
############################

from functools import partial
from jax import vmap

def local_U_energy_site_i(site: int, xloc: jnp.array):
    energy = lax.cond( (xloc[site]!=0) * (xloc[site+L]==0),  lambda : U_hub, lambda : 0.)
    return energy

local_U_energy_vmap_array = vmap(local_U_energy_site_i, in_axes=(0, None))

@jit
def local_energy_U(xloc: jnp.array): # Local energy for the Hubbard-U term
    loc_U = local_U_energy_vmap_array(jnp.arange(L),xloc)
    return loc_U.sum()


@partial(jit, static_argnums=(6,7,8,9)) # Local energy for the Phonons
def local_energy_phonons(xloc, S_z, occupied_sites, X_Phonons, Y_Phonons, params, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons):
    x_energy = jnp.power(X_Phonons,2).sum() + jnp.power(Y_Phonons,2).sum()
    lapl_log_Xphon = wf_lapl_X_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons)
    grad_log_Xphon = wf_grad_X_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons)
    lapl_log_Yphon = wf_lapl_Y_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons)
    grad_log_Yphon = wf_grad_Y_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons)
    p_energy = -jnp.sum(lapl_log_Xphon)-jnp.dot(grad_log_Xphon,grad_log_Xphon)-jnp.sum(lapl_log_Yphon)-jnp.dot(grad_log_Yphon,grad_log_Yphon)
    return 0.5*omega*(x_energy + p_energy )

@partial(jit, static_argnums=(3,)) 
def e_local_electron_i(l_index, state, params, log_amplitude_NOT_determinant):

    def compute_new_det(l_index, K_new_index, state, params, phon_difference):
        K_index = (state.occupied_sites)[l_index]
        occupied_sites_new = (state.occupied_sites).at[l_index].set(K_new_index)
        sign_new, log_det_new = determinant_fixed_U(state.U,occupied_sites_new)

        xloc_new = (state.xloc).at[K_new_index].set((state.xloc)[K_index])
        xloc_new = xloc_new.at[K_index].set(0)
        S_z_new = (state.S_z).at[K_index%L].set((state.S_z)[K_index%L]-(1.-2.*(K_index/L).astype(int)))
        S_z_new = S_z_new.at[K_new_index%L].set(S_z_new[K_new_index%L]+(1.-2.*(K_new_index/L).astype(int)))
        new_no_det_log_amplitude = log_amplitude_NOT_determinant(params, S_z_new, xloc_new, state.X_Phonons,state.Y_Phonons) 

        return t_hub*(1-(alpha*phon_difference*bond_hopping_difference_sign[K_index%L,K_new_index%L]))*(1-(K_new_index/L).astype(int)*2)*state.sign*sign_new*jnp.exp(new_no_det_log_amplitude+log_det_new-state.log_amp)

    def give_0(l_index, K_new_index, state, params, phon_difference):
        return 0.

    energy = 0.
    K_index=(state.occupied_sites)[l_index]

    phonon_difference = jnp.array([state.Y_Phonons[ivic[K_index%L,0,0]] -state.Y_Phonons[K_index%L], state.X_Phonons[ivic[K_index%L,1,0]] -state.X_Phonons[K_index%L], state.Y_Phonons[ivic[K_index%L,2,0]] -state.Y_Phonons[K_index%L], state.X_Phonons[ivic[K_index%L,3,0]] -state.X_Phonons[K_index%L] ])

    for NN_move in jnp.arange(imulti[0]):
        K_new_index=ivic[K_index%L,NN_move,0]+L*(K_index/L).astype(int)
        energy_add = lax.cond((state.xloc)[K_new_index]==0, compute_new_det, give_0, l_index, K_new_index, state, params, phonon_difference[NN_move])
        energy = energy + energy_add

    return energy

local_t_energy_vmap_arr = vmap(e_local_electron_i, in_axes=(0, None, None, None))


@partial(jit, static_argnums=(2,3)) # Local energy for the SSH-hopping term
def local_energy_t(state, params, get_U_mat, log_amplitude_NOT_determinant) -> float:
    no_det_log_amplitude = log_amplitude_NOT_determinant(params, state.S_z, state.xloc, state.X_Phonons,state.Y_Phonons) 
    U_mat = get_U_mat(params,state.X_Phonons,state.Y_Phonons)
    sign_, logdet = determinant_fixed_U(U_mat,state.occupied_sites)
    state = state.replace(U=U_mat)
    state = state.replace(log_amp=no_det_log_amplitude + logdet)
    state = state.replace(sign=sign_)
    energy_per_e = local_t_energy_vmap_arr(jnp.arange(N_e),state, params, log_amplitude_NOT_determinant)
    return energy_per_e.sum()

##################################################
### Compute non-diagonal four-body observables ###
### Namely bond order and pairing correlations ###
###################################################

# Hopping amplitude single electrons to compute bond correlations

@partial(jit, static_argnums=(4,))
def amp_electron_i_hopp_direction_v(l_index, state, params, direction_v, log_amplitude_NOT_determinant):

    def compute_new_det(l_index, K_new_index, state, params):
        K_index = (state.occupied_sites)[l_index]
        occupied_sites_new = (state.occupied_sites).at[l_index].set(K_new_index)
        sign_new, log_det_new = determinant_fixed_U(state.U,occupied_sites_new)

        xloc_new = (state.xloc).at[K_new_index].set((state.xloc)[K_index])
        xloc_new = xloc_new.at[K_index].set(0)
        S_z_new = (state.S_z).at[K_index%L].set((state.S_z)[K_index%L]-(1.-2.*(K_index/L).astype(int)))
        S_z_new = S_z_new.at[K_new_index%L].set(S_z_new[K_new_index%L]+(1.-2.*(K_new_index/L).astype(int)))
        new_no_det_log_amplitude = log_amplitude_NOT_determinant(params, S_z_new, xloc_new, state.X_Phonons,state.Y_Phonons) 

        return (1-(K_new_index/L).astype(int)*2)*state.sign*sign_new*jnp.exp(new_no_det_log_amplitude+log_det_new-state.log_amp)

    def give_0(l_index, K_new_index, state, params):
        return 0.

    amp_ratio = 0.
    K_index=state.occupied_sites[l_index]
    # Moving in direction_v at nearest neighbour!!!
    K_new_index=ivic[K_index%L,direction_v,0]+L*(K_index/L).astype(int)
    amp_ratio = lax.cond(state.xloc[K_new_index]==0, compute_new_det, give_0, l_index, K_new_index, state, params)

    return amp_ratio


vmapped_amp_electrons_hopping_direction_v = vmap(amp_electron_i_hopp_direction_v, in_axes=(0, None, None, None, None))

def give_ivic_below_NN(site):
    return ivic[site,2,0]
vmapped_ivic_below_NN = vmap(give_ivic_below_NN)

def give_ivic_left_NN(site):
    return ivic[site,3,0]
vmapped_ivic_left_NN = vmap(give_ivic_left_NN)


@partial(jit, static_argnums=(2,3))
def local_bond_order_param(state, params, get_U_mat, log_amplitude_NOT_determinant):
    no_det_log_amplitude = log_amplitude_NOT_determinant(params, state.S_z, state.xloc, state.X_Phonons,state.Y_Phonons) 
    U_mat = get_U_mat(params,state.X_Phonons,state.Y_Phonons)
    sign_, logdet = determinant_fixed_U(U_mat,state.occupied_sites)
    state = state.replace(U=U_mat)
    state = state.replace(log_amp=no_det_log_amplitude + logdet)
    state = state.replace(sign=sign_)

    amp_bonds_above_hopp = vmapped_amp_electrons_hopping_direction_v(jnp.arange(N_e), state, params, 0, log_amplitude_NOT_determinant)
    amp_bonds_right_hopp = vmapped_amp_electrons_hopping_direction_v(jnp.arange(N_e), state, params, 1, log_amplitude_NOT_determinant)
    amp_bonds_below_hopp = vmapped_amp_electrons_hopping_direction_v(jnp.arange(N_e), state, params, 2, log_amplitude_NOT_determinant)
    amp_bonds_left__hopp = vmapped_amp_electrons_hopping_direction_v(jnp.arange(N_e), state, params, 3, log_amplitude_NOT_determinant)

    bond_y_amplitudes = jnp.zeros(L)
    bond_y_amplitudes = bond_y_amplitudes.at[state.occupied_sites%L].add(amp_bonds_above_hopp)
    bond_y_amplitudes = bond_y_amplitudes.at[vmapped_ivic_below_NN(state.occupied_sites%L)].add(amp_bonds_below_hopp)

    bond_x_amplitudes = jnp.zeros(L)
    bond_x_amplitudes = bond_x_amplitudes.at[state.occupied_sites%L].add(amp_bonds_right_hopp)
    bond_x_amplitudes = bond_x_amplitudes.at[vmapped_ivic_left_NN(state.occupied_sites%L)].add(amp_bonds_left__hopp)

    return bond_x_amplitudes, bond_y_amplitudes 
        

# Hopping amplitude of two electrons to compute pairing correlations

@jit
def _2D_positions(i,j):
  i_x = i // Ly
  i_y = i % Ly
  j_x = j // Ly
  j_y = j % Ly

  r_I_x  = (i_x - j_x + Lx) % Lx
  r_II_x = (j_x - i_x + Lx) % Lx
  r_I_y  = (i_y - j_y + Ly) % Ly
  r_II_y = (j_y - i_y + Ly) % Ly

  return jnp.array([r_I_y + r_I_x * Ly, r_II_y + r_II_x * Ly])

@partial(jit, static_argnums=(2,))
def amp_two_electrons_jumping_right_right_and_left_left_fixed_l_index(l_index, val, log_amplitude_NOT_determinant):

    m_index, state, params, delta_amplitudes = val
    
    def compute_new_det(l_index, m_index, K_new_index, I_new_index, state, params):
        K_index = (state.occupied_sites)[l_index]
        I_index = (state.occupied_sites)[m_index]
        occupied_sites_new = (state.occupied_sites).at[l_index].set(K_new_index)
        occupied_sites_new = occupied_sites_new.at[m_index].set(I_new_index)
        sign_new, log_det_new = determinant_fixed_U(state.U,occupied_sites_new)

        xloc_new = (state.xloc).at[K_new_index].set((state.xloc)[K_index])
        xloc_new = xloc_new.at[K_index].set(0)
        xloc_new = xloc_new.at[I_new_index].set(xloc_new[I_index])
        xloc_new = xloc_new.at[I_index].set(0)
        S_z_new = (state.S_z).at[K_index%L].set((state.S_z)[K_index%L]+1.)
        S_z_new = S_z_new.at[K_new_index%L].set(S_z_new[K_new_index%L]+1.)
        S_z_new = S_z_new.at[I_index%L].set(S_z_new[I_index%L]-1.)
        S_z_new = S_z_new.at[I_new_index%L].set(S_z_new[I_new_index%L]-1.) 
        new_no_det_log_amplitude = log_amplitude_NOT_determinant(params, S_z_new, xloc_new, state.X_Phonons,state.Y_Phonons) 

        return state.sign*sign_new*jnp.exp(new_no_det_log_amplitude+log_det_new-state.log_amp)

    def give_0(l_index, m_index, K_new_index, I_new_index, state, params):
        return 0.

    K_index=state.occupied_sites[l_index]
    I_index=state.occupied_sites[m_index]

    # Singlets along y direction and then x direction
    for direction in range(2):
        K_new_index=ivic[K_index%L,2+direction,0]+L*( 1 - (K_index/L).astype(int) )
        I_new_index=ivic[I_index%L,2+direction,0]+L*( 1 - (I_index/L).astype(int) )
        amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
        delta_amplitudes = delta_amplitudes.at[ direction, _2D_positions(K_index%L,I_index%L) ].add(  amp_ratio  )

        K_new_index=ivic[K_index%L,0+direction,0]+L*( 1 - (K_index/L).astype(int) )
        I_new_index=ivic[I_index%L,0+direction,0]+L*( 1 - (I_index/L).astype(int) )
        amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
        delta_amplitudes = delta_amplitudes.at[ direction, _2D_positions(K_new_index%L,I_new_index%L) ].add(  amp_ratio  )

        K_new_index=ivic[K_index%L,2+direction,0]+L*( 1 - (K_index/L).astype(int) )
        I_new_index=ivic[I_index%L,0+direction,0]+L*( 1 - (I_index/L).astype(int) )
        amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
        delta_amplitudes = delta_amplitudes.at[ direction, _2D_positions(K_index%L,I_new_index%L) ].add(  amp_ratio  )

        K_new_index=ivic[K_index%L,0+direction,0]+L*( 1 - (K_index/L).astype(int) )
        I_new_index=ivic[I_index%L,2+direction,0]+L*( 1 - (I_index/L).astype(int) )
        amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
        delta_amplitudes = delta_amplitudes.at[ direction, _2D_positions(K_new_index%L,I_index%L) ].add(  amp_ratio  )


    # Singlets along x and y direction simultaneouslyy 
    K_new_index=ivic[K_index%L,2,0]+L*( 1 - (K_index/L).astype(int) )
    I_new_index=ivic[I_index%L,3,0]+L*( 1 - (I_index/L).astype(int) )
    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
    delta_amplitudes = delta_amplitudes.at[ 2, _2D_positions(K_index%L,I_index%L)[0] ].add(  amp_ratio  )

#    K_new_index=ivic[K_index%L,3,0]+L*( 1 - (K_index/L).astype(int) )
#    I_new_index=ivic[I_index%L,2,0]+L*( 1 - (I_index/L).astype(int) )
#    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
#    delta_amplitudes = delta_amplitudes.at[ 2, _2D_positions(K_index%L,I_index%L) ].add(  amp_ratio  )

    K_new_index=ivic[K_index%L,0,0]+L*( 1 - (K_index/L).astype(int) )
    I_new_index=ivic[I_index%L,1,0]+L*( 1 - (I_index/L).astype(int) )
    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
    delta_amplitudes = delta_amplitudes.at[ 2, _2D_positions(K_new_index%L,I_new_index%L)[0] ].add(  amp_ratio  )

#    K_new_index=ivic[K_index%L,1,0]+L*( 1 - (K_index/L).astype(int) )
#    I_new_index=ivic[I_index%L,0,0]+L*( 1 - (I_index/L).astype(int) )
#    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
#    delta_amplitudes = delta_amplitudes.at[ 2, _2D_positions(K_new_index%L,I_new_index%L) ].add(  amp_ratio  )

    K_new_index=ivic[K_index%L,2,0]+L*( 1 - (K_index/L).astype(int) )
    I_new_index=ivic[I_index%L,1,0]+L*( 1 - (I_index/L).astype(int) )
    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
    delta_amplitudes = delta_amplitudes.at[ 2, _2D_positions(K_index%L,I_new_index%L)[0] ].add(  amp_ratio  )

#    K_new_index=ivic[K_index%L,3,0]+L*( 1 - (K_index/L).astype(int) )
#    I_new_index=ivic[I_index%L,0,0]+L*( 1 - (I_index/L).astype(int) )
#    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
#    delta_amplitudes = delta_amplitudes.at[ 2, _2D_positions(K_index%L,I_new_index%L) ].add(  amp_ratio  )

    K_new_index=ivic[K_index%L,0,0]+L*( 1 - (K_index/L).astype(int) )
    I_new_index=ivic[I_index%L,3,0]+L*( 1 - (I_index/L).astype(int) )
    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
    delta_amplitudes = delta_amplitudes.at[ 2, _2D_positions(K_new_index%L,I_index%L)[0] ].add(  amp_ratio  )

#    K_new_index=ivic[K_index%L,1,0]+L*( 1 - (K_index/L).astype(int) )
#    I_new_index=ivic[I_index%L,2,0]+L*( 1 - (I_index/L).astype(int) )
#    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0) + (state.xloc[K_new_index]==0)*(I_new_index==K_index), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
#    delta_amplitudes = delta_amplitudes.at[ 2, _2D_positions(K_new_index%L,I_index%L) ].add(  amp_ratio  )

    # Onsite pairing correlations

    K_new_index = K_index%L + L*( 1 - (K_index/L).astype(int) )
    I_new_index = I_index%L + L*( 1 - (I_index/L).astype(int) )
    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0), compute_new_det, give_0, l_index, m_index, K_new_index, I_new_index, state, params)
    delta_amplitudes = delta_amplitudes.at[ 3, _2D_positions(K_index%L,I_index%L) ].add(  amp_ratio  )


    return m_index, state, params, delta_amplitudes

@partial(jit, static_argnums=(2,))
def amp_two_electrons_jumping_right_right_and_left_left_fixed_m_index(m_index, val, log_amplitude_NOT_determinant):

    state, params, delta_amplitudes = val

    val_I = m_index, state, params, delta_amplitudes

    val_I = lax.fori_loop(N_e_up,N_e,partial(amp_two_electrons_jumping_right_right_and_left_left_fixed_l_index, log_amplitude_NOT_determinant=log_amplitude_NOT_determinant), val_I)

    m_index, state, params, delta_amplitudes = val_I

    return state, params, delta_amplitudes

@partial(jit, static_argnums=(3,))
def amp_two_electrons_jumping_right_right_and_left_left(state, params, delta_amplitudes, log_amplitude_NOT_determinant):

    state, params, delta_amplitudes = lax.fori_loop(0,N_e_up,partial(amp_two_electrons_jumping_right_right_and_left_left_fixed_m_index,log_amplitude_NOT_determinant=log_amplitude_NOT_determinant),( state, params, delta_amplitudes ) )

    return state, params, delta_amplitudes


@partial(jit, static_argnums=(2,))
def amp_one_electron_jumping_fixed_l_index(l_index, val, log_amplitude_NOT_determinant):

    state, params, delta_amplitudes = val

    def compute_new_det(l_index, K_new_index, I_new_index, state, params):
        K_index = (state.occupied_sites)[l_index]
        occupied_sites_new = (state.occupied_sites).at[l_index].set(I_new_index)
        sign_new, log_det_new = determinant_fixed_U(state.U,occupied_sites_new)

        xloc_new = (state.xloc).at[I_new_index].set((state.xloc)[K_index])
        xloc_new = xloc_new.at[K_index].set(0)
        S_z_new = (state.S_z).at[K_index%L].set((state.S_z)[K_index%L]+1.)
        S_z_new = S_z_new.at[I_new_index%L].set(S_z_new[I_new_index%L]-1.)
        new_no_det_log_amplitude = log_amplitude_NOT_determinant(params, S_z_new, xloc_new, state.X_Phonons,state.Y_Phonons) 

        return state.sign*sign_new*jnp.exp(new_no_det_log_amplitude+log_det_new-state.log_amp)

    def give_0(l_index, K_new_index, I_new_index, state, params):
        return 0.
    
    def give_1(l_index, K_new_index, I_new_index, state, params):
        return 1.
    
    # CASE K_new_index == I_index and no electron is in I_index
    K_index=state.occupied_sites[l_index]

    for direction in range(2):
        K_new_index=ivic[K_index%L,2+direction,0]+L*( 1 - (K_index/L).astype(int) )
        I_index = K_new_index
        I_new_index=ivic[I_index%L,2+direction,0]+L*( 1 - (I_index/L).astype(int) )
        amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0), compute_new_det, give_0, l_index, K_new_index, I_new_index, state, params)
        delta_amplitudes= delta_amplitudes.at[ direction, _2D_positions(K_new_index%L,I_new_index%L)  ].add(  amp_ratio  )
        delta_amplitudes= delta_amplitudes.at[ direction, 0 ].add( lax.cond( state.xloc[K_new_index]==0, give_1, give_0, l_index, K_new_index, I_new_index, state, params)  )


        K_new_index=ivic[K_index%L,0+direction,0]+L*( 1 - (K_index/L).astype(int) )
        I_index = K_new_index
        I_new_index=ivic[I_index%L,0+direction,0]+L*( 1 - (I_index/L).astype(int) )
        amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0), compute_new_det, give_0, l_index, K_new_index, I_new_index, state, params)
        delta_amplitudes= delta_amplitudes.at[ direction, _2D_positions(K_index%L,I_index%L)  ].add(  amp_ratio  )
        delta_amplitudes= delta_amplitudes.at[ direction, 0 ].add( lax.cond( state.xloc[K_new_index]==0, give_1, give_0, l_index, K_new_index, I_new_index, state, params)  )


    K_new_index=ivic[K_index%L,2,0]+L*( 1 - (K_index/L).astype(int) )
    I_index = K_new_index
    I_new_index=ivic[I_index%L,3,0]+L*( 1 - (I_index/L).astype(int) )
    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0), compute_new_det, give_0, l_index, K_new_index, I_new_index, state, params)
    delta_amplitudes= delta_amplitudes.at[ 2, _2D_positions(K_new_index%L,I_new_index%L)[0]  ].add(  amp_ratio  )
    I_new_index=ivic[I_index%L,1,0]+L*( 1 - (I_index/L).astype(int) )
    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0), compute_new_det, give_0, l_index, K_new_index, I_new_index, state, params)
    delta_amplitudes= delta_amplitudes.at[ 2, _2D_positions(K_new_index%L,I_index%L)[0]  ].add(  amp_ratio  )

    K_new_index=ivic[K_index%L,0,0]+L*( 1 - (K_index/L).astype(int) )
    I_index = K_new_index
    I_new_index=ivic[I_index%L,1,0]+L*( 1 - (I_index/L).astype(int) )
    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0), compute_new_det, give_0, l_index, K_new_index, I_new_index, state, params)
    delta_amplitudes= delta_amplitudes.at[ 2, _2D_positions(K_index%L,I_index%L)[0]  ].add(  amp_ratio  )
    I_new_index=ivic[I_index%L,3,0]+L*( 1 - (I_index/L).astype(int) )
    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0), compute_new_det, give_0, l_index, K_new_index, I_new_index, state, params)
    delta_amplitudes= delta_amplitudes.at[ 2, _2D_positions(K_index%L,I_new_index%L)[0]  ].add(  amp_ratio  )
        

#    K_new_index=ivic[K_index%L,3,0]+L*( 1 - (K_index/L).astype(int) )
#    I_index = K_new_index
#    I_new_index=ivic[I_index%L,2,0]+L*( 1 - (I_index/L).astype(int) )
#    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0), compute_new_det, give_0, l_index, K_new_index, I_new_index, state, params)
#    delta_amplitudes= delta_amplitudes.at[ 2, _2D_positions(K_new_index%L,I_new_index%L)  ].add(  amp_ratio  )
#    I_new_index=ivic[I_index%L,0,0]+L*( 1 - (I_index/L).astype(int) )
#    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0), compute_new_det, give_0, l_index, K_new_index, I_new_index, state, params)
#    delta_amplitudes= delta_amplitudes.at[ 2, _2D_positions(K_new_index%L,I_index%L)  ].add(  amp_ratio  )

#    K_new_index=ivic[K_index%L,1,0]+L*( 1 - (K_index/L).astype(int) )
#    I_index = K_new_index
#    I_new_index=ivic[I_index%L,0,0]+L*( 1 - (I_index/L).astype(int) )
#    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0), compute_new_det, give_0, l_index, K_new_index, I_new_index, state, params)
#    delta_amplitudes= delta_amplitudes.at[ 2, _2D_positions(K_index%L,I_index%L)  ].add(  amp_ratio  )
#    I_new_index=ivic[I_index%L,2,0]+L*( 1 - (I_index/L).astype(int) )
#    amp_ratio = lax.cond( (state.xloc[K_new_index]==0)*(state.xloc[I_new_index]==0), compute_new_det, give_0, l_index, K_new_index, I_new_index, state, params)
#    delta_amplitudes= delta_amplitudes.at[ 2, _2D_positions(K_index%L,I_new_index%L)  ].add(  amp_ratio  )

    K_new_index = K_index%L
    amp_ratio = lax.cond( state.xloc[K_new_index]==0, give_1, give_0, l_index, K_new_index, K_index, state, params)
    delta_amplitudes= delta_amplitudes.at[ 3, 0  ].add(  amp_ratio  )
        
    return state, params, delta_amplitudes


@partial(jit, static_argnums=(3,))
def amp_one_electron_jumping(state, params, delta_amplitudes, log_amplitude_NOT_determinant):

    state, params, delta_amplitudes = lax.fori_loop(N_e_up,N_e,partial(amp_one_electron_jumping_fixed_l_index,log_amplitude_NOT_determinant=log_amplitude_NOT_determinant),( state, params, delta_amplitudes ) )

    return state, params, delta_amplitudes


@partial(jit, static_argnums=(2,3))
def local_delta_order_param(state, params, get_U_mat, log_amplitude_NOT_determinant):
    no_det_log_amplitude= log_amplitude_NOT_determinant(params, state.S_z, state.xloc, state.X_Phonons, state.Y_Phonons) 
    U_mat = get_U_mat(params,state.X_Phonons,state.Y_Phonons)
    sign_, logdet = determinant_fixed_U(U_mat,state.occupied_sites)
    state = state.replace(U=U_mat)
    state = state.replace(log_amp=no_det_log_amplitude + logdet)
    state = state.replace(sign=sign_)

    delta_amplitudes = jnp.zeros((4,L))

    state, params, delta_amplitudes = amp_two_electrons_jumping_right_right_and_left_left(state, params, delta_amplitudes, log_amplitude_NOT_determinant)
    state, params, delta_amplitudes = amp_one_electron_jumping(state, params, delta_amplitudes, log_amplitude_NOT_determinant)

    delta_amplitudes /= two_L
    delta_amplitudes = delta_amplitudes.at[2,:].set(delta_amplitudes[2,:]*2.)

    return delta_amplitudes

#############################
### Correlation functions ###
### and Fourier transform ###
#############################

# Compute the correlation function looping over translations
@jit
def corr_2D(vec_n):

    def translation_ax1(translated_vec, _):
        return jnp.roll(translated_vec,1, axis=1), jnp.dot( jnp.roll(translated_vec,1, axis=1).flatten(), vec_n.flatten() )

    def translation_both_axes( vec, _  ):
        appo, corr_unrolled = lax.scan(translation_ax1,jnp.roll(vec,-1, axis=1),jnp.zeros(Ly))
        appo = jnp.roll(appo,1, axis=1)
        return jnp.roll(appo,1, axis=0), corr_unrolled

    appo, corr_unrolled = lax.scan(translation_both_axes,vec_n.reshape(Lx,Ly),jnp.zeros(Lx))

    return corr_unrolled.flatten()/L



def r_times_k(r_vec, kx, ky):
  return (r_vec%Ly)*ky + (r_vec//Ly)*kx

vmapped_r_times_k = vmap(r_times_k, in_axes=(0, None, None))

@jit
def Fourier_corr_2D(corr): # Compute the Fourier transform of the correlation function

  def Fourier_one_component(vec, k):
    return vec, jnp.real(jnp.dot(vec.flatten(),jnp.exp(1j*vmapped_r_times_k(jnp.arange(L),k[0],k[1]))))

  kx_arr = (2.*jnp.pi/Lx)*jnp.repeat(jnp.arange(Lx),Ly)
  ky_arr = (2.*jnp.pi/Ly)*jnp.tile(jnp.arange(Ly),Lx)
  k_arr = jnp.stack((kx_arr,ky_arr)).transpose()

  _, Fourier_vec = lax.scan(Fourier_one_component,corr,k_arr)

  return Fourier_vec


########################################
### Functions to measure observables ###
########################################

from jax.flatten_util import ravel_pytree
import jax.tree_util


# Used to compute the derivative of the wavefunction
# and then update the parameters by SR
@partial(jit, static_argnums=(2,3,4,5,6,7,8))
def local_obs_SR(state, params, get_U_mat, log_amplitude_NOT_determinant, grad_wf, grad_Xphonons_wf, lapl_Xphonons_wf, grad_Yphonons_wf, lapl_Yphonons_wf):
    # local energy per spin
    e_L = local_energy_t(state, params, get_U_mat, log_amplitude_NOT_determinant) + local_energy_U(state.xloc)+ local_energy_phonons(state.xloc, state.S_z,  state.occupied_sites, state.X_Phonons, state.Y_Phonons, params, grad_Xphonons_wf, lapl_Xphonons_wf, grad_Yphonons_wf, lapl_Yphonons_wf)
    # local operators for the estimation of the gradient of the energy
    O_L = grad_wf(params, state.occupied_sites, state.xloc, state.S_z,  state.X_Phonons, state.Y_Phonons)
    O_flat, _ = ravel_pytree(O_L)
    eO_L = e_L * O_flat
    OO_L = jnp.outer(jnp.conj(O_flat), O_flat)
    return {'e': e_L, 'O' : O_flat, 'eO' : eO_L, 'OO' : OO_L}

# Used to compute the observables at fixed parameters
@partial(jit, static_argnums=(2,3,4,5,6,7,8))
def local_obs(state, params, get_U_mat, log_amplitude_NOT_determinant, grad_wf, grad_Xphonons_wf, lapl_Xphonons_wf, grad_Yphonons_wf, lapl_Yphonons_wf):
    # local energy per spin
    e_L = local_energy_t(state, params, get_U_mat, log_amplitude_NOT_determinant) + local_energy_U(state.xloc)+ local_energy_phonons(state.xloc, state.S_z,  state.occupied_sites, state.X_Phonons, state.Y_Phonons, params, grad_Xphonons_wf, lapl_Xphonons_wf, grad_Yphonons_wf, lapl_Yphonons_wf)

    appo_x_first = (state.xloc).astype(jnp.bool_)
    appo_x_second = appo_x_first.astype(jnp.int32)

    corr_n = corr_2D(appo_x_second[:L] + ( - appo_x_second[L:2*L] + 1. ) )
    N_q = Fourier_corr_2D(corr_n)
    corr_s = corr_2D(appo_x_second[:L] - ( - appo_x_second[L:2*L] + 1. ) )
    S_q = Fourier_corr_2D(corr_s)

    bond_x_amplitudes, bond_y_amplitudes = local_bond_order_param(state, params, get_U_mat, log_amplitude_NOT_determinant)
    x_order_parameter_bonds = jnp.sum(bond_x_amplitudes*stagg_x)
    y_order_parameter_bonds = jnp.sum(bond_y_amplitudes*stagg_y)
    corr_bonds_x = corr_2D(bond_x_amplitudes)
    Bx_q = Fourier_corr_2D(corr_bonds_x)
    corr_bonds_y = corr_2D(bond_y_amplitudes)
    By_q = Fourier_corr_2D(corr_bonds_y)
    X_diff = state.X_Phonons - state.X_Phonons[vmapped_ivic_left_NN(jnp.arange(L))]
    Y_diff = state.Y_Phonons - state.Y_Phonons[vmapped_ivic_below_NN(jnp.arange(L))]

    delta_corr = local_delta_order_param(state, params, get_U_mat, log_amplitude_NOT_determinant)

    return {'e': e_L, 'corr_n' : corr_n, 'N_q' : N_q, 'corr_s' : corr_s, 'S_q' : S_q, 'corr_bx' : corr_bonds_x, 'Bx_q': Bx_q, 'corr_by' : corr_bonds_y, 'By_q': By_q, 'order_bonds_x': x_order_parameter_bonds, 'order_bonds_y': y_order_parameter_bonds, 'bond_amplitudes_x': bond_x_amplitudes, 'bond_amplitudes_y': bond_y_amplitudes, 'X_diff': X_diff, 'Y_diff': Y_diff, 'corr_dx' : delta_corr[1,:], 'corr_dy' : delta_corr[0,:], 'corr_dxy' : delta_corr[2,:], 'corr_d_onsite' : delta_corr[3,:]}


