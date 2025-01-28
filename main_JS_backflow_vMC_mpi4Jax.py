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

from lattice import *
from set_system import *
from wave_function import *
from jax import random, jit, grad, hessian
from jax.flatten_util import ravel_pytree
import time

########################################
########################################
### Functions to compute observables ###
########################################
########################################


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
def local_energy_U(xloc: jnp.array):
    loc_U = local_U_energy_vmap_array(jnp.arange(L),xloc)
    return loc_U.sum()


@partial(jit, static_argnums=(6,7,8,9))
def local_energy_phonons(xloc, S_z, occupied_sites, X_Phonons, Y_Phonons, params, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons):
    x_energy = jnp.power(X_Phonons,2).sum() + jnp.power(Y_Phonons,2).sum()
    lapl_log_Xphon = wf_lapl_X_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons)
    grad_log_Xphon = wf_grad_X_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons)
    lapl_log_Yphon = wf_lapl_Y_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons)
    grad_log_Yphon = wf_grad_Y_phonons(params,occupied_sites,xloc,S_z,X_Phonons,Y_Phonons)
    p_energy = -jnp.sum(lapl_log_Xphon)-jnp.dot(grad_log_Xphon,grad_log_Xphon)-jnp.sum(lapl_log_Yphon)-jnp.dot(grad_log_Yphon,grad_log_Yphon)
    return 0.5*omega*(x_energy + p_energy )

@jit
def e_local_electron_i(l_index, state, params ):

    def compute_new_det(l_index, K_new_index, state, params, phon_difference):
        K_index = (state.occupied_sites)[l_index]
        occupied_sites_new = (state.occupied_sites).at[l_index].set(K_new_index)
        sign_new, log_det_new = determinant_fixed_U(state.U,occupied_sites_new)

        xloc_new = (state.xloc).at[K_new_index].set((state.xloc)[K_index])
        xloc_new = xloc_new.at[K_index].set(0)
        S_z_new = (state.S_z).at[K_index%L].set((state.S_z)[K_index%L]-(1.-2.*(K_index/L).astype(int)))
        S_z_new = S_z_new.at[K_new_index%L].set(S_z_new[K_new_index%L]+(1.-2.*(K_new_index/L).astype(int)))
        new_no_det_log_amplitude = log_jastrow_amplitude(params,S_z_new,xloc_new) + log_ph_amplitude(params,state.X_Phonons,state.Y_Phonons,xloc_new)

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

local_t_energy_vmap_arr = vmap(e_local_electron_i, in_axes=(0, None, None))


@jit
def local_energy_t(state, params) -> float:
    no_det_log_amplitude = log_jastrow_amplitude(params,state.S_z,state.xloc) + log_ph_amplitude(params,state.X_Phonons,state.Y_Phonons,state.xloc)
    U_mat = get_U_mat(params,state.X_Phonons,state.Y_Phonons)
    sign_, logdet = determinant_fixed_U(U_mat,state.occupied_sites)
    state = state.replace(U=U_mat)
    state = state.replace(log_amp=no_det_log_amplitude + logdet)
    state = state.replace(sign=sign_)
    energy_per_e = local_t_energy_vmap_arr(jnp.arange(N_e),state, params )
    return energy_per_e.sum()


# Hopping amplitude single electrons to compute bond correlations

@jit
def amp_electron_i_hopp_direction_v(l_index, state, params, direction_v):

    def compute_new_det(l_index, K_new_index, state, params):
        K_index = (state.occupied_sites)[l_index]
        occupied_sites_new = (state.occupied_sites).at[l_index].set(K_new_index)
        sign_new, log_det_new = determinant_fixed_U(state.U,occupied_sites_new)

        xloc_new = (state.xloc).at[K_new_index].set((state.xloc)[K_index])
        xloc_new = xloc_new.at[K_index].set(0)
        S_z_new = (state.S_z).at[K_index%L].set((state.S_z)[K_index%L]-(1.-2.*(K_index/L).astype(int)))
        S_z_new = S_z_new.at[K_new_index%L].set(S_z_new[K_new_index%L]+(1.-2.*(K_new_index/L).astype(int)))
        new_no_det_log_amplitude = log_jastrow_amplitude(params,S_z_new,xloc_new) + log_ph_amplitude(params,state.X_Phonons,state.Y_Phonons,xloc_new)

        return (1-(K_new_index/L).astype(int)*2)*state.sign*sign_new*jnp.exp(new_no_det_log_amplitude+log_det_new-state.log_amp)

    def give_0(l_index, K_new_index, state, params):
        return 0.

    amp_ratio = 0.
    K_index=state.occupied_sites[l_index]
    # Moving in direction_v at nearest neighbour!!!
    K_new_index=ivic[K_index%L,direction_v,0]+L*(K_index/L).astype(int)
    amp_ratio = lax.cond(state.xloc[K_new_index]==0, compute_new_det, give_0, l_index, K_new_index, state, params)

    return amp_ratio


vmapped_amp_electrons_hopping_direction_v = vmap(amp_electron_i_hopp_direction_v, in_axes=(0, None, None, None))

def give_ivic_below_NN(site):
    return ivic[site,2,0]
vmapped_ivic_below_NN = vmap(give_ivic_below_NN)

def give_ivic_left_NN(site):
    return ivic[site,3,0]
vmapped_ivic_left_NN = vmap(give_ivic_left_NN)


@jit
def local_bond_order_param(state, params) -> float:
    no_det_log_amplitude = log_jastrow_amplitude(params,state.S_z,state.xloc) + log_ph_amplitude(params,state.X_Phonons,state.Y_Phonons,state.xloc)
    U_mat = get_U_mat(params,state.X_Phonons,state.Y_Phonons)
    sign_, logdet = determinant_fixed_U(U_mat,state.occupied_sites)
    state = state.replace(U=U_mat)
    state = state.replace(log_amp=no_det_log_amplitude + logdet)
    state = state.replace(sign=sign_)

    amp_bonds_above_hopp = vmapped_amp_electrons_hopping_direction_v(jnp.arange(N_e), state, params, 0)
    amp_bonds_right_hopp = vmapped_amp_electrons_hopping_direction_v(jnp.arange(N_e), state, params, 1)
    amp_bonds_below_hopp = vmapped_amp_electrons_hopping_direction_v(jnp.arange(N_e), state, params, 2)
    amp_bonds_left__hopp = vmapped_amp_electrons_hopping_direction_v(jnp.arange(N_e), state, params, 3)

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

@jit
def amp_two_electrons_jumping_right_right_and_left_left_fixed_l_index(l_index, val):

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
        new_no_det_log_amplitude = log_jastrow_amplitude(params,S_z_new,xloc_new) + log_ph_amplitude(params,state.X_Phonons,state.Y_Phonons,xloc_new) 

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

@jit
def amp_two_electrons_jumping_right_right_and_left_left_fixed_m_index(m_index, val):

    state, params, delta_amplitudes = val

    val_I = m_index, state, params, delta_amplitudes

    val_I = lax.fori_loop(N_e_up,N_e,amp_two_electrons_jumping_right_right_and_left_left_fixed_l_index, val_I)

    m_index, state, params, delta_amplitudes = val_I

    return state, params, delta_amplitudes

@jit
def amp_two_electrons_jumping_right_right_and_left_left(state, params, delta_amplitudes):

    state, params, delta_amplitudes = lax.fori_loop(0,N_e_up,amp_two_electrons_jumping_right_right_and_left_left_fixed_m_index,( state, params, delta_amplitudes ) )

    return state, params, delta_amplitudes


@jit
def amp_one_electron_jumping_fixed_l_index(l_index, val):

    state, params, delta_amplitudes = val

    def compute_new_det(l_index, K_new_index, I_new_index, state, params):
        K_index = (state.occupied_sites)[l_index]
        occupied_sites_new = (state.occupied_sites).at[l_index].set(I_new_index)
        sign_new, log_det_new = determinant_fixed_U(state.U,occupied_sites_new)

        xloc_new = (state.xloc).at[I_new_index].set((state.xloc)[K_index])
        xloc_new = xloc_new.at[K_index].set(0)
        S_z_new = (state.S_z).at[K_index%L].set((state.S_z)[K_index%L]+1.)
        S_z_new = S_z_new.at[I_new_index%L].set(S_z_new[I_new_index%L]-1.)
        new_no_det_log_amplitude = log_jastrow_amplitude(params,S_z_new,xloc_new) + log_ph_amplitude(params,state.X_Phonons,state.Y_Phonons,xloc_new) 

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


@jit
def amp_one_electron_jumping(state, params, delta_amplitudes):

    state, params, delta_amplitudes = lax.fori_loop(N_e_up,N_e,amp_one_electron_jumping_fixed_l_index,( state, params, delta_amplitudes ) )

    return state, params, delta_amplitudes


@jit
def local_delta_order_param(state, params) -> float:
    no_det_log_amplitude = log_jastrow_amplitude(params,state.S_z,state.xloc) + log_ph_amplitude(params,state.X_Phonons,state.Y_Phonons,state.xloc)
    U_mat = get_U_mat(params,state.X_Phonons,state.Y_Phonons)
    sign_, logdet = determinant_fixed_U(U_mat,state.occupied_sites)
    state = state.replace(U=U_mat)
    state = state.replace(log_amp=no_det_log_amplitude + logdet)
    state = state.replace(sign=sign_)

    delta_amplitudes = jnp.zeros((4,L))

    state, params, delta_amplitudes = amp_two_electrons_jumping_right_right_and_left_left(state, params, delta_amplitudes)
    state, params, delta_amplitudes = amp_one_electron_jumping(state, params, delta_amplitudes)

    delta_amplitudes /= two_L
    delta_amplitudes = delta_amplitudes.at[2,:].set(delta_amplitudes[2,:]*2.)

    return delta_amplitudes


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
def Fourier_corr_2D(corr):

  def Fourier_one_component(vec, k):
    return vec, jnp.real(jnp.dot(vec.flatten(),jnp.exp(1j*vmapped_r_times_k(jnp.arange(L),k[0],k[1]))))

  kx_arr = (2.*jnp.pi/Lx)*jnp.repeat(jnp.arange(Lx),Ly)
  ky_arr = (2.*jnp.pi/Ly)*jnp.tile(jnp.arange(Ly),Lx)
  k_arr = jnp.stack((kx_arr,ky_arr)).transpose()

  _, Fourier_vec = lax.scan(Fourier_one_component,corr,k_arr)

  return Fourier_vec


from jax.flatten_util import ravel_pytree
import jax.tree_util


@partial(jit, static_argnums=(2,3,4,5,6,7,))
def local_obs_SR(state, params, wf, grad_wf, grad_Xphonons_wf, lapl_Xphonons_wf, grad_Yphonons_wf, lapl_Yphonons_wf):
    # local energy per spin
    e_L = local_energy_t(state, params) + local_energy_U(state.xloc)+ local_energy_phonons(state.xloc, state.S_z,  state.occupied_sites, state.X_Phonons, state.Y_Phonons, params, grad_Xphonons_wf, lapl_Xphonons_wf, grad_Yphonons_wf, lapl_Yphonons_wf)
    # local operators for the estimation of the gradient of the energy
    O_L = grad_wf(params, state.occupied_sites, state.xloc, state.S_z,  state.X_Phonons, state.Y_Phonons)
    O_flat, _ = ravel_pytree(O_L)
    eO_L = e_L * O_flat
    OO_L = jnp.outer(jnp.conj(O_flat), O_flat)
    return {'e': e_L, 'O' : O_flat, 'eO' : eO_L, 'OO' : OO_L}


@partial(jit, static_argnums=(2,3,4,5,6,7,))
def local_obs(state, params, wf, grad_wf, grad_Xphonons_wf, lapl_Xphonons_wf, grad_Yphonons_wf, lapl_Yphonons_wf):
    # local energy per spin
    e_L = local_energy_t(state, params) + local_energy_U(state.xloc)+ local_energy_phonons(state.xloc, state.S_z,  state.occupied_sites, state.X_Phonons, state.Y_Phonons, params, grad_Xphonons_wf, lapl_Xphonons_wf, grad_Yphonons_wf, lapl_Yphonons_wf)

    appo_x_first = (state.xloc).astype(jnp.bool_)
    appo_x_second = appo_x_first.astype(jnp.int32)

    corr_n = corr_2D(appo_x_second[:L] + ( - appo_x_second[L:2*L] + 1. ) )
    N_q = Fourier_corr_2D(corr_n)
    corr_s = corr_2D(appo_x_second[:L] - ( - appo_x_second[L:2*L] + 1. ) )
    S_q = Fourier_corr_2D(corr_s)

    bond_x_amplitudes, bond_y_amplitudes = local_bond_order_param(state, params)
    x_order_parameter_bonds = jnp.sum(bond_x_amplitudes*stagg_x)
    y_order_parameter_bonds = jnp.sum(bond_y_amplitudes*stagg_y)
    corr_bonds_x = corr_2D(bond_x_amplitudes)
    Bx_q = Fourier_corr_2D(corr_bonds_x)
    corr_bonds_y = corr_2D(bond_y_amplitudes)
    By_q = Fourier_corr_2D(corr_bonds_y)
    X_diff = state.X_Phonons - state.X_Phonons[vmapped_ivic_left_NN(jnp.arange(L))]
    Y_diff = state.Y_Phonons - state.Y_Phonons[vmapped_ivic_below_NN(jnp.arange(L))]

    delta_corr = local_delta_order_param(state, params)

    return {'e': e_L, 'corr_n' : corr_n, 'N_q' : N_q, 'corr_s' : corr_s, 'S_q' : S_q, 'corr_bx' : corr_bonds_x, 'Bx_q': Bx_q, 'corr_by' : corr_bonds_y, 'By_q': By_q, 'order_bonds_x': x_order_parameter_bonds, 'order_bonds_y': y_order_parameter_bonds, 'bond_amplitudes_x': bond_x_amplitudes, 'bond_amplitudes_y': bond_y_amplitudes, 'X_diff': X_diff, 'Y_diff': Y_diff, 'corr_dx' : delta_corr[1,:], 'corr_dy' : delta_corr[0,:], 'corr_dxy' : delta_corr[2,:], 'corr_d_onsite' : delta_corr[3,:]}


#######################################
#######################################
### Functions to perform Metropolis ###
#######################################
#######################################

@struct.dataclass
class acceptance():
    acc_call_hop_NN: int
    tot_call_hop_NN: int
    acc_call_spin_flip: int
    tot_call_spin_flip: int
    acc_call_phonon_step: int
    tot_call_phonon_step: int


@partial(jit, static_argnums=(2,))
def single_hop_NN_step(i, vals, wf):
    state, params, a, rands = vals

    a = a.replace(tot_call_hop_NN=a.tot_call_hop_NN+1)
    l_index = (rands[i, 0] * N_e).astype(int)
    where_to_move = (rands[i, 1] * imulti[0]).astype(int)
    K_index= state.occupied_sites[l_index]
    K_new_index=ivic[K_index%L,where_to_move,0]+L*((K_index/L).astype(int))

    def metropolis_func(vals, wf):
        state, a, l_index, K_index, K_new_index = vals
        occupied_sites_new = (state.occupied_sites).at[l_index].set(K_new_index)
        xloc_new = (state.xloc).at[K_new_index].set((state.xloc)[K_index])
        xloc_new = xloc_new.at[K_index].set(0)
        S_z_new = (state.S_z).at[K_index%L].set((state.S_z)[K_index%L]-(1.-2.*(K_index/L).astype(int)))
        S_z_new = S_z_new.at[K_new_index%L].set(S_z_new[K_new_index%L]+(1.-2.*(K_new_index/L).astype(int)))
        sign, log_det_new = determinant_fixed_U(state.U,occupied_sites_new)
        log_amp_new = log_jastrow_amplitude(params,S_z_new,xloc_new)+log_ph_amplitude(params,state.X_Phonons,state.Y_Phonons,xloc_new)+log_det_new

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

    state, a = lax.cond( (state.xloc)[K_new_index]==0 , partial(metropolis_func, wf=wf), no_metropolis_func, (state, a, l_index, K_index, K_new_index))
    
    return [state, params, a, rands]

@partial(jit, static_argnums=(2,))
def single_spin_flip_step(i, vals, wf):
    state, params, a, rands = vals

    a = a.replace(tot_call_spin_flip=a.tot_call_spin_flip+1)

    l_index = (rands[i, 0] * N_e).astype(int)
    where_to_move = (rands[i, 1] * imulti[0]).astype(int)
    K_index=(state.occupied_sites)[l_index]
    K_new_index=ivic[K_index%L,where_to_move,0]+L*((K_index/L).astype(int))

    I_index=K_index%L+L*(1-(K_index/L).astype(int))
    I_new_index=K_new_index%L+L*(1-(K_new_index/L).astype(int))
    m_index = ((state.xloc)[I_index]).astype(int)-1

    def metropolis_func(vals, wf):
        state, a = vals
        occupied_sites_new = (state.occupied_sites).at[l_index].set(K_new_index)
        occupied_sites_new = occupied_sites_new.at[m_index].set(I_new_index)
        xloc_new = (state.xloc).at[K_new_index].set((state.xloc)[K_index])
        xloc_new = xloc_new.at[K_index].set(0)
        xloc_new = xloc_new.at[I_new_index].set(xloc_new[I_index])
        xloc_new = xloc_new.at[I_index].set(0)
        sign, log_det_new = determinant_fixed_U(state.U,occupied_sites_new)
        log_amp_new = log_jastrow_amplitude(params,state.S_z,xloc_new)+log_ph_amplitude(params,state.X_Phonons,state.Y_Phonons,xloc_new)+log_det_new

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

    state, a = lax.cond( ((state.xloc)[K_new_index]==0)*((state.xloc)[I_new_index]==0)*(m_index>-1) , partial(metropolis_func, wf=wf), no_metropolis_func, (state, a))

    return [state, params, a, rands]


@partial(jit, static_argnums=(2,))
def single_phonon_step(i, vals, wf):
    state, params, a, rands = vals

    a = a.replace(tot_call_phonon_step=a.tot_call_phonon_step+1)
    site_index = (rands[i, 0] * two_L).astype(int)
    displacement_x = jnp.zeros(L)
    displacement_x = displacement_x.at[site_index%L].set((rands[i, 1]-0.5)*displ_phon_move*(1-(site_index/L).astype(int)))
    displacement_y = jnp.zeros(L)
    displacement_y = displacement_y.at[site_index%L].set((rands[i, 1]-0.5)*displ_phon_move*((site_index/L).astype(int)))

    def metropolis_func(vals, wf):
        state, a, site_index, displacement_x, displacement_y = vals
        U_mat_new = get_U_mat(params, state.X_Phonons + displacement_x, state.Y_Phonons + displacement_y)
        sign, log_det_new = determinant_fixed_U(U_mat_new, state.occupied_sites)
        log_amp_new = log_jastrow_amplitude(params, state.S_z, state.xloc) + log_ph_amplitude(params, state.X_Phonons + displacement_x, state.Y_Phonons + displacement_y, state.xloc) + log_det_new

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

    state, a = partial(metropolis_func, wf=wf)((state, a, site_index, displacement_x, displacement_y))

    return [state, params, a, rands]


@partial(jit, static_argnums=(2,))
def single_Metropolis_step(i, vals, wf):
    state, params, a, rands = vals

    def electron_move(n, vals):
        return lax.cond( rands[n, 3] < p_spin_flip, partial(single_spin_flip_step, wf=wf),  partial(single_hop_NN_step, wf=wf), n, vals)

    vals = lax.cond( rands[3*i, 3] < p_moving_electrons, electron_move, partial(single_phonon_step, wf=wf), 3*i, vals)
    vals = lax.cond( rands[3*i+1, 3] < p_moving_electrons, electron_move, partial(single_phonon_step, wf=wf), 3*i+1, vals)
    vals = lax.cond( rands[3*i+2, 3] < p_moving_electrons, electron_move, partial(single_phonon_step, wf=wf), 3*i+2, vals)

    return vals


@partial(jit, static_argnums=(2,3,4,5,6,7,))
def sweep(i, var, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons):
    state, params, key, a, mean_values = var

    # Generate all the random numbers
    key, subkey = random.split(key, num=2)
    rand = random.uniform(subkey, (3*sparse_ave_length, 4))

    state, params, a, rand = lax.fori_loop(0, sparse_ave_length, partial(single_Metropolis_step, wf=wf), [state, params, a, rand])

    mean_values = jax.tree_util.tree_map(lambda x, y : x + y, mean_values, local_obs(state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons))

    return [state, params, key, a, mean_values]


@partial(jit, static_argnums=(2,3,4,5,6,7,))
def sweep_SR(i, var, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons):
    state, params, key, a, mean_values = var

    # Generate all the random numbers
    key, subkey = random.split(key, num=2)
    rand = random.uniform(subkey, (3*sparse_ave_length, 4))

    state, params, a, rand = lax.fori_loop(0, sparse_ave_length, partial(single_Metropolis_step, wf=wf), [state, params, a, rand])

    mean_values = jax.tree_util.tree_map(lambda x, y : x + y, mean_values, local_obs_SR(state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons))

    return [state, params, key, a, mean_values]
        

##################################
### Stochastic Reconfiguration ###
##################################

@partial(jit, static_argnums=(0, 3, 4, 5, 6, 7, 8,))
def mc_sweeps_SR(nsweeps, state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a):

    # inizialize mean_values
    mean_values = local_obs_SR(state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons)
    mean_values = jax.tree_util.tree_map(lambda x: x*0, mean_values)

    state, params, key, a, mean_values = lax.fori_loop(0, nsweeps, partial(sweep_SR, wf=wf, wf_grad=wf_grad, wf_grad_X_phonons=wf_grad_X_phonons, wf_lapl_X_phonons=wf_lapl_X_phonons, wf_grad_Y_phonons=wf_grad_Y_phonons, wf_lapl_Y_phonons=wf_lapl_Y_phonons), [state, params, key, a, mean_values])

    mean_values = jax.tree_util.tree_map(lambda x: x/nsweeps, mean_values)

    return state, key, a, mean_values

@partial(jit, static_argnums=(0, 3, 4, 5, 6, 7, 8,))
def mc_sweeps_Thermaliz(nsweeps, state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a):

    mean_values = local_obs_SR(state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons)
    state, params, key, a, mean_values = lax.fori_loop(0, nsweeps, partial(sweep_SR, wf=wf, wf_grad=wf_grad, wf_grad_X_phonons=wf_grad_X_phonons, wf_lapl_X_phonons=wf_lapl_X_phonons, wf_grad_Y_phonons=wf_grad_Y_phonons, wf_lapl_Y_phonons=wf_lapl_Y_phonons), [state, params, key, a, mean_values])

    return state, key, a


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

        alpha_dot=jnp.linalg.solve(S_matrix_TOT,E_O_x_TOT)

        norm_ = lax.max(jnp.linalg.norm(alpha_dot),2.)

        param_array = param_array.at[i,:].set(par_flat)

        par_flat += alpha_dot*gamma*(2./norm_)

        energy_array = energy_array.at[i].set(E_TOT)


    par_flat, _ = mpi4jax.bcast(par_flat, root=0, comm=comm, token=tok4)

    params = unravel_params(par_flat)
        
    return [params, gamma, energy_array, param_array]
    

def SR_n_iterations(N_iterations, nsweeps, state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a, gamma, energy_array, params_array, MAX_time, epsilon=10e-4):

    start_time = time.time()

    state = state.replace(U = get_U_mat(params, state.X_Phonons, state.Y_Phonons) )
    state = state.replace(log_amp = wf(params, state.occupied_sites, state.xloc, state.S_z, state.X_Phonons, state.Y_Phonons) )

    state, key, a, _ = mc_sweeps_SR(nsweeps, state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

    if (rank==0):
        print("\n", flush=True)
    for step in range(N_iterations):

        state, key, a = mc_sweeps_Thermaliz(int(nsweeps/20), state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons,  key, a)

        state, key, a, mean_values = mc_sweeps_SR(nsweeps, state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

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


#################
### Measuring ###
#################
             
@partial(jit, static_argnums=(0, 3, 4, 5, 6, 7, 8,))
def mc_sweeps_Measure(nsweeps, state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a):

    # inizialize mean_values
    mean_values = local_obs(state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons)
    mean_values = jax.tree_util.tree_map(lambda x: x*0, mean_values)

    state, params, key, a, mean_values = lax.fori_loop(0, nsweeps, partial(sweep, wf=wf, wf_grad=wf_grad, wf_grad_X_phonons=wf_grad_X_phonons, wf_lapl_X_phonons=wf_lapl_X_phonons, wf_grad_Y_phonons=wf_grad_Y_phonons, wf_lapl_Y_phonons=wf_lapl_Y_phonons), [state, params, key, a, mean_values])

    mean_values = jax.tree_util.tree_map(lambda x: x/nsweeps, mean_values)

    return state, key, a, mean_values


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



                             
def block_average(N_blocks, L_each_block, state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a, energy_array, observables, MAX_time):

    start_time = time.time()

    state = state.replace(U = get_U_mat(params, state.X_Phonons, state.Y_Phonons) )
    state = state.replace(log_amp = wf(params, state.occupied_sites, state.xloc, state.S_z, state.X_Phonons, state.Y_Phonons) )

    state, key, a, _ = mc_sweeps_Measure(L_each_block, state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

    if (rank==0):
        print("\n", flush=True)
    for step in range(N_blocks):

        state, key, a, mean_values = mc_sweeps_Measure(L_each_block, state, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a)

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
state_system = state_system.replace(U =  get_U_mat(params, state_system.X_Phonons, state_system.Y_Phonons))

if (rank==0):
    print(params)
    print("\n\n")

##############################
### Running the simulation ###
##############################

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

    state_system, params, key, a, earr, parr = SR_n_iterations(n_SR_steps, int(n_sweeps/n_chains), state_system, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a, dt_step, earr, parr, MAX_time)

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

    state_system, earr, observables = block_average(N_blocks, int(L_each_block/n_chains), state_system, params, wf, wf_grad, wf_grad_X_phonons, wf_lapl_X_phonons, wf_grad_Y_phonons, wf_lapl_Y_phonons, key, a, earr, [corr_n_arr, Nq_arr, corr_s_arr, Sq_arr, corr_bx_arr, Bx_q_arr, corr_by_arr, By_q_arr, order_par_bond_x, order_par_bond_y, bond_amplitudes_arr_X, bond_amplitudes_arr_Y, X_diff_arr, Y_diff_arr, corr_dx_arr, corr_dy_arr, corr_dxy_arr, corr_d_onsite_arr ], MAX_time)

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



