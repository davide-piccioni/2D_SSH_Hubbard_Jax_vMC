import jax.numpy as jnp
from SSH_Hubbard.lattice import *
from SSH_Hubbard.set_system import *


#################################
### Local state of the system ###
#################################

from flax import struct

class state_Fermions_and_Bosons(struct.PyTreeNode):
    xloc : jnp.array            # array of coordinates of the fermions
    occupied_sites :jnp.array   # postions of the fermions, ordered as their construction operators act on vacuum
    S_z : jnp.array             # S_z (spin along z) of the fermions
    X_Phonons : jnp.array       # phonon coordinates for X-modes
    Y_Phonons : jnp.array       # phonon coordinates for Y-modes
    U : jnp.array               # unitary matrix that diagonalizes the auxiliary Hamiltonian
    log_amp : jnp.float64       # log amplitude of the wave function
    sign : jnp.float64          # sign of the wave function


#################################
### Variational wave function ###
#################################

from jax import lax
from flax import linen as nn
from jax import jit, grad, jacfwd, hessian 
from typing import Sequence


class slater_det(nn.Module):  
    hopping_list: Sequence[int]
    sWave_pair_list: Sequence[int]
    dWave_pair_list: Sequence[int]

    def setup(self):
        self.hopping_values = self.param("hopp_par", nn.initializers.constant(-1.), (len(self.hopping_list),), jnp.float64)
        self.sWave_pair_values = self.param("pair_Swave_par", nn.initializers.constant(0.),  (len(self.sWave_pair_list),), jnp.float64)
        self.dWave_pair_values = self.param("pair_dwave_par", nn.initializers.constant(0.),  (len(self.dWave_pair_list),), jnp.float64)

        self.Jastrow_spin = self.param("spin_jastrow", nn.initializers.constant(0.),  ((1+Lx//2)*(1+Ly//2)-1,), jnp.float64)

        self.Jastrow_coupling_Xel = self.param("s_coupling_Xel_jastrow", nn.initializers.constant(0.),  ((1+Lx//2)*(1+Ly//2)-2,), jnp.float64)
        self.Jastrow_coupling_Yel = self.param("s_coupling_Yel_jastrow", nn.initializers.constant(0.),  ((1+Lx//2)*(1+Ly//2)-2,), jnp.float64)

        self.phonons_XX_jastrow = self.param("phonons_XX_jastrow", nn.initializers.constant(0.),  ( (1+Lx//2)*(1+Ly//2)-1,), jnp.float64)
        self.phonons_XY_jastrow = self.param("phonons_XY_jastrow", nn.initializers.constant(0.),  ( (1+Lx//2)*(1+Ly//2)-1,), jnp.float64)
        self.phonons_YY_jastrow = self.param("phonons_YY_jastrow", nn.initializers.constant(0.),  ( (1+Lx//2)*(1+Ly//2)-1,), jnp.float64)


#        self.bZ = self.param("bZ", nn.initializers.constant(0.7), (1,), jnp.float64)
        self.g = self.param("g", nn.initializers.constant(0.7), (1,), jnp.float64)
        self.f_Swave = self.param("f_Swave", nn.initializers.constant(0.), (1,), jnp.float64)
        self.f_dwave = self.param("f_dwave", nn.initializers.constant(0.), (1,), jnp.float64)
        self.z_x = self.param("z_x", nn.initializers.constant(0.5), (1,), jnp.float64)
        self.z_y = self.param("z_y", nn.initializers.constant(0.5), (1,), jnp.float64)
        self.x_rescaled = self.param("x_rescaled", nn.initializers.constant(0.), (1,), jnp.float64)
        self.y_rescaled = self.param("y_rescaled", nn.initializers.constant(0.), (1,), jnp.float64)
        
    def __call__(self, occupied_sites, xloc, S_z, X_Phonons, Y_Phonons):        

########################
### Determinant part ###
########################

        # The auxiliary Hamiltonian H_0 is built from the hopping and pairing terms
        # and the backflow terms that depend directly upon the phonon configurations. 
        # Then the matrix of the orbitals U that diagonalizes H_0 is computed.
        # The determinant is then computed.

        H_0 = jnp.zeros((2*L,2*L))

        H_0 += -1.*hopping_array_matrices[1,:,:]

#        H_0 += eps_rand_mat

#        H_0 += self.bZ * AF_magnetic_field_matrix_Bz; 
        
        for i,index in enumerate(self.hopping_list):
            H_0 += self.hopping_values[i]*hopping_array_matrices[index,:,:]
        
        for i,index in enumerate(self.sWave_pair_list):
            H_0 += self.sWave_pair_values[i]*sWave_pair_array_matrices[index,:,:]

        for i,index in enumerate(self.dWave_pair_list):
            H_0 += self.dWave_pair_values[i]*dWave_pair_array_matrices[index,:,:]
     
        phonons_matrix_Xbonds = jnp.zeros((L,L))
        phonons_matrix_Xbonds = fill_NN_X_matrix(X_Phonons, phonons_matrix_Xbonds)
        phonons_matrix_Ybonds = jnp.zeros((L,L))
        phonons_matrix_Ybonds = fill_NN_Y_matrix(Y_Phonons, phonons_matrix_Ybonds)

        H_0 += self.g*jnp.block([[phonons_matrix_Xbonds+phonons_matrix_Ybonds,jnp.zeros((L,L))],[jnp.zeros((L,L)),-phonons_matrix_Xbonds-phonons_matrix_Ybonds]])
        H_0 += self.f_Swave*jnp.block([[jnp.zeros((L,L)),phonons_matrix_Xbonds+phonons_matrix_Ybonds],[phonons_matrix_Xbonds+phonons_matrix_Ybonds,jnp.zeros((L,L))]])
        H_0 += self.f_dwave*jnp.block([[jnp.zeros((L,L)),phonons_matrix_Xbonds-phonons_matrix_Ybonds],[phonons_matrix_Xbonds-phonons_matrix_Ybonds,jnp.zeros((L,L))]])

        eigs, U = jnp.linalg.eigh(H_0)

#         det = jnp.linalg.det(U[occupied_sites,:N_e])
        (sign, logdet) = jnp.linalg.slogdet(U[occupied_sites,:N_e])
        
####################
### Jastrow part ###
####################

        # Charge-charge Jastrow factor
        # It is built from S_z since the code works with Particle-Hole representation
        # Hence, the down electrons are seen as holes and vice versa.

        vmat = unroll_vpot_Mat(self.Jastrow_spin)
        log_jastrow_amp = S_z.T@vmat@S_z

###################
### Phonon part ###
###################

        # Coherent state of the phonons
        log_phonons_coherent_amp =  -0.5*self.x_rescaled[0]*jnp.power(X_Phonons-self.z_x*stagg_x,2).sum()
        log_phonons_coherent_amp += -0.5*self.y_rescaled[0]*jnp.power(Y_Phonons-self.z_y*stagg_y,2).sum()

        # Phonon-phonon Jastrow factor
        vmat_XX = unroll_vpot_Mat(self.phonons_XX_jastrow)
        log_phonons_coherent_amp += X_Phonons.T@vmat_XX@X_Phonons 

        vmat_XY = unroll_vpot_Mat(self.phonons_XY_jastrow)
        log_phonons_coherent_amp += Y_Phonons.T@vmat_XY@X_Phonons 
        log_phonons_coherent_amp += X_Phonons.T@vmat_XY@Y_Phonons 

        vmat_YY = unroll_vpot_Mat(self.phonons_YY_jastrow)
        log_phonons_coherent_amp += Y_Phonons.T@vmat_YY@Y_Phonons 

        # Electron-phonon Jastrow factor
        vmat_X_elPH = unroll_vpot_Mat_el_Phonons(self.Jastrow_coupling_Xel,X_Phonons) 
        log_phonons_coherent_amp += S_z.T@vmat_X_elPH@S_z
        vmat_Y_elPH = unroll_vpot_Mat_el_Phonons(self.Jastrow_coupling_Yel,Y_Phonons) 
        log_phonons_coherent_amp += S_z.T@vmat_Y_elPH@S_z

        return logdet + log_phonons_coherent_amp + log_jastrow_amp
    
    # Method to compute the matrix U of the orbitals alone
    def get_U(self, params, X_Phonons, Y_Phonons):
        H_0 = jnp.zeros((2*L,2*L))

        H_0 += -1.*hopping_array_matrices[1,:,:]

#        H_0 += params['params']["bZ"] * AF_magnetic_field_matrix_Bz; 

        for i,index in enumerate(hopping_list):
            H_0 += params['params']["hopp_par"][i]*hopping_array_matrices[index]

        for i,index in enumerate(sWave_pair_list):
            H_0 += params['params']["pair_Swave_par"][i]*sWave_pair_array_matrices[index]

        for i,index in enumerate(dWave_pair_list):
            H_0 += params['params']["pair_dwave_par"][i]*dWave_pair_array_matrices[index]


        phonons_matrix_Xbonds = jnp.zeros((L,L))
        phonons_matrix_Xbonds = fill_NN_X_matrix(X_Phonons, phonons_matrix_Xbonds)
        phonons_matrix_Ybonds = jnp.zeros((L,L))
        phonons_matrix_Ybonds = fill_NN_Y_matrix(Y_Phonons, phonons_matrix_Ybonds)

        H_0 += params['params']["g"]*jnp.block([[phonons_matrix_Xbonds+phonons_matrix_Ybonds,jnp.zeros((L,L))],[jnp.zeros((L,L)),-phonons_matrix_Xbonds-phonons_matrix_Ybonds]])
        H_0 += params['params']["f_Swave"]*jnp.block([[jnp.zeros((L,L)),phonons_matrix_Xbonds+phonons_matrix_Ybonds],[phonons_matrix_Xbonds+phonons_matrix_Ybonds,jnp.zeros((L,L))]])
        H_0 += params['params']["f_dwave"]*jnp.block([[jnp.zeros((L,L)),phonons_matrix_Xbonds-phonons_matrix_Ybonds],[phonons_matrix_Xbonds-phonons_matrix_Ybonds,jnp.zeros((L,L))]])

        eigs, U = jnp.linalg.eigh(H_0)

        return U

    # Method to compute the log amplitude of the Phonon-Phonon and Phonon-electrons Jastrow alone
    def get_only_log_phonons_amplitude(self, params, X_Phonons, Y_Phonons, xloc):
        log_phonons_coherent_amp =  -0.5*params['params']['x_rescaled'][0]*jnp.power(X_Phonons-params['params']["z_x"]*stagg_x,2).sum()
        log_phonons_coherent_amp += -0.5*params['params']['y_rescaled'][0]*jnp.power(Y_Phonons-params['params']["z_y"]*stagg_y,2).sum()

        vmat_XX = unroll_vpot_Mat(params['params']['phonons_XX_jastrow'])
        log_phonons_coherent_amp += X_Phonons.T@vmat_XX@X_Phonons 

        vmat_XY = unroll_vpot_Mat(params['params']['phonons_XY_jastrow'])
        log_phonons_coherent_amp += Y_Phonons.T@vmat_XY@X_Phonons 
        log_phonons_coherent_amp += X_Phonons.T@vmat_XY@Y_Phonons 

        vmat_YY = unroll_vpot_Mat(params['params']['phonons_YY_jastrow'])
        log_phonons_coherent_amp += Y_Phonons.T@vmat_YY@Y_Phonons 

        
        _appo_x_first_ = xloc.astype(jnp.bool_)
        _appo_x_second_ = _appo_x_first_.astype(jnp.int32)
        _S_z_ = 1.*(_appo_x_second_[:L] - _appo_x_second_[L:2*L])

        vmat_X_elPH = unroll_vpot_Mat_el_Phonons(params['params']['s_coupling_Xel_jastrow'],X_Phonons) 
        log_phonons_coherent_amp += _S_z_.T@vmat_X_elPH@_S_z_
        vmat_Y_elPH = unroll_vpot_Mat_el_Phonons(params['params']['s_coupling_Yel_jastrow'],Y_Phonons) 
        log_phonons_coherent_amp += _S_z_.T@vmat_Y_elPH@_S_z_

        return log_phonons_coherent_amp

    # Method to compute the log amplitude of the charge-charge Jastrow alone
    def get_only_log_Jastrow_amplitude(self, params, S_z, xloc):

        vmat = unroll_vpot_Mat(params['params']['spin_jastrow'])
        log_jastrow_amp = S_z.T@vmat@S_z

        return log_jastrow_amp 

@jit
def determinant_fixed_U(U,occupied_sites):
    (sign, logdet) = jnp.linalg.slogdet(U[occupied_sites,:N_e])
    return sign, logdet 


# Old method to compute Jastrow preserving translational and rotational invariance
@jit
def O_x_scra_OLD_Transl_invariant(S_z, O_x):
    
    def update_O_x_fixed_i(i, val):
        def single_update_fixed_i(j, val):
            S_z, O_x, i = val
            O_x = O_x.at[distances[i,j]].set(O_x[distances[i,j]]+S_z[i]*S_z[j])
            return S_z, O_x, i
        S_z, O_x = val
        S_z, O_x, i = lax.fori_loop(0, L, single_update_fixed_i, (S_z, O_x, i))
        return S_z, O_x
    
    S_z, O_x = lax.fori_loop(0, L, update_O_x_fixed_i, (S_z, O_x))
    O_x = O_x.at[0].set(jnp.power(S_z,2).sum())
    return O_x

# New method to compute Jastrow breaking rotational invariance
# This jastrow preserves translational invariance and reflection along each axis
@jit
def unroll_vpot_Mat(vpot_2D_flat):
    
    vpot_2D = jnp.concatenate((vpot_2D_flat,jnp.array([0,]))).reshape(1+Lx//2,1+Ly//2)

    vpot_2D_complete = jnp.block([
        [vpot_2D,                      jnp.fliplr(vpot_2D[:,1:-1])                 ],
        [jnp.flipud(vpot_2D[1:-1,:]),  jnp.flipud(jnp.fliplr(vpot_2D[1:-1,1:-1]))  ]
    ])
    
    def unroll_vpot_Mat_ax1(vpot_ex, _):
        return jnp.roll(vpot_ex,1, axis=1), jnp.roll(vpot_ex,1, axis=1).flatten()

    def unroll_vpot_Mat_both_axes( vpot_2D, _  ):
        appo, vmat_Ly_lines = lax.scan(unroll_vpot_Mat_ax1,jnp.roll(vpot_2D,-1, axis=1),jnp.zeros(Ly))
        appo = jnp.roll(appo,1, axis=1)
        return jnp.roll(appo,1, axis=0), vmat_Ly_lines

    appo, vmat = lax.scan(unroll_vpot_Mat_both_axes,vpot_2D_complete,jnp.zeros(Lx))

    vmat = vmat.reshape(L,L)
    
    return vmat

# New method to compute Jastrow breaking rotational invariance
# This jastrow preserves translational invariance and reflection along each axis
@jit
def unroll_vpot_Mat_el_Phonons(vpot_2D_flat, XorY_phon):
    
    vpot_2D = jnp.concatenate((jnp.array([0,]),vpot_2D_flat,jnp.array([0,]))).reshape(1+Lx//2,1+Ly//2)

    vpot_2D_complete = jnp.block([
        [vpot_2D,                      jnp.fliplr(vpot_2D[:,1:-1])                 ],
        [jnp.flipud(vpot_2D[1:-1,:]),  jnp.flipud(jnp.fliplr(vpot_2D[1:-1,1:-1]))  ]
    ])
    
    def unroll_vpot_Mat_ax1(vpot_ex, _):
        return jnp.roll(vpot_ex,1, axis=1), jnp.roll(vpot_ex,1, axis=1).flatten()

    def unroll_vpot_Mat_both_axes( vpot_2D, _  ):
        appo, vmat_Ly_lines = lax.scan(unroll_vpot_Mat_ax1,jnp.roll(vpot_2D,-1, axis=1),jnp.zeros(Ly))
        appo = jnp.roll(appo,1, axis=1)
        return jnp.roll(appo,1, axis=0), vmat_Ly_lines

    appo, vmat = lax.scan(unroll_vpot_Mat_both_axes,vpot_2D_complete,jnp.zeros(Lx))

    vmat = vmat.reshape(L,L)
                              
    phon_mat = jnp.tile(XorY_phon, (1,L) ).reshape(L,L)
    
    return jnp.multiply(jnp.multiply(vmat,phon_mat.T-phon_mat),bond_hopping_difference_sign)

# Methods to fill the auxiliary Hamiltonian with the backflow terms computed from the phonon coordinates
@jit
def fill_NN_X_matrix(Xphons, phon_matrix):

    def fill_element(phon_matrx, i):
        j = ivic[i,1,0]
        phon_matrx = phon_matrx.at[i,j].set((Xphons[j]-Xphons[i])*bond_hopping_difference_sign[i,j])
        j = ivic[i,3,0]
        phon_matrx = phon_matrx.at[i,j].set((Xphons[j]-Xphons[i])*bond_hopping_difference_sign[i,j])
        return phon_matrx, i
    phon_matrix, _ = lax.scan(fill_element, phon_matrix, jnp.arange(0, L))
    return phon_matrix

@jit
def fill_NN_Y_matrix(Xphons, phon_matrix):

    def fill_element(phon_matrx, i):
        j = ivic[i,0,0]
        phon_matrx = phon_matrx.at[i,j].set((Xphons[j]-Xphons[i])*bond_hopping_difference_sign[i,j])
        j = ivic[i,2,0]
        phon_matrx = phon_matrx.at[i,j].set((Xphons[j]-Xphons[i])*bond_hopping_difference_sign[i,j])
        return phon_matrx, i
    phon_matrix, _ = lax.scan(fill_element, phon_matrix, jnp.arange(0, L))
    return phon_matrix
    
        
