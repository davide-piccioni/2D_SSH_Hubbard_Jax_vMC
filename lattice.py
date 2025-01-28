import jax.numpy as jnp


Lx = 8 # Length of the lattice
n_distances = 14 # Number of total distances
n_max_dist = 4 # Cutoff distance for hopping, pairing terms in the slater determinant


# The code works only on Square Lattices right now, possibly use Lx even
Ly    = Lx
L     = Lx*Ly 
two_L = 2*L

# Opening ivic and checking everyhing works
with open("./Lattice_Maker/lattice.npy", "rb") as file:
    L_check = jnp.load(file)
    two_L_check = jnp.load(file)
    n_distances_check = jnp.load(file)
    ivic = jnp.load(file)
    distances = jnp.load(file)
    imulti = tuple(jnp.load(file))

if(L_check!=L or two_L_check!=two_L or n_distances!=n_distances_check):
    print("ERROR with lattice dimension")
    print("Inside \"lattice.npy\" you have: L = ",L_check,"; n_distances = ",n_distances_check)
    print("Inside \"lattice.py\"  you set: L = ",L,"; n_distances = ",n_distances)
    exit()

# Matrices to construct slater determinant from an auxiliary quadratic hamiltonian H_0
hopping_array_matrices = jnp.zeros((n_max_dist,two_L,two_L))
sWave_pair_array_matrices = jnp.zeros((n_max_dist,two_L,two_L))
dWave_pair_array_matrices = jnp.zeros((3,two_L,two_L))
AF_magnetic_field_matrix_Bz  = jnp.zeros((two_L,two_L))

for i in range(L):
    hopping_array_matrices = hopping_array_matrices.at[0,i,i].set(-1.)
    hopping_array_matrices = hopping_array_matrices.at[0,i+L,i+L].set(1.)

for distance in range(1,n_max_dist):
    for i in range(L):
        for j in range(imulti[distance]):
            hopping_array_matrices = hopping_array_matrices.at[distance,i,ivic[i,j,distance-1]].set(1.)
            hopping_array_matrices = hopping_array_matrices.at[distance,i+L,ivic[i,j,distance-1]+L].set(-1.)

for i in range(L):
    sWave_pair_array_matrices = sWave_pair_array_matrices.at[0,i,i+L].set(1.)
    sWave_pair_array_matrices = sWave_pair_array_matrices.at[0,i+L,i].set(1.)

for distance in range(1,n_max_dist):
    for i in range(L):
        for j in range(imulti[distance]):
            sWave_pair_array_matrices = sWave_pair_array_matrices.at[distance,i,ivic[i,j,distance-1]+L].set(1.)
            sWave_pair_array_matrices = sWave_pair_array_matrices.at[distance,i+L,ivic[i,j,distance-1]].set(1.)

for distance in range(3):
    for i in range(L):
        for j in jnp.array([0,2]):
            dWave_pair_array_matrices = dWave_pair_array_matrices.at[distance,i,ivic[i,j,distance]+L].set(1.)
            dWave_pair_array_matrices = dWave_pair_array_matrices.at[distance,i+L,ivic[i,j,distance]].set(1.)
        for j in jnp.array([1,3]):
            dWave_pair_array_matrices = dWave_pair_array_matrices.at[distance,i,ivic[i,j,distance]+L].set(-1.)
            dWave_pair_array_matrices = dWave_pair_array_matrices.at[distance,i+L,ivic[i,j,distance]].set(-1.)

for i, sign_i in enumerate( (jnp.outer(jnp.cos(jnp.pi*2*(Lx//2)*jnp.arange(Lx)/Lx),jnp.cos(jnp.pi*2*(Ly//2)*jnp.arange(Ly)/Ly))).flatten() ):
    AF_magnetic_field_matrix_Bz = AF_magnetic_field_matrix_Bz.at[i,i].set(sign_i)
    AF_magnetic_field_matrix_Bz = AF_magnetic_field_matrix_Bz.at[i+L,i+L].set(sign_i)


# Given the expression  \sum_r (x_{r+e_x} - x_r)*( c^{\dagger}_{r+e_x} c_r + c^{\dagger}_r c_{r+e_x} ) 
# At each bond [r,r+e_x] is assigned a ceratin amplitude:  x_{r+e_x} - x_r
# When in the code I have two (nearest neighbour) sites R and R', with an electron jumping
# from R to R', the corresponding amplitude is either
# x_{R} - x_{R'} or x_{R'} - x_{R}. But WHICH ONE?
# The answer to the question is:
# bond = (x_{R'} - x_{R})*bond_hopping_difference_sign[R,R']
# The matrix element of "bond_hopping_difference_sign" takes care of the sign...


bond_hopping_difference_sign = jnp.zeros((L,L))

for distance in range(n_distances):
    if imulti[distance]>=4:
        for i in range(L):
            for j in range(imulti[distance]//2):
                bond_hopping_difference_sign = bond_hopping_difference_sign.at[i,ivic[i,j,distance]].set(1.)
            for j in range( (imulti[distance]//2),imulti[distance] ):
                bond_hopping_difference_sign = bond_hopping_difference_sign.at[i,ivic[i,j,distance]].set(-1.)
    elif imulti[distance]<=2:
        for j in range(imulti[distance]):
            appo_list = []
            for i in range(L):
                if ((i in appo_list) and ( ivic[i,j,distance] in appo_list))==False:
                    bond_hopping_difference_sign = bond_hopping_difference_sign.at[i,ivic[i,j,distance]].set(1.)
                    bond_hopping_difference_sign = bond_hopping_difference_sign.at[ivic[i,j,distance],i].set(-1.)
                    appo_list.append(i)
                    appo_list.append(ivic[i,j,distance])


# May be useful to break rotational symmetries inside Jastrow factors

two_Lx = Lx
n_distances_1D_x = Lx//2

# imulti_1D_x and ivic_1D_x
imulti_1D_x = 2*jnp.ones(Lx//2).astype(int) # In an even linear chain you have 2 neighbours at each distance < Lx/2, one at Lx/2
imulti_1D_x = imulti_1D_x.at[-1].set(1)
imulti_1D_x = tuple(imulti_1D_x)

ivic_1D_x = jnp.zeros((Lx,max(imulti_1D_x),n_distances_1D_x)).astype(int) #Even linear chain: Lx/2 independent distances!

for i in range(Lx):
    ivic_1D_x = ivic_1D_x.at[i,0,-1].set( (i+Lx//2)%Lx )
    for k in range(Lx//2-1):
        ivic_1D_x = ivic_1D_x.at[i,0,k].set( (i+k+1)%Lx )
        ivic_1D_x = ivic_1D_x.at[i,1,k].set( (i-k-1)%Lx )

distances_1D_x=jnp.tile(jnp.arange(Lx,dtype=int),(Lx,1))
distances_1D_x=jnp.abs(distances_1D_x-jnp.transpose(distances_1D_x)-jnp.rint((distances_1D_x-jnp.transpose(distances_1D_x))/float(Lx))*Lx).astype(int)


two_Ly = Ly
n_distances_1D_y = Ly//2

# imulti_1D_y and ivic_1D_y
imulti_1D_y = 2*jnp.ones(Ly//2).astype(int) # In an even linear chain you have 2 neighbours at each distance < Ly/2, one at Ly/2
imulti_1D_y = imulti_1D_y.at[-1].set(1)
imulti_1D_y = tuple(imulti_1D_y)

ivic_1D_y = jnp.zeros((Ly,max(imulti_1D_y),n_distances_1D_y)).astype(int) #Even linear chain: Ly/2 independent distances!

for i in range(Ly):
    ivic_1D_y = ivic_1D_y.at[i,0,-1].set( (i+Ly//2)%Ly )
    for k in range(Ly//2-1):
        ivic_1D_y = ivic_1D_y.at[i,0,k].set( (i+k+1)%Ly )
        ivic_1D_y = ivic_1D_y.at[i,1,k].set( (i-k-1)%Ly )

distances_1D_y=jnp.tile(jnp.arange(Ly,dtype=int),(Ly,1))
distances_1D_y=jnp.abs(distances_1D_y-jnp.transpose(distances_1D_y)-jnp.rint((distances_1D_y-jnp.transpose(distances_1D_y))/float(Ly))*Ly).astype(int)

def which_site_given_xy(x_coord,y_coord):
    return y_coord + x_coord*Ly

