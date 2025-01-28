import numpy as np
from numba import jit

def read_lattice(file_text, small=1.0e-10):
    enne=np.zeros((2,2))
    small=1.0e-10

    a = np.zeros((2,2)) #Primitive vectors
    b = np.zeros((2,2)) #Reciprocal vectors, see https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d


    ########## Reading part ##########

    # Actually read from file
    f = open(file_text, "r")
    lines=f.readlines()


    listWords = (lines[0]).split(" ")
    oper1 = listWords[0]
    a1    = float(listWords[1])
    oper2 = listWords[2]
    a2    = float(listWords[3])
    phi   = float(listWords[4])


    if (oper1=="'sqrt'"):
        a1=np.sqrt(a1)
    elif (oper1!="''"):
        print("wrong operator")
        exit()
    if (oper2=="'sqrt'"):
        a2=np.sqrt(a2)
    elif (oper2!="''"):
        print("wrong operator")
        exit()

    phi=phi*np.pi/180.

    a[0,0] = a1
    a[1,0] = 0.
    a[0,1] = a2*np.cos(phi)
    a[1,1] = a2*np.sin(phi)
    b=np.linalg.inv(a)

    nbase=int(lines[1][0]) 

    xb=np.zeros(nbase)
    yb=np.zeros(nbase)

    if nbase != 1:
        for i in range(nbase):
            listWords = (lines[2+i]).split(" ")
            oper3 = listWords[0]
            axx    = float(listWords[1])
            oper4 = listWords[2]
            ayy    = float(listWords[3])
            if (oper3=="'sqrt'"):
                axx=np.sqrt(axx)
            if (oper4=="'sqrt'"):
                ayy=np.sqrt(ayy)
            xb[i]=a[0,0]*axx+a[0,1]*ayy
            yb[i]=a[1,0]*axx+a[1,1]*ayy

    listWords = (lines[-1]).split(" ")
    enne[0,0]=float(listWords[0])   
    enne[0,1]=float(listWords[1])   
    enne[1,0]=float(listWords[2])   
    enne[1,1]=float(listWords[3])   

    if (enne[0,0]<=0. or enne[1,1]<=0.):
        print("box positioning")
        exit()

    if np.abs(np.linalg.det(enne))-int(np.abs(np.linalg.det(enne)))>1-small:
        nsite=int(np.abs(np.linalg.det(enne)))+1
    else:
        nsite=int(np.abs(np.linalg.det(enne)))

    enne1=np.linalg.inv(enne) #  Inverse N matrix
    
    ########## End Reading part ##########

    return a, b, enne, enne1, nbase, nsite, xb, yb
   
@jit(nopython=True)
def cluster(a, b, enne, enne1, nbase, nsite, xb, yb, small=1.0e-10):
    
    ########## Cluster and distances part ##########

    amat = np.zeros((2,2))
    Abig = np.zeros((2,2))
    Bbig = np.zeros(2) 
    XBig = np.zeros(2) 

    ndim=0

    x  = np.zeros((nsite,nbase))
    y  = np.zeros((nsite,nbase))
    i1 = np.zeros(nsite)
    i2 = np.zeros(nsite)

    n1max=int(enne[0,0]+max(0,enne[1,0]))
    n1min=int(min(0,enne[1,0]))
    n2max=int(enne[1,1]+max(0,enne[0,1]))
    n2min=int(min(0,enne[0,1]))

    for i in range(2):
        for j in range(2):
            amat[i,j]=np.dot(a[:,i],a[:,j])


    for n1 in range(n1min,n1max+1):
        for n2 in range(n2min,n2max+1):
            for i in range(2):
                for j in range(2):
                    Abig[i,j]=np.dot(enne[j,:],amat[:,i])
                Bbig[i]=n1*amat[0,i]+n2*amat[1,i]
            XBig = np.linalg.solve(Abig, Bbig)
            iflag = False
            for i in range(2):
                if (np.abs(XBig[i])<small):
                    XBig[i] = 0.
                if (np.abs(XBig[i]-1)<small):
                    XBig[i] = 1.
                if (XBig[i]<0. or XBig[i]>=1.):
                    iflag = True
            if iflag == False:
                for j in range(nbase):
                    x[ndim,j]=xb[j]+float(n1)*a[0,0]+float(n2)*a[0,1]
                    y[ndim,j]=yb[j]+float(n1)*a[1,0]+float(n2)*a[1,1]
                i1[ndim]=n1
                i2[ndim]=n2
                ndim=ndim+1

    if(nsite!=ndim):
        print("wrong site number ",nsite," ",ndim)

    print("Nsite=",ndim)
    print("Nbase=",nbase)
    print("\nCLUSTER\n")
    
    # Distances

    ntotalsite=nsite*nbase

    dist = np.zeros((ntotalsite,ntotalsite))

    # print("\nDISTANCES\n")

    for it in range(ntotalsite):
        i=int(it/nbase)
        ii=it%nbase
        xi=x[i,ii]
        yi=y[i,ii]
        for jt in range(ntotalsite):
            j=int(jt/nbase)
            jj=jt%nbase
            xj=x[j,jj]
            yj=y[j,jj]

            xij=xi-xj
            yij=yi-yj
            d2=xij**2+yij**2
            dist[it,jt]=d2
            # The shortest distance may cross PBCs...
            for isg1 in range(-1,2,1):
                for isg2 in range(-1,2,1):
                    xt=xij+(isg1*enne[0,0]+isg2*enne[1,0])*a[0,0]+(isg1*enne[0,1]+isg2*enne[1,1])*a[0,1]
                    yt=yij+(isg1*enne[0,0]+isg2*enne[1,0])*a[1,0]+(isg1*enne[0,1]+isg2*enne[1,1])*a[1,1]
                    d2t=xt**2+yt**2
                    if (d2t<dist[it,jt]):
                        dist[it,jt]=d2t
            dist[it,jt]=np.sqrt(dist[it,jt])
    #         print(it,jt,dist[it,jt])

     ########## End Cluster and distances part ##########

    return x, y, i1, i2, dist
    
def indip_distances(dist, small=1.0e-10):
    #  Independent distances

#     dindip=np.unique(dist.round(decimals=-int(np.log10(small))+1))[1:] #the [1:] at the end is to discard the distance 0., the round function is essential!
#     nind=dindip.size
    
    nind_previous=-1
    nind=-10
    
    dec=-int(np.log10(small))+2
    while nind!=nind_previous:
        nind_previous=nind
        dindip=np.unique(dist.round(decimals=dec))[1:] #the [1:] at the end is to discard the distance 0., the round function is essential!
        nind=dindip.size
        dec=dec-1

    print("\nTOTAL INDEPENDENT DISTANCES = "+str(nind)+"\n")

    for i in range(nind):
        print(i+1,dindip[i])
        
    dist_matrix = np.empty_like(dist)
    
    for it in range(np.shape(dist)[0]):
        for jt in range(np.shape(dist)[0]):
            if it==jt:
                dist_matrix[it,jt] = 0
            else:
                for ind_dist in range(len(dindip)):
                    if np.abs(dist[it,jt]-dindip[ind_dist])<small:
                        dist_matrix[it,jt] = ind_dist+1

    return dindip, dist_matrix

@jit(nopython=True)
def momenta_out(b, enne, enne1, nsite):
    #  Allowed momenta (in units of 2*pi)

    nmom=0
    n1max=int(enne[0,0]+max(0,enne[0,1]))
    n1min=int(min(0,enne[0,1]))
    n2max=int(enne[1,1]+max(0,enne[1,0]))
    n2min=int(min(0,enne[1,0]))

    uno=1.-small
    with open('momenta.out', 'w') as momenta_output:
        for n1 in range(n1min,n1max+1):
            for n2 in range(n2min,n2max+1):
                eig1=enne1[0,0]*n1+enne1[0,1]*n2
                eig2=enne1[1,0]*n1+enne1[1,1]*n2
                if (eig1>=0. and eig1<uno and eig2>=0. and eig2<uno):
                    qx=eig1*b[0,0]+eig2*b[0,1]
                    qy=eig1*b[1,0]+eig2*b[1,1]
                    nmom=nmom+1
                    momenta_output.write(str(qx)+"   "+str(qy)+"\n")

    print("\nIndipendent momenta=",nmom,"\n")      

    if(nmom!=ndim):
        print("Problems in determining the momenta")

    with open('positions_lattice.out', 'w') as lattice_output:
        for i in range(ndim):
            lattice_output.write(str(x[i,0])+"   "+str(y[i,0])+"\n")
    
    return

@jit(nopython=True)
def table_neighbours(dist, dindip, ndim, nbase, nsite, i1, i2, enne1, small=1.0e-10):
    
    ########## Table of neighbours ##########   

    nindm=dindip.size
    ntotalsite=nsite*nbase
    imulti = np.zeros(nindm)
    imark  = np.zeros(ndim*nbase)

    for k in range(nindm):
        d=dindip[k]
        for i in range(ndim*nbase):
            imark[i]=0
            for j in range(ndim*nbase):
                dij=dist[i,j]
                if(np.abs(dij-d)<small):
                    imark[i]=imark[i]+1
        maxmulti=0   
        for i in range(ndim*nbase):
            maxmulti=max(maxmulti,imark[i])
        imulti[k]=maxmulti

    maxmulti=0   
    for k in range(nindm):
        maxmulti=max(maxmulti,imulti[k])
    maxmulti=int(maxmulti)

    ivic=np.zeros(((ntotalsite,maxmulti,nindm)))


    for k in range(nindm):
        d=dindip[k]
        for i in range(ndim*nbase):
            ii=0
            for j in range(ndim*nbase):
                dij=dist[i,j]
                if(np.abs(dij-d)<small):
                    ivic[i,ii,k]=j
                    ii=ii+1             
               
    ########################################################

    isymt=np.zeros((ntotalsite,ndim))  # table of translational symmetries

    for i in range(ndim): #sites
        for k in range(ndim): #translations
            ik1=int(i1[i])+int(i1[k])
            ik2=int(i2[i])+int(i2[k])
            iflag=0
            for l in range(ndim):
                l1=ik1-int(i1[l])
                l2=ik2-int(i2[l])
                test1=enne1[0,0]*l1+enne1[1,0]*l2
                test2=enne1[0,1]*l1+enne1[1,1]*l2
                if(np.abs(test1-float(int(test1)))<small and np.abs(test2-float(int(test2)))<small):
                    iflag=iflag+1
                    ik=l
            if(iflag!=1):
                print('wrong reduction')
                return
            for ii in range(nbase):
                j  =  i*nbase+ii
                jj = ik*nbase+ii
                isymt[j,k]=jj

    ivict=np.zeros((ntotalsite,maxmulti,nindm))
    imark=np.zeros(ndim)
    ikm=0
    iflagref=False

    if(nbase!=1): #  we do not use inversion symmetry
        for k in range(nindm):
            for i in range(nbase):
                for j in range(int(imulti[k])):
                    ii=int(np.abs(ivic[i,j,k]))
                    if(ii!=-1.): #????
                        for it in range(nsite): # traslations
                            m=isymt[i,it]
                            l=isymt[ii,it]
                            ivict[int(m),j,k]=l
    else: # we order neighbours using the inversion symmetry 
        for k in range(nindm):
            imark=np.zeros(ndim)
            jshift=int(imulti[k]/2)
            jj=-1
            for j in range(int(imulti[k])):
                i=int(np.abs(ivic[0,j,k]))
                if(imark[i]==0):
                    imark[i]=1 # mark the site
                    iflag=0
                    ik1=int(i1[i])-int(i1[0])
                    ik2=int(i2[i])-int(i2[0])
                    ikm1=-ik1+int(i1[0])
                    ikm2=-ik2+int(i2[0])
                    for l in range(ndim): #reduction
                        l1=ikm1-int(i1[l])
                        l2=ikm2-int(i2[l])
                        test1=enne1[0,0]*l1+enne1[1,0]*l2
                        test2=enne1[0,1]*l1+enne1[1,1]*l2
                        if(np.abs(test1-float(int(test1)))<small and np.abs(test2-float(int(test2)))<small):
                            iflag=iflag+1
                            ikm=l         # inverted site
                    if(iflag!=1):
                        print('wrong reduction')
                        return
                    if(i!=ikm): # there is an inverted site
                        jj=jj+1
                        iflagref=False
                        imark[ikm]=1  # ...and it is inverted
                    elif(i==ikm): # the inverted site is the site itself
                        jj=jj+1
                        iflagref=True
                    for it in range(nsite): # traslations
                        m=isymt[0,it]
                        l=isymt[i,it]
                        if(iflagref==False):
                            ivict[int(m),jj,k]=l
                            l=isymt[ikm,it]
                            ivict[int(m),jj+jshift,k]=l
                        elif(iflagref==True):
                            ivict[int(m),jj,k]=l

    ########## End Table of neighbours ########## 


    return ivic, ivict, maxmulti, imulti

def print_ivic(ivic, ndim, nbase, nindm, imulti):
    
    #In this table the order of neighbours for each site is given by the lexicographic order of the sites
    print('\nTable of neighbours\n')
    for k in range(nindm):
        print('At distance k=',k+1)
        for i in range(ndim*nbase):
            print(i+1,end='      ')
            for j in range(int(imulti[k])):
                print(ivic[i,j,k]+1,end='  ')
            print(' ')
    return
    
def print_ivict(ivict, ndim, nbase, nindm, imulti):
    
    #  In this table the order of neighbours for each site is given by the order of the first site
    print('\nOrdered Table of neighbours\n')
    for k in range(nindm):
        print('At distance k=',k+1)
        for i in range(ndim*nbase):
            print(i+1,end='      ')
            for j in range(int(imulti[k])):
                print(ivict[i,j,k]+1,end='  ')
            print(' ')
    return

def info_out(nsite, nbase, nindm, maxmulti, imulti):
    ########## Write Information ##########       
    
    with open('info.out', 'w') as info_output:
        info_output.write(str(nsite)+"   "+str(nbase)+"   "+str(nindm)+"   "+str(maxmulti)+"\n")
        info_output.write(str(imulti)+"\n\n")
    #     info_output.write(str(isymt)+"\n")
    #     info_output.write(str(ivict)+"\n")

    ########## End Write Information ##########    
    
    return

    
import time

begin = time.time()

#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
#%#%#%                                                                              #%#%#% 
#%#%#%     ENTRIES                                                                  #%#%#%    
#%#%#%     a1=modulus of the first primitive vector                                 #%#%#% 
#%#%#%     a2=modulus of the second primitive vector                                #%#%#% 
#%#%#%     phi=angle between the two primitive vectors                              #%#%#% 
#%#%#%                                                                              #%#%#% 
#%#%#%     enne[i,j] is a 2x2 integer matrix defining the two translations which    #%#%#% 
#%#%#%     identify the cluster:                                                    #%#%#% 
#%#%#%     T_i = sum_j enne[i,j]*a_j                                                #%#%#% 
#%#%#%     Namely:                                                                  #%#%#% 
#%#%#%     T_1 = n_{0,0}*a_1 + n_{0,1}*a_2                                          #%#%#% 
#%#%#%     T_2 = n_{1,0}*a_1 + n_{1,1}*a_2                                          #%#%#% 
#%#%#%     We assume that the leading components are enne[0,0]>0 and enne[1,1]>0    #%#%#% 
#%#%#%                                                                              #%#%#% 
#%#%#%     We assume that the basis is given in terms of the components with        #%#%#% 
#%#%#%     respect to the primitive vectors.                                        #%#%#% 
#%#%#%     For example for graphene it will be given by:                            #%#%#% 
#%#%#%     x_0 = a_1*0 + a_2*0  ; x_1 = a_1*1/3 + a_2*1/3                           #%#%#% 
#%#%#%                                                                              #%#%#% 
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

epsilon=1.0e-10

a, b, enne, enne1, nbase, nsite, xb, yb = read_lattice(file_text="geometry.d", small=epsilon)
# a, b, enne, enne1, nbase, nsite, xb, yb = read_lattice(file_text="triangular_geometry.d", small=epsilon)

x, y, i1, i2, dist = cluster(a, b, enne, enne1, nbase, nsite, xb, yb, small=epsilon)

dindip, dist_matrix = indip_distances(dist, small=epsilon)

ivic, ivict, maxmulti, imulti  = table_neighbours(dist, dindip, nsite, nbase, nsite, i1, i2, enne1, small=epsilon)

#print_ivict(ivict, nsite, nbase, dindip.size, imulti)

#info_out(nsite, nbase, dindip.size, maxmulti, imulti)

print("\n")
print(time.time()-begin)
print("Elapsed time: ",time.time()-begin,"\n\n")



ivict = ivict.astype(int)
dist_matrix = dist_matrix.astype(int)
imulti = imulti.astype(int)

with open("../lattice.npy", "wb") as file:
    np.save(file,nbase*nsite)
    np.save(file,2*nbase*nsite)
    np.save(file,len(dindip))
    np.save(file,ivict)
    np.save(file,dist_matrix)
    np.save(file,imulti)


print("File \"lattice.npy\" edited \n\n")
