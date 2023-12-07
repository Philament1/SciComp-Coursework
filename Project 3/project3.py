"""Scientific Computation Project 3
Your CID here
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#use scipy as needed
import scipy
from scipy import sparse as sp
import time

#===== Code for Part 1=====#

def plot_field(lat,lon,u,time,levels=20):
    """
    Generate contour plot of u at particular time
    Use if/as needed
    Input:
    lat,lon: latitude and longitude arrays
    u: full array of wind speed data
    time: time at which wind speed will be plotted (index between 0 and 364)
    levels: number of contour levels in plot
    """
    plt.figure()
    plt.contourf(lon,lat,u[time,:,:],levels)
    plt.axis('equal')
    plt.grid()
    plt.xlabel('longitude')
    plt.ylabel('latitude')

    return None

def plot_field2(lat,lon,u, time_points,levels=20):
    fig, ax = plt.subplots(len(time_points), 1)
    
    for i, t in enumerate(time_points):
        ax[i].contourf(lon, lat, u[t,:,:], levels)
        ax[i].axis('equal')
        ax[i].grid()
        ax[i].set_xlabel('longitude')
        ax[i].set_ylabel('latitude')


def part1():#add input if needed
    """
    Code for part 1
    """ 

    #--- load data ---#
    d = np.load(r'Project 3\data1.npz')
    lat = d['lat'];lon = d['lon'];u=d['u']
    #-------------------------------------#

    #Add code here 

    L, M, N = u.shape # L = 365, M = 16, N = 144

    A = u.reshape(L, M * N).T
    A_bar = np.mean(A, axis=1)
    A -= A_bar[:, None]  #   A ((M x N) x L) is u unrolled into L columns of length M x N vectors

    U, S, WT = np.linalg.svd(A, full_matrices=False)     #   Getting U ((M x N) x L) and S (L)
    rank = S[S>1e-11].size
    print(f'rank A: {rank}')  

    k = rank      #  How many principal components?
    
    Atilde = np.dot(U[:, :k].T, A)     #   Atilde (k x N) our new variables 

    #   Reconstruction

    A_PCA = np.dot(U[:, :k], Atilde)     #   A_pca ((M x N) x L) reconstruction of A
    A_PCA += A_bar[:, None]
    utilde = A_PCA.T.reshape(L, M, N)

    #   PLOTS

    plot_field2(lat, lon, u, np.linspace(0, 364, 5, dtype=int))

    #   Singular values
    plt.figure()
    plt.semilogy(S[:-1])

    #   Spatial patterns
    # plt.figure()
    # plt.imshow(U[:,0].reshape((M, N)), cmap='bwr', interpolation='nearest')
    plot_field2(lat, lon, U.T.reshape((L, M, N)), np.arange(0, 5))

    #   Temporal trends
    plt.figure()
    plt.plot(Atilde[0])
    plt.plot(Atilde[1])

    #   Reconstruction
    plot_field2(lat, lon, utilde, np.linspace(0, 364, 5, dtype=int))

    #   PC2 vs PC1
    plt.figure()
    plt.scatter(Atilde[0], Atilde[1])
    
    # fig, axs = plt.subplots(3, 3, figsize = (15, 15))
    # for i, ax in enumerate(axs.reshape(-1)):
    #     ax.contourf(lon,lat,utilde[i,:,:], 20)
    #     ax.axis('equal')
    #     ax.grid()

    
    plt.show()

    return None #modify if needed

#===== Code for Part 2=====#
def part2(f,method=2):
    """
    Question 2.1 i)
    Input:
        f: m x n array
        method: 1 or 2, interpolation method to use
    Output:
        fI: interpolated data (using method)
    """

    m,n = f.shape
    fI = np.zeros((m-1,n)) #use/modify as needed

    if method==1:
        fI = 0.5*(f[:-1,:]+f[1:,:])
    else:
        #Coefficients for method 2
        alpha = 0.3
        a = 1.5
        b = 0.1
        
        #coefficients for near-boundary points
        a_bc,b_bc,c_bc,d_bc = (5/16,15/16,-5/16,1/16)

        #add code here

        #   Using sparse
        A = sp.diags([[alpha]*(m-3) + [0], 1, [0] + [alpha]*(m-3)], [-1,0,1]).toarray()

        diag0 = [a_bc]+[a/2]*(m-3)+[b_bc]
        diag1 = [b/2]*(m-3)+[c_bc]
        diag2 = [0]*(m-4)+[d_bc]

        B = sp.diags([diag2, diag1, diag0, diag0[::-1], diag1[::-1], diag2[::-1]], [-2, -1,0,1,2, 3], shape=(m-1,m)).toarray()

        fI = np.linalg.solve(A, B @ f)

    return fI #modify as needed

def part2_analyze():
    """
    Add input/output as needed
    """

    #----- Code for generating grid, use/modify/discard as needed ----#
    n,m = 100,100 #arbitrary grid sizes
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    xg,yg = np.meshgrid(x,y)
    dy = y[1]-y[0]
    yI = y[:-1]+dy/2 #grid for interpolated data
    #--------------------------------------------# 

    #add code here

    xIg, yIg = np.meshgrid(x, yI)

    def testing_and_plots(func):
        f = func(xg, yg)
        fI = func(xIg, yIg)

        t = time.time()
        fI1 = part2(f, method=1)
        t1 = time.time()-t

        t = time.time()
        fI2 = part2(f, method=2)
        t2 = time.time()-t

        print(t1)
        print(t2)

        error1 = abs(fI1-fI)
        RMSE1 = np.mean(error1**2)
        print(RMSE1)
        error2 = abs(fI2-fI)
        RMSE2 = np.mean(error2**2)
        print(RMSE2)

        fig, ax = plt.subplots(1, 3)
        im0 = ax[0].pcolormesh(xg, yg, f)
        im1 = ax[1].pcolormesh(xIg, yIg, error1)
        ax[2].pcolormesh(xIg, yIg, error2, vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])
        fig.colorbar(im0)
        fig.colorbar(im1, ax=[ax[1], ax[2]])
        plt.show()

    testing_and_plots(lambda x, y: np.sin(2*np.pi*x) * np.cos(2*np.pi*y))
    testing_and_plots(lambda x, y: np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + np.where(y < 0.5, 0, 0.5))
   
    #   Plots

    # fig, ax = plt.subplots(1, 4)
    # ax[0].pcolormesh(xg, yg, f1)
    # ax[1].pcolormesh(xIg, yIg, f1I)    
    # ax[2].pcolormesh(xIg, yIg, f1I1)
    # ax[3].pcolormesh(xIg, yIg, f1I2)  
    # plt.show()

    return None #modify as needed



#===== Code for Part 3=====#
def part3q1(y0,alpha,beta,b,c,tf=200,Nt=800,err=1e-6,method="RK45"):
    """
    Part 3 question 1
    Simulate system of 2n nonlinear ODEs

    Input:
    y0: Initial condition, size 2*n array
    alpha,beta,b,c: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x 2*n array containing y at
            each time step including the initial condition.
    """
    
    #Set up parameters, arrays

    n = y0.size//2
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,2*n))
    yarray[0,:] = y0


    def RHS(t,y):
        """
        Compute RHS of model
        """
        #add code here
        u = y[:n];v=y[n:]
        r2 = u**2+v**2
        nu = r2*u
        nv = r2*v
        cu = np.roll(u,1)+np.roll(u,-1)
        cv = np.roll(v,1)+np.roll(v,-1)

        dydt = alpha*y
        dydt[:n] += beta*(cu-b*cv)-nu+c*nv+b*(1-alpha)*v
        dydt[n:] += beta*(cv+b*cu)-nv-c*nu-b*(1-alpha)*u

        return dydt


    sol = solve_ivp(RHS, (tarray[0],tarray[-1]), y0, t_eval=tarray, method=method,atol=err,rtol=err)
    yarray = sol.y.T 
    return tarray,yarray


def part3_analyze(display = False):#add/remove input variables if needed
    """
    Part 3 question 1: Analyze dynamics generated by system of ODEs
    """

    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0 = np.zeros(2*n)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #---Example code for computing solution, use/modify/discard as needed---#
    c = 1.3
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    u,v = y[:,:n],y[:,n:]

    if display:
        plt.figure()
        plt.contourf(np.arange(n),t,u,20)


    #-------------------------------------------#

    #Add code here

    # FFT analysis

    Nt_1 = len(u)

    Sf = np.fft.fftshift(np.fft.fft(u, axis=0))
    f = np.fft.fftshift(np.fft.fftfreq(Nt_1, t[1]-t[0]))

    # Plots
    
    plt.figure()
    for i in range(n):
        plt.plot(t, u[:,i])

    plt.figure()
    Nt_1 = len(u)
    for j in range(Nt_1):
        plt.plot(np.arange(n), u[j])

    plt.figure()
    plt.plot(f, Sf[:, 0])

    plt.show()


    return None #modify if needed


def part3q2(x,c=1.0):

    """
    Code for part 3, question 2
    """
    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #Compute solution
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    A = y[:,:n]

    #Analyze code here
    l1,v1 = np.linalg.eigh(A.T.dot(A))
    v2 = A.dot(v1)
    A2 = (v2[:,:x]).dot((v1[:,:x]).T)
    e = np.sum((A2.real-A)**2)

    return A2.real,e


if __name__=='__main__':
    x=None #Included so file can be imported
    #Add code here to call functions above if needed

    part2_analyze()