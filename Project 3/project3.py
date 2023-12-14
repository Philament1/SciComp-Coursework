"""Scientific Computation Project 3
Your CID here
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#use scipy as needed
import scipy
from scipy import sparse as sp
from scipy import fft
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
    # plt.figure()
    # plt.contourf(lon,lat,u[time,:,:],levels)
    # plt.axis('equal')
    # plt.grid()
    # plt.xlabel('longitude')
    # plt.ylabel('latitude')

    plt.figure()
    contour = plt.contourf(lon,lat,u[time,:,:],levels)
    plt.axis('equal')
    plt.grid()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.colorbar(contour)

    return None

def plot_field2(lat,lon,u, time_points,levels=20):
    fig, ax = plt.subplots(len(time_points), 1)
    cmap = plt.cm.get_cmap('viridis')
    
    for i, t in enumerate(time_points):
        contour = ax[i].contourf(lon, lat, u[t,:,:], levels)
        ax[i].set_aspect('equal')
        ax[i].grid()
        ax[i].set_xlabel('longitude')
        ax[i].set_ylabel('latitude')

    fig.colorbar(contour, ax=ax, orientation='vertical')

def PCA(A, k=0):
    """
    Apply principal component analysis to A using SVD

    Input:
    A: p x N row-centered matrix, where p is number of features/attributes, N is number of data points
    k: number of principal components (default=0 uses rank of A)
    Output:
    PC: k x p matrix, each row is a principal component
    S: length k array, singular values
    Atilde: k x N matrix, transformed data
    """
    U, S, WT = np.linalg.svd(A, full_matrices=False)  # Find singular values and principal components using SVD
    rank = S[S>1e-11].size  # Find rank of A by checking number of non-zero singular values
    print(f'rank A: {rank}')  

    if k == 0:  # If k is not specified, we use the rank of A
        k = rank

    PC = U[:, :k].T  # Principal components

    Atilde = PC @ A  # Transformed data
    return S[:k], PC, Atilde

def part1(time_as_datapoints = True):#add input if needed
    """
    Code for part 1
    """ 

    #--- load data ---#
    d = np.load(r'Project 3\data1.npz')
    lat = d['lat'];lon = d['lon'];u=d['u']
    #-------------------------------------#

    #Add code here 

    #   PLOTS
    plot_field2(lat, lon, u, np.linspace(0, 364, 5, dtype=int))

    L, M, N = u.shape # L = 365, M = 16, N = 144
    
    if time_as_datapoints:
        '''CASE 1: TIME AS DATA POINTS'''

        A = u.reshape(L, M * N).T  # A ((M x N) x L) is u unrolled into L columns of length M x N vectors
        A_bar = np.mean(A, axis=1)
        A -= A_bar[:, None]   # Centering the rows of A
        k = L
        S, PC, Atilde = PCA(A, k)

        # #   Reconstruction
        # A_red = PC.T @ Atilde     #   A_red ((M x N) x L) reconstruction of A
        # A_red += A_bar[:, None]
        # utilde = A_red.T.reshape(L, M, N)
        # plot_field2(lat, lon, utilde, np.linspace(0, 364, 5, dtype=int))

        #   Singular values
        plt.figure()
        plt.title('Singular values')
        plt.semilogy(S)
        plt.ylabel('Singular value')
        plt.xlabel('Principal Component')

        #   Explained variance
        cum_var_rat = np.cumsum(S**2/np.sum(S**2))

        plt.figure()
        plt.plot(cum_var_rat[:50])
        plt.axhline(y=1, color='r', label='Total Variance')
        plt.axhline(y=0.8, color='k', linestyle='--')
        plt.ylabel('% explained variance')
        plt.xlabel('Principal Component')
        plt.legend()

        # Principal components (along space)
        fig, ax = plt.subplots(3, 1)
        for i in range(3):
            contour = ax[i].contourf(lon, lat, PC.reshape((k, M, N))[i,:,:], 20)
            ax[i].set_aspect('equal')
            ax[i].grid()
            ax[i].set_title(f'PC{i+1}')
            ax[i].set_xlabel('longitude')
            ax[i].set_ylabel('latitude')
        fig.colorbar(contour, ax=ax, orientation='vertical')

        # plot_field(lat, lon, PC.reshape((k, M, N)), 0)

        #   Temporal trends (plotting rows of Atilde)
        plt.figure()
        plt.plot(Atilde[0], label = 'PC1')
        plt.plot(Atilde[1], label = 'PC2')
        plt.xlabel('Day')
        plt.ylabel('Projection along PC')
        plt.legend()

        #  Correlation between subsequent days
        plt.figure()
        plt.scatter(Atilde[0][:-1], Atilde[0][1:], marker='x', label='Projection along PC1')
        plt.scatter(Atilde[1][:-1], Atilde[1][1:], label='Projection along PC2')
        plt.xlabel('Value at day t')
        plt.ylabel('Value at day t+1')
        plt.legend()


        #   Fourier on temporal trends
        plt.figure()
        #  mode = np.arange(-L//2, L//2)

        window = np.hanning(len(Atilde[0]))
        Atilde_windowed = Atilde[0] * window

        spect1 = abs(fft.fftshift(fft.fft(Atilde_windowed)))
        freq = fft.fftshift(fft.fftfreq(L))
        plt.semilogy(freq[L//2:], spect1[L//2:], label='FFT')

        # plt.figure()
        freq, welch1 = scipy.signal.welch(Atilde[0])
        plt.semilogy(freq, welch1, label='Welch')
        plt.xlabel('f')
        plt.ylabel('spectral density')
        plt.ylim(1, 10**6)
        plt.legend()

        # # PC2 vs PC1
        # plt.figure()
        # plt.scatter(Atilde[0], Atilde[1])

        plt.show()
    
    else:
        '''CASE 2: LOCATIONS AS DATA POINTS'''

        A = u.reshape(L, M * N)  # A (L x (M x N)) is u unrolled into M x N columns of length L vectors

        A_bar = np.mean(A, axis=1)
        A -= A_bar[:, None]   # Centering the rows of A
        S, PC, Atilde = PCA(A)

        #   Singular values
        plt.figure()
        plt.title('Singular values')
        plt.semilogy(S)
        plt.ylabel('Singular value')
        plt.xlabel('Principal Component')

        #   Explained variance
        cum_var_rat = np.cumsum(S**2/np.sum(S**2))

        plt.figure()
        plt.plot(cum_var_rat[:50])
        plt.axhline(y=1, color='r')
        plt.axhline(y=0.8, color='k', linestyle='--')
        plt.ylabel('% explained variance')
        plt.xlabel('Principal Component')

        #   Principal components (along time)
        plt.figure()
        plt.plot(PC[0], label = 'PC1')
        plt.plot(PC[1], label = 'PC2')
        plt.xlabel('Day')
        plt.legend()

        #   Fourier on PCs
        plt.figure()
        mode = np.arange(-L//2, L//2)

        window = np.hanning(len(PC[0]))
        PC_windowed = PC[0] * window

        spect1 = abs(fft.fftshift(fft.fft(PC_windowed)))
        freq = fft.fftshift(fft.fftfreq(L))
        plt.semilogy(freq[L//2:], spect1[L//2:], label='FFT')

        # plt.figure()
        freq, welch1 = scipy.signal.welch(PC[0])
        plt.semilogy(freq, welch1, label='Welch')
        plt.xlabel('f')
        plt.ylabel('spectral density')
        #  plt.ylim(1, 10**6)
        plt.legend()

        #   Spatial patterns (plotting rows of Atilde)
        # plot_field2(lat, lon, Atilde.reshape((L, M, N)), np.arange(0,3))

        fig, ax = plt.subplots(2, 1)
        cmap = plt.cm.get_cmap('viridis')
        
        for i in range(2):
            contour = ax[i].contourf(lon, lat, Atilde.reshape((L, M, N))[i,:,:], 20)
            ax[i].set_aspect('equal')
            ax[i].grid()
            ax[i].set_title(f'Projection along PC{i+1}')
            ax[i].set_xlabel('longitude')
            ax[i].set_ylabel('latitude')
        fig.colorbar(contour, ax=ax, orientation='vertical')

        # #   PC1 vs PC2
        # plt.figure()
        # plt.scatter(Atilde[0], Atilde[1])

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

        #  LHS matrix in diagonal ordered form
        AB = np.vstack([np.full(m-1, alpha), np.ones(m-1), np.full(m-1, alpha)])
        AB[0, 0] = AB[0, 1] = 0
        AB[-1, -1] = AB[-1, -2] = 0

        #  RHS rectangular banded matrix
        diag0 = [a_bc]+[a/2]*(m-3)+[b_bc]
        diag1 = [b/2]*(m-3)+[c_bc]
        diag2 = [0]*(m-4)+[d_bc]
        B = sp.diags([diag2, diag1, diag0, diag0[::-1], diag1[::-1], diag2[::-1]], [-2, -1,0,1,2, 3], shape=(m-1,m)).toarray()

        fI = scipy.linalg.solve_banded((1,1), AB, B @ f)  # Solving linear equations

    return fI #modify as needed

def part2_analyze():
    """
    Add input/output as needed
    """

    #----- Code for generating grid, use/modify/discard as needed ----#
    n,m = 50,40 #arbitrary grid sizes
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    xg,yg = np.meshgrid(x,y)
    dy = y[1]-y[0]
    yI = y[:-1]+dy/2 #grid for interpolated data
    #--------------------------------------------# 

    #add code here

    #   Meshgrid for interpolated values
    xIg, yIg = np.meshgrid(x, yI)

    def testing_and_plots(func):
        #   Applying function to meshgrids
        f = func(xg, yg)
        fI = func(xIg, yIg)

        #   Running and timing
        N = 1000
        t1 = 0
        t2 = 0
        for i in range(N):
            t = time.time()
            fI1 = part2(f, method=1)
            t1 += time.time()-t

            t = time.time()
            fI2 = part2(f, method=2)
            t2 += time.time()-t
        t1 /= N
        t2 /= N
        print(f'Time1: {t1}')
        print(f'Time2: {t2}')

        #   Finding error
        error1 = abs(fI1-fI)
        mean_error1 = np.mean(error1)
        print(f'error1: {mean_error1}')
        error2 = abs(fI2-fI)
        mean_error2 = np.mean(error2)
        print(f'error2: {mean_error2}')

        #   Accuracy and error plots
        fig, ax = plt.subplots(1, 3)
        im0 = ax[0].pcolormesh(xg, yg, f)
        fig.colorbar(im0)
        ax[0].set_title('Function')

        im1 = ax[1].pcolormesh(xIg, yIg, error1)
        fig.colorbar(im1)
        ax[1].set_title('Method 1 interpolation error')

        im2 = ax[2].pcolormesh(xIg, yIg, error2)
        fig.colorbar(im2)
        ax[2].set_title('Method 2 interpolation error')

        plt.tight_layout()

    testing_and_plots(lambda x, y: np.sin(2*np.pi*x) * np.cos(2*np.pi*y))  # test with oscillating function
    testing_and_plots(lambda x, y: (3/4) * np.exp(-((9*x-2)**2)/4 - ((9*y-2)**2)/4) + \
                               (3/4) * np.exp(-((9*x+1)**2)/49 - ((9*y+1)**2)/10) + \
                               (1/2) * np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4) - \
                               (1/5) * np.exp(-((9*x-4)**2) - ((9*y-7)**2)))  # test with Franke's function
    
    #   Testing with different grid sizes

    fig, ax = plt.subplots(1, 2)
    m_list = np.arange(40, 240, 40, dtype='int')  # change as required
    error_list1 = []
    error_list2 = []
    for m in m_list:
        #----- Code for generating grid, use/modify/discard as needed ----#
        n = m + 10
        x = np.linspace(0,1,n)
        y = np.linspace(0,1,m)
        xg,yg = np.meshgrid(x,y)
        dy = y[1]-y[0]
        yI = y[:-1]+dy/2 #grid for interpolated data
        #--------------------------------------------# 
        xIg, yIg = np.meshgrid(x, yI)

        func = lambda x, y: (3/4) * np.exp(-((9*x-2)**2)/4 - ((9*y-2)**2)/4) + \
                               (3/4) * np.exp(-((9*x+1)**2)/49 - ((9*y+1)**2)/10) + \
                               (1/2) * np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4) - \
                               (1/5) * np.exp(-((9*x-4)**2) - ((9*y-7)**2))
        
        f = func(xg, yg)
        fI = func(xIg, yIg)
        
        fI1 = part2(f, method=1)
        fI2 = part2(f, method=2)

        error1 = abs(fI1-fI)
        error1 = np.mean(error1, axis=1)
        error2 = abs(fI2-fI)
        error2 = np.mean(error2, axis=1)
        error_list1.append(np.mean(error1))
        error_list2.append(np.mean(error2))

        ax[0].semilogy(yI, error1, label=f'm={m}')
        ax[0].set_ylim(10**-13, 10**-2)
        ax[0].set_xlabel('y')
        ax[0].set_ylabel('error')
        ax[0].set_title('Method 1')
        ax[0].grid(True)
        ax[1].semilogy(yI, error2, label=f'm={m}')
        ax[1].set_ylim(10**-13, 10**-2)
        ax[1].set_xlabel('y')
        ax[1].set_ylabel('error')
        ax[1].set_title('Method 2')
        ax[1].grid(True)

    fig.legend(loc='lower left')
    plt.tight_layout()
    
    plt.figure()
    plt.semilogy(m_list, error_list1, label='Method 1')
    plt.semilogy(m_list, error_list2, label='Method 2')
    plt.xlabel('m')
    plt.ylabel('error')
    plt.grid()
    plt.legend()

    plt.show()


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
    n = 1500 # default 4000

    tf = 200
    Nt = 800

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0 = np.zeros(2*n)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #---Example code for computing solution, use/modify/discard as needed---#
    c_vals = np.array([0.5, 1.2, 1.3, 1.5])
    u_list = []
    for c in c_vals:
        if c == 1.3:
            t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
            y0_ = y[-1,:]
            t,y = part3q1(y0_,alpha,beta,b,c,method='RK45',err=1e-6)
            u,v = y[:,:n],y[:,n:]
            u_list.append(u)
        else:
            t1,y1 = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
            y01 = y1[-1,:]
            t1,y1 = part3q1(y01,alpha,beta,b,c,method='RK45',err=1e-6)
            u1,v1 = y1[:,:n],y1[:,n:]
            u_list.append(u1)
        if display:
            plt.figure()
            plt.contourf(np.arange(n),t,u,20)


    #-------------------------------------------#

    #Add code here

    x = u[:,100:-99]
    Nt_1, m = x.shape
    times = np.linspace(0, tf, Nt_1)

    #  Time plot
    plt.figure()
    for i in range(m):
        plt.plot(t, x[:,i])

    #  Space plot
    plt.figure()
    for j in range(Nt_1):
        plt.plot(np.arange(100, n-99), x[j])

    #  FFT analysis

    Sf = np.fft.fftshift(np.fft.fft(x, axis=0))
    f = np.fft.fftshift(np.fft.fftfreq(Nt_1, t[1]-t[0]))

    plt.figure()
    plt.plot(f, Sf[:, 0])
    plt.figure()
    plt.semilogy(f, Sf[:, 0])

    #  Correlation dimension
    def corr_dim(u, c, logeps_min, logeps_max):
        D = scipy.spatial.distance.pdist(u[:,100:-99])
        eps = np.logspace(logeps_min-1, logeps_max+1, 30)
        C = np.array([D[D < ep].size*2/(m*(m-1)) for ep in eps])

        plt.figure()
        plt.loglog(eps, C, marker = 'x', label=r'C($\epsilon$)')
        
        eps_fit = np.logspace(logeps_min, logeps_max, 20)
        C_fit = np.array([D[D < ep].size*2/(n*(n-1)) for ep in eps_fit])
        d, p_corr = np.polyfit(np.log(eps_fit), np.log(C_fit), 1)
        print(d)

        plt.loglog(eps, np.exp(p_corr)*(eps**d), linestyle='--', label=f'least squares fit, slope={d}')
        plt.title(f'c = {c}')
        plt.legend()
        
    c_num = len(c_vals)
    logeps_min_vals = [-1, -1, -0.5, 0.5]
    logeps_max_vals = [2, 2, 2, 1.8]

    for i in range(c_num):
        corr_dim(u_list[i], c_vals[i], logeps_min_vals[i], logeps_max_vals[i])

    #  Oscillation mapping
    
    dx = np.diff(x[:,0])
    d2x  = dx[:-1]*dx[1:]
    ind = np.argwhere(d2x<0)
    
    plt.figure()
    plt.plot(x[:,0])


    plt.show()

    return None #modify if need

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
    y0 = np.zeros(2*n)
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

    # part1(time_as_datapoints=False)
    part2_analyze()
    # part3_analyze()
