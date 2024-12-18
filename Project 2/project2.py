"""
Code for Scientific Computation Project 2
Please add college id here
CID: 02027072
"""
import heapq
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#use scipy in part 2 as needed

#===== Codes for Part 1=====#
def searchGPT(graph, source, target):
    # Initialize distances with infinity for all nodes except the source
    distances = {node: float('inf') for node in graph}
    distances[source] = 0

    # Initialize a priority queue to keep track of nodes to explore
    priority_queue = [(0, source)]  # (distance, node)

    # Initialize a dictionary to store the parent node of each node in the shortest path
    parents = {}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # If the current node is the target, reconstruct the path and return it
        if current_node == target:
            path = []
            while current_node in parents:
                path.insert(0, current_node)
                current_node = parents[current_node]
            path.insert(0, source)
            return current_distance,path

        # If the current distance is greater than the known distance, skip this node
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = max(distances[current_node], weight['weight'])
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parents[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return float('inf')  # No path exists


def searchPKR(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    """

    Fdict = {}
    Mdict = {}
    Mlist = []
    dmin = float('inf')
    n = len(G)
    G.add_node(n)
    heapq.heappush(Mlist,[0,s])
    Mdict[s]=Mlist[0]
    found = False

    while len(Mlist)>0:
        dmin,nmin = heapq.heappop(Mlist)
        if nmin == x:
            found = True
            break

        Fdict[nmin] = dmin

        for m,en,wn in G.edges(nmin,data='weight'):
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = max(dmin,wn)
                if dcomp<Mdict[en][0]:
                    l = Mdict.pop(en)
                    l[1] = n
                    lnew = [dcomp,en]
                    heapq.heappush(Mlist,lnew)
                    Mdict[en]=lnew
            else:
                dcomp = max(dmin,wn)
                lnew = [dcomp,en]
                heapq.heappush(Mlist, lnew)
                Mdict[en] = lnew

    return dmin


def searchPKR2(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    Output: Should produce equivalent output to searchGPT given same input
    """

    #Add code here

    #return dmin and list

    Fdict = {}
    Mdict = {}
    Mlist = []
    dmin = float('inf')
    n = len(G)
    G.add_node(n)
    heapq.heappush(Mlist,[0,s])
    Mdict[s]=Mlist[0]
    parents = {}    #   Added parents dictionary
    found = False

    while len(Mlist)>0:
        dmin,nmin = heapq.heappop(Mlist)
        print(f"Node:", nmin)
        if nmin == x:
            found = True
            break

        Fdict[nmin] = dmin

        for m,en,wn in G.edges(nmin,data='weight'):
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = max(dmin,wn)
                if dcomp<Mdict[en][0]:
                    l = Mdict.pop(en)
                    l[1] = n
                    lnew = [dcomp,en]
                    heapq.heappush(Mlist,lnew)
                    Mdict[en]=lnew
                    parents[en] = nmin  #   Update parent
            else:
                dcomp = max(dmin,wn)
                lnew = [dcomp,en]
                heapq.heappush(Mlist, lnew)
                Mdict[en] = lnew
                parents[en] = nmin  #   Introduce parent

    path = []   #   Instantiate return path
    if found:   #   Only run if target is found
        m = x   #   Begin from target node
        while m != s:   #   Iterate until source node is appended
            path.append(m)  #   Add parent to list
            m = parents[m]
        path.append(s)
        path.reverse()  #   Reverse list

    return dmin, path   #   Return dmin, path


#===== Code for Part 2=====#
def part2q1(y0,tf=1,Nt=5000):
    """
    Part 2, question 1
    Simulate system of n nonlinear ODEs

    Input:
    y0: Initial condition, size n array
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x n array containing y at
            each time step including the initial condition.
    """
    import numpy as np
    
    #Set up parameters, arrays
    n = y0.size
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,n))
    yarray[0,:] = y0
    beta = 10000/np.pi**2
    alpha = 1-2*beta
    
    def RHS(t,y):
        """
        Compute RHS of model
        """        
        dydt = np.zeros_like(y)
        for i in range(1,n-1):
            dydt[i] = alpha*y[i]-y[i]**3 + beta*(y[i+1]+y[i-1])
        
        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[-1])
        dydt[-1] = alpha*y[-1]-y[-1]**3 + beta*(y[0]+y[-2])

        return dydt 


    #Compute numerical solutions
    dt = tarray[1]
    for i in range(Nt):
        yarray[i+1,:] = yarray[i,:]+dt*RHS(0,yarray[i,:])

    return tarray,yarray

def part2q1new(y0,tf=40,Nt=800):
    """
    Part 2, question 1
    Simulate system of n nonlinear ODEs

    Input:
    y0: Initial condition, size n array
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x n array containing y at
            each time step including the initial condition.
    """
    from scipy.integrate import solve_ivp
    from scipy import sparse
    
    #Set up parameters, arrays
    n = y0.size
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,n))
    yarray[0,:] = y0
    beta = 10000/np.pi**2
    alpha = 1-2*beta

    A = sparse.diags([beta, [beta]*(n-1), [alpha]*n, [beta]*(n-1), beta], [-(n-1), -1, 0, 1, n-1])  #   Sparse matrix for linear part of ODE

    def RHS(t,y):
        """
        Compute RHS of model
        """        
        dydt = A @ y - y**3     #   Combining linear part with cubic component

        return dydt 
    
    sol = solve_ivp(RHS, [tarray[0],tarray[-1]], y0, method='BDF', t_eval=tarray, vectorized=True)  #   Efficiently computed solution
    yarray = np.transpose(sol.y)    #   Transposed to match the return in part2q1

    return tarray,yarray


def part2q2(): #add input variables if needed
    """
    Add code used for part 2 question 2.
    Code to load initial conditions is included below
    """

    data = np.load(r'Project 2\project2.npy') #modify/discard as needed
    y0A = data[0,:] #first initial condition
    y0B = data[1,:] #second initial condition

    #Add code here

    import scipy

    n = len(y0A)
    t_max = 40
    Nt = 600

    #   Finding solutions to IVP
    tA, yA = part2q1new(y0A, t_max, Nt)
    tB, yB = part2q1new(y0B, t_max, Nt)

    #   Finding equilibrium points close to IVP
    beta = 10000/np.pi**2
    alpha = 1-2*beta
    
    def RHS(y):
        """
        Compute RHS of model
        """        
        dydt = np.zeros_like(y)
        dydt[1:-1] = alpha*y[1:-1]-y[1:-1]**3 + beta*(y[2:]+y[:-2])
        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[-1])
        dydt[-1] = alpha*y[-1]-y[-1]**3 + beta*(y[0]+y[-2])

        return dydt 

    solA = scipy.optimize.root(RHS, y0A)
    yA_eq = solA.x

    solB = scipy.optimize.root(RHS, y0B)
    yB_eq = solB.x

    #   Perturbation analysis
    def get_pert_sol(eq, t, init):
        """
        Get the general solution of RHS for perturbations 
        """
        M = scipy.sparse.diags([alpha - 3*eq**2] + [beta]*4, [0, 1, -1, n-1, -(n-1)])

        l, v = np.linalg.eig(M.toarray())
        c = scipy.linalg.solve(v, init)

        print(np.max(l))
        print(np.min(l))

        sol = np.exp(np.outer(t, l)) @ (v * c[:None]).T

        return sol

    yA_sol = get_pert_sol(yA_eq, tA, y0A)
    yB_sol = get_pert_sol(yB_eq, tB, y0B)

    #   Plots and figures

    ##  IVP plots for A
    fig, ax = plt.subplots(1, 2)

    ax[0].set_xlabel('t')
    ax[0].set_ylabel('y(t)')
    for k in range(n):
        ax[0].plot(tA, yA[:,k])

    ax[1].set_xlabel('index / k')
    ax[1].set_ylabel('y[k]')
    for t_step in range(Nt+1):
        ax[1].plot(range(n), yA[t_step])

    ##  Perturbation plots for A
    fig, ax = plt.subplots(1, 2)

    ax[0].set_xlabel('t')
    ax[0].set_ylabel('y(t)')
    for k in range(n):
        ax[0].plot(tA, yA_sol[:,k])

    ax[1].set_xlabel('index / k')
    ax[1].set_ylabel('y[k]')
    for t_step in range(Nt+1):
        ax[1].plot(range(n), yA_sol[t_step])
    
    ##  IVP plots for B
    fig, ax = plt.subplots(1, 2)

    ax[0].set_xlabel('t')
    ax[0].set_ylabel('y(t)')
    for k in range(n):
        ax[0].plot(tB, yB[:,k])

    ax[1].set_xlabel('index / k')
    ax[1].set_ylabel('y[k]')
    for t_step in range(Nt+1):
        ax[1].plot(range(n), yB[t_step])

    ##  Perturbation plots for B
    fig, ax = plt.subplots(1, 2)

    ax[0].set_xlabel('t')
    ax[0].set_ylabel('y(t)')
    for k in range(n):
        ax[0].plot(tB, yB_sol[:,k])

    ax[1].set_xlabel('index / k')
    ax[1].set_ylabel('y[k]')
    for t_step in range(Nt+1):
        ax[1].plot(range(n), yB_sol[t_step])

    plt.show()

    return None #modify as needed


def part2q3(tf=10,Nt=1000,mu=0.2,seed=1):
    """
    Input:
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf
    mu: model parameter
    seed: ensures same random numbers are generated with each simulation

    Output:
    tarray: size Nt+1 array
    X size n x Nt+1 array containing solution
    """

    #Set initial condition
    y0 = np.array([0.3,0.4,0.5])
    np.random.seed(seed)
    n = y0.size #must be n=3
    Y = np.zeros((Nt+1,n)) #may require substantial memory if Nt, m, and n are all very large
    Y[0,:] = y0

    Dt = tf/Nt
    tarray = np.linspace(0,tf,Nt+1)
    beta = 0.04/np.pi**2
    alpha = 1-2*beta

    def RHS(t,y):
        """
        Compute RHS of model
        """
        dydt = np.array([0.,0.,0.])
        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[2])
        dydt[1] = alpha*y[1]-y[1]**3 + beta*(y[0]+y[2])
        dydt[2] = alpha*y[2]-y[2]**3 + beta*(y[0]+y[1])

        return dydt 

    dW= np.sqrt(Dt)*np.random.normal(size=(Nt,n))

    #Iterate over Nt time steps
    for j in range(Nt):
        y = Y[j,:]
        F = RHS(0,y)
        Y[j+1,0] = y[0]+Dt*F[0]+mu*dW[j,0]
        Y[j+1,1] = y[1]+Dt*F[1]+mu*dW[j,1]
        Y[j+1,2] = y[2]+Dt*F[2]+mu*dW[j,2]

    return tarray,Y


def part2q3Analyze(): #add input variables as needed
    """
    Code for part 2, question 3
    """

    N_sim = 1000

    #   Analysis for upper plot
    mus = [0.2, 0.8]

    ##  mu=0
    t, y0 = part2q3(mu = 0)
    
    ##  mu=0.2
    y_all1 = []
    for j in range(N_sim):  #   Running 1000 simulations and averaging
        t, y = part2q3(mu = mus[0], seed=j)
        y_all1.append(y)

    y_all1 = np.array(y_all1)
    y_bar1 = np.mean(y_all1, axis=0)

    ##  mu=0.8
    y_all2 = []             #   Running 1000 simulations and averaging
    for j in range(N_sim):
        t, y = part2q3(mu = mus[1], seed=j)
        y_all2.append(y)

    y_all2 = np.array(y_all2)
    y_bar2 = np.mean(y_all2, axis=0)

    #   Analysis for lower plot
    mus = np.linspace(-1, 1, 21)    #   Range of mus
    final_ys = []

    for mu in mus:
        final_y = []
        if mu != 0:     
            for j in range(N_sim):  #   Running 1000 simulations and averaging
                t, y = part2q3(mu = mu, seed=j)
                final_y.append(y[-1,:])
            final_y = np.array(final_y)
            final_y = np.mean(final_y, axis=0)
        else:           #   If mu is 0, we only need to run 1 solution as there is no Brownian motion
            t, y = part2q3(mu = mu)
            final_y = y[-1,:]
        final_ys.append(final_y)

    final_ys = np.array(final_ys)

    #   Plots
    fig, ax = plt.subplots(2, 1)

    ax[0].set_title('Y against time')
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('Y(t)')
    for k in range(3):
        ax[0].plot(t, y0[:,k], color = 'red', label='mu=0' if k==0 else '')
        ax[0].plot(t, y_bar1[:,k], color = 'blue', label='mu=0.2' if k==0 else '')
        ax[0].plot(t, y_bar2[:,k], color = 'green', label='mu=0.8'  if k==0 else '')
    ax[0].legend()

    ax[1].set_title('Y(t=10) against mu')
    ax[1].set_xlabel('mu')
    ax[1].set_ylabel('Y(t=10)')
    ax[1].plot(mus, final_ys[:,0], color = 'orange', label='y0_0 = 0.3')
    ax[1].plot(mus, final_ys[:,1], color = 'purple', label='y0_1 = 0.4')
    ax[1].plot(mus, final_ys[:,2], color = 'olive', label='y0_2 = 0.5')
    ax[1].legend()
    ax[1].set_xticks(np.linspace(-1, 1, 21))

    plt.tight_layout()
    plt.show()
    
    return None #modify as needed
