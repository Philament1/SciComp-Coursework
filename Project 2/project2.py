"""
Code for Scientific Computation Project 2
Please add college id here
CID:
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

    G = G.copy()        # Added

    Fdict = {}
    Mdict = {}
    Mlist = []
    dmin = float('inf')
    n = len(G)
    G.add_node(n)
    heapq.heappush(Mlist,[1,s])
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
    beta = 0.04/np.pi**2
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
    
    #Set up parameters, arrays
    n = y0.size
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,n))
    yarray[0,:] = y0
    beta = 0.04/np.pi**2
    alpha = 1-2*beta
    

    return tarray,yarray


def part2q2(): #add input variables if needed
    """
    Add code used for part 2 question 2.
    Code to load initial conditions is included below
    """

    data = np.load('project2.npy') #modify/discard as needed
    y0A = data[0,:] #first initial condition
    y0B = data[1,:] #second initial condition

    #Add code here


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
        dydt = np.array([0,0,0])
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
  
    #add code for generating figures and any other relevant calculations here

    return None #modify as needed
