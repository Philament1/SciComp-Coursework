"""
Code for Scientific Computation Project 1
Please add college id here
CID: 02027072
"""


#===== Code for Part 1=====#
def part1(Xin,istar):
    """
    Sort list of integers in non-decreasing order
    Input:
    Xin: List of N integers
    istar: integer between 0 and N-1 (0<=istar<=N-1)
    Output:
    X: Sorted list
    """
    X = Xin.copy() 
    for i,x in enumerate(X[1:],1):
        if i<=istar:
            ind = 0
            for j in range(i-1,-1,-1):
                if x>=X[j]:
                    ind = j+1
                    break                   
        else:
            a = 0
            b = i-1
            while a <= b:
                c = (a+b) // 2
                if X[c] < x:
                    a = c + 1
                else:
                    b = c - 1
            ind = a

        X[ind+1:i+1] = X[ind:i]
        X[ind] = x

    return X


def part1_time(inputs=None):
    """Examine dependence of walltimes of part1 function on N and istar
        You may modify the input/output as needed.
    """

    #Add code here for part 1, question 2

    import numpy as np
    import matplotlib.pyplot as plt
    import time
    
    def timer(N, istar, sort=False, desc=False):
        """
        Timer for part1 using a list of length N from a sample of integers from 0 to 2N inclusive
        """
        
        input = list(np.random.randint(0, 2*N, size=N))

        if sort:
            input.sort()
            if desc:
                input.reverse()
        
        t1 = time.time()
        for i in range(100):
            part1(input, istar)
        t2 = time.time()

        return (t2-t1)/100

    N_list = np.logspace(1, 3, num=100, dtype=int)    #   N values
    istar_vals = ["0", "N//2", "N-1"]       #  istar cases

    mm = len(N_list)
    nn = len(istar_vals)

    #   Timings where each row is a different N value, and columns are different istar cases:
    times_asc = np.zeros((mm,nn))       #   Ascending sorted list
    times_desc = np.zeros((mm,nn))      #   Descending sorted list
    times_random = np.zeros((mm,nn))    #   Random list sampled from 0 to 2*N

    #   Timing each case of N, istar, and input list
    for j, N in enumerate(N_list):
        istars = [0, N//2, N-1]
        for k, istar in enumerate(istars):
            t1 = timer(N, istar, sort=True)
            times_asc[j,k] = t1
            print(N, istar_vals[k], "asc", t1)

            t2 = timer(N, istar, sort=True, desc=True)
            times_desc[j,k] = t2
            print(N, istar_vals[k], "desc", t2)

            t3 = timer(N, istar)
            times_random[j,k] = t3
            print(N, istar_vals[k], "rand", t3)

    #   Plots
                
    fig, ax = plt.subplots(2,2)

    ax[0,0].set_title("Time against N for a list in descending order")
    for k in range(nn):
        ax[0,0].plot(N_list, times_desc[:,k], label=istar_vals[k])

    ax[0,1].set_title("Time against N for a list in ascending order")
    for k in range(nn):
        ax[0,1].plot(N_list, times_asc[:,k], label=istar_vals[k])
    
    for j in range(2):
        ax[0,j].legend()
        ax[0,j].set_xlabel('N')
        ax[0,j].set_ylabel('Time (s)')
    
    ax[1,0].set_title("Time against NlogN for istar=0")
    ax[1,0].plot([N*np.log2(N) for N in N_list], times_desc[:,0], color='y', label="Descending")
    ax[1,0].plot([N*np.log2(N) for N in N_list], times_random[:,0], color='m', label="Random 0 to 2N")
    ax[1,0].plot([N*np.log2(N) for N in N_list], times_asc[:,0], color='c', label="Ascending")
    ax[1,0].legend(fontsize='small')
    ax[1,0].set_xlabel('NlogN')
    ax[1,0].set_ylabel('Time (s)')

    ax[1,1].set_title("Time against N^2 for istar=N-1")
    ax[1,1].plot([N**2 for N in N_list], times_desc[:,2], color='y', label="Descending")
    ax[1,1].plot([N**2 for N in N_list], times_random[:,2], color='m', label="Random 0 to 2N")
    ax[1,1].plot([N**2 for N in N_list], times_asc[:,2], color='c', label="Ascending")
    ax[1,1].legend(fontsize='small')
    ax[1,1].set_xlabel('N^2')
    ax[1,1].set_ylabel('Time (s)')

    plt.tight_layout()
    plt.show()

    return None #Modify if needed



#===== Code for Part 2=====#

def part2(S,T,m):
    """Find locations in S of all length-m sequences in T
    Input:
    S,T: length-n and length-l gene sequences provided as strings

    Output:
    L: A list of lists where L[i] is a list containing all locations 
    in S where the length-m sequence starting at T[i] can be found.
   """
    #Size parameters
    n = len(S) 
    l = len(T) 
    
    L = [[] for i in range(l-m+1)] #use/discard as needed

    #Add code here for part 2, question 1

    #  Function used from lecture slides
    def char2base4(S):
        """Convert gene test_sequence
        string to list of ints
        """
        c2b = {}
        c2b['A']=0
        c2b['C']=1
        c2b['G']=2
        c2b['T']=3
        L=[]
        for s in S:
            L.append(c2b[s])
        return L

    #  Function used from lecture slides
    def heval(L,Base,Prime):
        """Convert list L to base-10 number mod Prime
        where Base specifies the base of L
        """
        f = 0
        for l in L[:-1]:
            f = Base*(l+f)
        h = (f + (L[-1])) % Prime
        return h
        
    X = char2base4(S)
    Y = char2base4(T)

    q = 9973     # Chosen prime

    #  Base length-m hash in S
    hi = heval(X[:m],4,q)
    #  Base length-m hash in T
    hp = heval(Y[:m],4,q)
    #  Dictionary for all length-m hashes in T, with lists of corresponding index locations in T as the values
    hp_dict = {hp: [0]}

    #  Comparison of base length-m hash in T to base length-m hash in S
    ind=0   #  Index for X (S)
    if hi == hp:    #  Base hash comparison
        jnd = 0     #  Index for Y (T)
        if X[ind:ind+m] == Y[jnd:jnd+m]:    #  Character match check
            L[jnd].append(ind)

    bm = 4**m % q   # Computed here for efficiency

    #  Rabin Karp rolling hash for each length-m sub-string in T, adding to the dictionary, and comparison against base length-m hash in S
    for jnd in range(1, l-m+1):
        hp = ((4*hp - int(Y[jnd-1])*bm + int(Y[jnd-1+m])) % q)  #  Hash calculation for length-m string in T
        if hp in hp_dict:   #  Adding hash and corresponding index to the dictionary
            hp_dict[hp].append(jnd)
        else:
            hp_dict[hp] = [jnd]
        
        if hi == hp:        #  Comparison against base length-m hash in S
            if X[ind:ind+m] == Y[jnd:jnd+m]:    #  Character match check
                L[jnd].append(ind)
        

    #  Rabin Karp algorithm adapted from lecture slides
    for ind in range(1, n-m+1):
        hi = (4*hi - int(X[ind-1])*bm + int(X[ind-1+m])) % q    #  Hash calculation for length-m sub-string in S
        if hi in hp_dict:   #  Checking whether it matches a hash in the T hash dictionary
            for jnd in hp_dict[hi]:     #  For each index in the T hash dictionary value
                if X[ind:ind+m] == Y[jnd:jnd+m]:    #  Character match check
                    L[jnd].append(ind)  #  Append to L

    return L

if __name__=='__main__':

    #  Part 1

    part1_time()

    #Small example for part 2
    
    S = 'ATCGTACTAGTTATCGT'
    T = 'ATCGT'
    m = 3
    out = part2(S,T,m)

    #Large gene sequence from which S and T test sequences can be constructed
    infile = open(r"Project 1\test_sequence.txt") #file from lab 3
    sequence = infile.read()
    infile.close()

    '''
    import time

    t1 = time.time()
    test1 = part2(sequence, S, m)
    print(time.time()-t1)
    print(test1[0][:5])
    '''
