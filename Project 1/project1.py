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
    
    def sortedlist_timer(N, istar, ascending=True):
        """
        Timer for a sorted list of length N (can vary istar and ascending/descending)
        """
        
        if ascending:
            input = list(range(N))
        else:
            input = list(range(N-1, -1, -1))
        
        t1 = time.time()
        for i in range(10):
            part1(input, istar)
        t2 = time.time()

        return (t2-t1)/10
    
    def noduplicateslist_timer(N, istar):
        """
        Timer for a shuffled list with no duplicate integers of length N (can vary istar)
        """

        input = list(range(N))
        np.random.shuffle(input)

        t1 = time.time()
        for i in range(10):
            part1(input, istar)
        t2 = time.time()

        return (t2-t1)/10
    
    def randomlist_timer(N, istar, sample_range):
        """
        Timer for random list of length N sampled from 0 to sample_range (can vary istar)
        """

        input = np.random.randint(0, sample_range, size=N)

        t1 = time.time()
        for i in range(10):
            part1(input, istar)
        t2 = time.time()

        return (t2-t1)/10
    
    N_list = np.logspace(1, 4, num=10, dtype=int)    #   N values
    istar_vals = ["0", "N//2", "N-1"]       #  istar cases

    mm = len(N_list)
    nn = len(istar_vals)

    #   Timings where each row is a different N value, and columns are different istar cases:
    times_asc = np.zeros((mm,nn))       #   Ascending sorted list
    times_desc = np.zeros((mm,nn))      #   Descending sorted list
    times_nodup = np.zeros((mm,nn))     #   Random list with no duplicate values
    times_random = np.zeros((mm,nn))    #   Random list sampled from 0 to 2*N

    for j, N in enumerate(N_list):
        istars = [0, N//2, N-1]
        for k, istar in enumerate(istars):
            t1 = sortedlist_timer(N, istar, ascending=True)
            print(N, istar_vals[k], "ascending", t1)
            times_asc[j,k] = t1

            t2 = sortedlist_timer(N, istar, ascending=False)
            print(N, istar_vals[k], "descending", t2)
            times_desc[j,k] = t2

            t3 = noduplicateslist_timer(N, istar)
            print(N, istar_vals[k], "no duplicates", t3)
            times_nodup[j,k] = t3

            t4 = randomlist_timer(N, istar, 2*N)
            print(N, istar_vals[k], "random", t4)
            times_random[j,k] = t4
                
    fig, ax = plt.subplots(1,4)

    ax[0].set_title("Ascending")
    for k in range(nn):
        ax[0].plot(N_list, times_asc[:,k], label=istar_vals[k])
        ax[0].legend()

    ax[1].set_title("Descending")
    for k in range(nn):
        ax[1].plot(N_list, times_desc[:,k], label=istar_vals[k])
        ax[1].legend()

    ax[2].set_title("Random No Duplicates")
    for k in range(nn):
        ax[2].plot(N_list, times_nodup[:,k], label=istar_vals[k])
        ax[2].legend()

    ax[3].set_title("Random 0 to 2N")
    for k in range(nn):
        ax[3].plot(N_list, times_random[:,k], label=istar_vals[k])
        ax[3].legend()

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

    #  Function used from slides
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

    #  Function used from slides
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

    q = n-l     # Chosen prime

    #  Base length-m hash in S
    hi = heval(X[:m],4,q)

    #  Base length-m hash in T
    hp = [heval(Y[:m],4,q)]

    #  Comparison of base length-m hash in T to base length-m hash in S
    ind=0   #  index for S and X
    jnd=0   #  index for T and Y
    if hi==hp[jnd]:
        if X[ind:ind+m] == Y[jnd:jnd+m]:
            L[jnd].append(ind)

    bm = 4**m % q   # Computed here for efficiency

    #  Hash for each length-m string in T, and comparison against base length-m hash in S
    for jnd in range(1, l-m+1):
        hp.append((4*hp[-1] - int(Y[jnd-1])*bm + int(Y[jnd-1+m])) % q)
        if hi==hp[jnd]:
            if X[ind:ind+m] == Y[jnd:jnd+m]:
                L[jnd].append(ind)
        

    #  Rabin Karp algorithm adapted from slides
    for ind in range(1, n-m+1):
        hi = (4*hi - int(X[ind-1])*bm + int(X[ind-1+m])) % q
        for jnd in range(l-m+1):
            if hi==hp[jnd]:
                if X[ind:ind+m] == Y[jnd:jnd+m]:
                    L[jnd].append(ind)



    #for k in range(l-m+1):
    #    for ind in range(n-m+1):
    #        if T[k:k+m] == S[ind:ind+m]:
    #            L[k].append(ind)

    return L


if __name__=='__main__':

    #  Part 1

    part1_time()


    #Small example for part 2
    '''
    S = 'ATCGTACTAGTTATCGT'
    T = 'ATCGT'
    m = 3
    out = part2(S,T,m)

    print(out)

    #Large gene sequence from which S and T test sequences can be constructed
    infile = open(r"Project 1\test_sequence.txt") #file from lab 3
    sequence = infile.read()
    infile.close()
    '''

    
