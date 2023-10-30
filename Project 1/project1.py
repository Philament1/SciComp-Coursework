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
        if i<=istar:    #  Insertion sort
            ind = 0
            for j in range(i-1,-1,-1):
                if x>=X[j]:
                    ind = j+1
                    break                   
        else:   #  Binary insertion sort
            a = 0
            b = i-1
            while a <= b:
                c = (a+b) // 2
                if X[c] < x:
                    a = c + 1
                else:
                    b = c - 1
            ind = a
        
        #print(i, ind)
        #print(X[ind+1:i+1])

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

    def randomlist_timer(N, istar):

        import time

        input = np.random.randint(0, N-1, size=N)

        t1 = time.time()
        part1(input, istar)
        t2 = time.time()

        return t2-t1
    
    def sortedlist_timer(N, istar, ascending=True):
        
        import time

        if ascending:
            input = list(range(N))
        else:
            input = list(range(N-1, -1, -1))
        
        t1 = time.time()
        part1(input, istar)
        t2 = time.time()

        return t2-t1
        
    N_list = np.logspace(1, 9, num=9, dtype=int)

    #times = []

    for N in N_list:
        istars = np.linspace(0, N-1, num=2, dtype=int)
        for istar in istars:
            
            t = randomlist_timer(N, istar)
            print(N, istar, "random", t)

            t1 = sortedlist_timer(N, istar, ascending=True)
            print(N, istar, "ascending", t1)

            t2 = sortedlist_timer(N, istar, ascending=False)
            print(N, istar, "descending", t2)
            
        #times.append(t)
        

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

    bm = 4**m % q   # Computed here for efficiency

    #  Base length-m hash in S
    hi = heval(X[:m],4,q)

    #  Base length-m hash in T
    hp = [heval(Y[:m],4,q)]

    #  Comparison of base T to base S
    ind=0   #  index for S and X
    jnd=0   #  index for T and Y
    if hi==hp[jnd]:
        if X[ind:ind+m] == Y[jnd:jnd+m]:
            L[jnd].append(ind)

    #  Hash for each length-m string in T, and comparison against base length-m hash in S
    for jnd in range(1, l-m+1):
        hp.append((4*hp[-1] - int(Y[jnd-1])*bm + int(Y[jnd-1+m])) % q)
        if hi==hp[jnd]:
            if X[ind:ind+m] == Y[jnd:jnd+m]:
                L[jnd].append(ind)
        

    #  Rabin Karp algorithm
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

    
