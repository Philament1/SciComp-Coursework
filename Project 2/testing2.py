import numpy as np
from project2 import part2q1, part2q1new, part2q2, part2q3Analyze

part2q2()


    # def RHS(t,y):
    #     """
    #     Compute RHS of model
    #     """        
    #     dydt = np.zeros_like(y)
    #     dydt[1:-1] = alpha*y[1:-1]-y[1:-1]**3 + beta*(y[2:]+y[:-2])
    #     dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[-1])
    #     dydt[-1] = alpha*y[-1]-y[-1]**3 + beta*(y[0]+y[-2])

    #     return dydt 

#part2q3Analyze()

# import scipy

# y = np.array([2, 2, 3, 1, 2]) 
# n = len(y)
# beta = 5

# M  = scipy.sparse.diags([y] + [beta]*4, [0, 1, -1, n-1, -(n-1)])

# print(M.toarray())

# #print(scipy.sparse.linalg.eigs(M))


# print(M.toarray() * y[: None])