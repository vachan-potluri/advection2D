import numpy as np

# First order matrices
dx = 1./32
dy = 1./32
M = dx*dy/36*np.matrix(
        np.array([[4,2,2,1],
                  [2,4,1,2],
                  [2,1,4,2],
                  [1,2,2,4]])
)
print("Mass: \n{}".format(M))
M_inv = np.linalg.inv(M)
print("Mass inverse: \n{}".format(M_inv))

u = 1./np.sqrt(2)
v = u
Dx = dy/12*np.matrix(
        np.array([[-2,-2,-1,-1],
                  [2,2,1,1],
                  [-1,-1,-2,-2],
                  [1,1,2,2]])
)
Dy = dx/12*np.matrix(
        np.array([[-2,-1,-2,-1],
                  [-1,-2,-1,-2],
                  [2,1,2,1],
                  [1,2,1,2]])
)
D = u*Dx + v*Dy
print("Total differentiation matrix: \n{}".format(D))

F0 = dy/6*np.matrix(
        np.array([[2,0,1,0],
                  [0,0,0,0],
                  [1,0,2,0],
                  [0,0,0,0]])
)
F1 = dy/6*np.matrix(
        np.array([[0,0,0,0],
                  [0,2,0,1],
                  [0,0,0,0],
                  [0,1,0,2]])
)
F2 = dx/6*np.matrix(
        np.array([[2,1,0,0],
                  [1,2,0,0],
                  [0,0,0,0],
                  [0,0,0,0]])
)
F3 = dx/6*np.matrix(
        np.array([[0,0,0,0],
                  [0,0,0,0],
                  [0,0,2,1],
                  [0,0,1,2]])
)
print("Flux matrices: \n{}\n{}\n{}\n{}".format(F0,F1,F2,F3))

print("Stiffness matrix: \n{}".format(np.matmul(M_inv,D)))
print("Lifting matrices: \n{}\n{}\n{}\n{}".format(
        np.matmul(M_inv, F0),
        np.matmul(M_inv, F1),
        np.matmul(M_inv, F2),
        np.matmul(M_inv, F3)
))