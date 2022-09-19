import numpy as np

# Define global single-qubit gates
X = np.array([[0., 1.], [1., 0.]])
Y = 1.j * np.array([[0., -1.], [1., 0.]])
Z = np.array([[1., 0.], [0., -1.]])
I  = np.eye(2) # identity gate
Hd = np.array([[1., 1.], [1., -1.]])/np.sqrt(2) # Hadamard gate
S = np.array([[1., 0.], [0., 1.j]]) # phase gate
Sdag = np.array([[1., 0.], [0., -1.j]]) # S^\dag