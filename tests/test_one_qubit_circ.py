import sys
sys.path.append("../")
from one_qubit_circ import *
import numpy as np
import time

X = np.array([[0., 1.], [1., 0.]])
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

def test_control_op_1q():
    cx = control_op_1q(X)
    diff = np.linalg.norm(cx - CNOT)
    assert diff < 1e-15
def test_measure_ancilla():
    state = np.random.rand(4)
    state /= np.linalg.norm(state)
    prop_0 = np.linalg.norm(state[0])**2 + np.linalg.norm(state[1])**2
    prop_1 = np.linalg.norm(state[2])**2 + np.linalg.norm(state[3])**2
    assert (prop_0 + prop_1) - 1 < 1e-10

    # repeat the sampling
    N0, N1 = 0, 0
    Ns = 100000
    t1 = time.time()
    for i in range(Ns):
        out = measure_ancilla(state)
        if abs(out - 1) < 1e-10:
            N1 += 1
        elif abs(out) < 1e-10:
            N0 += 1
        else:
            raise ValueError("The measurement outcome should be 0 or 1!")
    t2 = time.time()
    print("time used to draw {} samples: {}".format(Ns, t2 - t1))
    p0 = 1. * N0 / Ns
    p1 = 1. * N1 / Ns

    assert np.abs(p0 - prop_0) < 1e-2
    print("Diff: {}".format(np.abs(p0 - prop_0)))


if __name__ == "__main__":
    #test_control_op_1q()
    test_measure_ancilla()