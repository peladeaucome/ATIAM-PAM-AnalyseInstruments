from guitare_config import *
from UK_parameters import *
from simu_config import t, FextS_NxS_Nt
from scipy.integrate import odeint
import mymodule2 as mm

MSinv = np.linalg.inv(MS)
MBinv = np.linalg.inv(MB)

ABG = W @ np.block([
    [-MSinv @ KS, np.zeros((NmS,NmB))],
    [np.zeros((NmB, NmS)), -MBinv @ KB]
])
ABD = W @ np.block([
    [-MSinv @ CS, np.zeros((NmS,NmB))],
    [np.zeros((NmB, NmS)), -MBinv @ CB]
])

A = np.block([
    [ ABG         , ABD      ]
])

U = FextS_NxS_Nt
U = phiS_Nx_NmS.T @ U


def derivative(q, time) :
    t_idx = mm.find_nearest_index(t, time)
    q_point = q[NmB+NmS:]
    q_point_point = A @ q + W @ np.block([
        [MSinv],
        [np.zeros((NmB,NmS))]
    ]) @ U[:,t_idx]

    Q_point = np.concatenate([q_point, q_point_point])

    # print(q_point_point.shape)
    return Q_point

sol = odeint(derivative, y0 = np.zeros(2*(NmB+NmS)), t=t)