from guitare_config import *
from UK_parameters import *
from simu_config import t, FextS_NxS_Nt, Fe
from scipy.integrate import odeint, solve_ivp
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

t2 = np.arange(len(t))
# def derivative(time, q) :
#     print(time)
#     # t_idx = mm.find_nearest_index(t, time)
#     q_point = q[NmB+NmS:]
#     q_point_point = A @ q + W @ np.block([
#         [MSinv],
#         [np.zeros((NmB,NmS))]
#     ]) @ U[:,int(time)]

#     Q_point = np.concatenate([q_point, q_point_point])

#     return Q_point

# sol = solve_ivp(derivative, t_span=(float(t2[0]), float(t2[-1])) ,y0 = np.zeros(2*(NmB+NmS)), t_eval=t2)

def derivative(q, time) :
    # print(time)
    t_idx = mm.find_nearest_index(t, time)
    q_point = q[NmB+NmS:]
    q_point_point = A @ q + W @ np.block([
        [MSinv],
        [np.zeros((NmB,NmS))]
    ]) @ U[:,t_idx]

    Q_point = np.concatenate([q_point, q_point_point])

    return Q_point

sol = odeint(derivative, y0 = np.zeros(2*(NmB+NmS)), t=t2)