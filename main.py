import numpy as np
import control as ct
import matplotlib.pyplot as plt
from kf import KalmanFilter
import plot_helper as plth

# constants
k1 = k2 = k3 = k4 = k5 = 1
c1 = c2 = c3 = c4 = c5 = 0.01
m1 = m2 = m3 = m4 = m5 = 1
num_states = 10
num_inputs = 1
num_sensors = 2
dt = 0.05

# dynamics
# jacobian(Xdot, X)
Fc = np.zeros((num_states, num_states))
Fc[(0, 2, 4, 6, 8), (1, 3, 5, 7, 9)] = (1, 1, 1, 1, 1)
Fc[1, :] = (-(k1+k2)/m1, -(c1+c2)/m1, k2/m1, c2/m1, 0, 0, 0, 0, 0, 0)
Fc[3, :] = (k2/m2, c2/m2, -(k2+k3)/m2, -(c2+k3)/m2, k3/m2, c3/m2, 0, 0, 0, 0)
Fc[5, :] = (0, 0, k3/m3, c3/m3, -(k3+k4)/m3, -(c3+k4)/m3, k4/m3, c4/m3, 0, 0)
Fc[7, :] = (0, 0, 0, 0, k4/m4, c4/m4, -(k4+k5)/m4, -(c4+c5)/m4, k5/m4, c5/m4)
Fc[9, :] = (0, 0, 0, 0, 0, 0, k5/m5, c5/m5, -k5/m5, -c5/m5)
# jacobian(Xdot, u)
Gc = np.zeros((num_states, num_inputs))
Gc[-1, 0] = 1/m5
# jacobian(z, X) where z is measurements
Hc = np.zeros((num_sensors, num_states))
Hc[0, :2] = (1, 0)
Hc[1, -2:] = (1, 0)
#jacobian(z, u)
Dc = np.zeros((num_sensors, num_inputs))
# state space model
sysc = ct.ss(Fc, Gc, Hc, Dc)
sysd = ct.sample_system(sysc, dt, method='zoh')

# simulate sys
X0 = np.zeros((num_states,1))
X0[-2, 0] = 1
tsim = np.arange(0, 50 + dt, dt)
num_k = len(tsim)
Qsim = 0.01
w = np.random.normal(0, np.sqrt(Qsim), num_k)
# out_initial = ct.initial_response(sysd, tsim, X0)
# X_clean = out_initial.states
out_forced = ct.input_output_response(sysd, tsim, w, X0)
X_noisy = out_forced.states

# generate mock measurements
Rq1 = 0.001
vq1 = np.random.normal(0, np.sqrt(Rq1), num_k)
Zq1 = X_noisy[0,:] + vq1
Zq1[0] = np.nan
Rq5 = 0.001
vq5 = np.random.normal(0, np.sqrt(Rq5), num_k)
Zq5 = X_noisy[8,:] + vq5
Zq5[0] = np.nan

# (a) Kalman Filter using only Zq1
# initialize kalman filter
Fa = sysd.A
Ga = sysd.B
Ha = sysd.C[0:1, :]
Qa = np.diag([Qsim])
Ra = np.diag([Rq1])
za = Zq1
X0a = X0
P0a = np.diag(np.tile((0.1, 1), 5))
KFa = KalmanFilter(X0a, P0a, Fa, Ga, Ha, Qa, Ra)

# run kalman filter
xpa = np.zeros((num_states, num_k))
xua = np.zeros_like(xpa)
Ppa = np.zeros((num_states, num_states, num_k))
Pua = np.zeros_like(Ppa)          
for k in range(num_k):
    if k == 0:
        xpa[:, k] = xua[:,k] = X0.ravel()
        Ppa[:, :, k] = Pua[:, :, k] = P0a
        continue
    KFa.predict(u=0)
    KFa.update(z=za[k])
    xpa[:, k] = KFa.xp.ravel()
    xua[:, k] = KFa.xu.ravel()
    Ppa[:, :, k] = KFa.Pp
    Pua[:, :, k] = KFa.Pu

# plot
sig_q1 = np.sqrt(Pua[0,0,:])
plt.figure()
plt.plot(tsim, xua[0, :], label='Estimated Position', linewidth=2)
plt.fill_between(tsim, xua[0,:] - 2*sig_q1, xua[0,:] + 2*sig_q1, label=r"$2\sigma$ Bound", color='purple', alpha=0.3)
plt.plot(tsim, X_noisy[0, :], label='True Position', linewidth=2)
plt.plot(tsim[1:], za[1:], label='Measurement', linewidth=.4, linestyle='--')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Position of mass 1, measuring mass 1')
plth.set_y_axis_limits(np.min(xua[0, :]), np.max(xua[0, :]), 20)

sig_q5 = np.sqrt(Pua[8,8,:])
plt.figure()
plt.plot(tsim, xua[8, :], label='Estimated Position', linewidth=2)
plt.fill_between(tsim, xua[8, :] - 2*sig_q5, xua[8,:] + 2*sig_q5, label = r"$2\sigma$ Bound", color='purple', alpha=0.3)
plt.plot(tsim, X_noisy[8, :], label='True Position', linewidth=2)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Position of mass 5, measuring mass 1')

plt.rcParams.update({'font.size': 20})
plt.show()
