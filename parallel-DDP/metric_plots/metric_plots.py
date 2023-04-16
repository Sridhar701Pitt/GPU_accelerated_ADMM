import numpy as np

res_u = np.loadtxt('./metric_plots/plot_res_u.txt')
res_x = np.loadtxt('./metric_plots/plot_res_x.txt')
res_x_lambda = np.loadtxt('./metric_plots/plot_res_x_lambda.txt')
res_u_lambda = np.loadtxt('./metric_plots/plot_res_u_lambda.txt')
config_constants = np.loadtxt('./metric_plots/config_constants.txt')
# Experiment Setting Params: ADMM_MAX_ITERS, RHO_ADMM, MAX_ITER, M, TOTAL_TIME, u_lims[0], u_lims[1], NUM_TIME_STEPS, TIME_STEP
ADMM_MAX_ITERS = config_constants[0]
RHO_ADMM = config_constants[1]
MAX_ITER = config_constants[2]
M = config_constants[3]
TOTAL_TIME = config_constants[4]
u_min = config_constants[5]
u_max = config_constants[6]
NUM_TIME_STEPS = config_constants[7]
TIME_STEP = config_constants[8]

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

plt.figure()
ax = plt.axes()
plt.plot(res_u[:], 'r', label='u')
plt.plot(res_x[:], 'b', label='x')
plt.plot(res_x_lambda[:], 'g', label='x_lambda')
plt.plot(res_u_lambda[:], 'y', label='u_lambda')
#plt.title('ADMM_MAX_ITERS: {}, RHO_ADMM: {}, MAX_ITER: {}, M: {}, TOTAL_TIME: {}, u_lims: {},{}'.format(ADMM_MAX_ITERS, RHO_ADMM, MAX_ITER, M, TOTAL_TIME, u_lims[0], u_lims[1]))

textstr = '\n'.join((
            r'ADMM_ITERS = %d' % (ADMM_MAX_ITERS, ),
            r'RHO_ADMM = %.2f' % (RHO_ADMM, ),
            r'DDP_ITERS = %d' % (MAX_ITER, ),
            r'Time_Horizon = %.1fs' % (TOTAL_TIME, ),
            r'u_lims = %.1f, %.1f' % (u_min, u_max, ),
            r'NUM_TIME_STEPS = %d' % (NUM_TIME_STEPS, ),
            r'TIME_STEP = %.2fs' % (TIME_STEP,)))

props = dict(facecolor='gold', alpha=0.5, boxstyle='round')
ax.text(0.75, 0.5, textstr, transform=ax.transAxes, fontsize=7,
        verticalalignment='top', bbox=props, )

plt.legend()
plt.xlabel('ADMM iteration')
plt.ylabel('residual')
plt.show()
