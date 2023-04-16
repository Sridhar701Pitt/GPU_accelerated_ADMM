import numpy as np

res_u = np.loadtxt('./metric_plots/plot_res_u.txt')
res_x = np.loadtxt('./metric_plots/plot_res_x.txt')
res_x_lambda = np.loadtxt('./metric_plots/plot_res_x_lambda.txt')
res_u_lambda = np.loadtxt('./metric_plots/plot_res_u_lambda.txt')
config_constants = np.loadtxt('./metric_plots/config_constants.txt')
# Experiment Setting Params: ADMM_MAX_ITERS, RHO_ADMM, MAX_ITER, M, TOTAL_TIME, u_lims[0], u_lims[1]
ADMM_MAX_ITERS = config_constants[0]
RHO_ADMM = config_constants[1]
MAX_ITER = config_constants[2]
M = config_constants[3]
TOTAL_TIME = config_constants[4]
u_lims = [config_constants[5], config_constants[6]]

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(res_u[:], 'r', label='u')
plt.plot(res_x[:], 'b', label='x')
plt.plot(res_x_lambda[:], 'g', label='x_lambda')
plt.plot(res_u_lambda[:], 'y', label='u_lambda')
plt.title('ADMM_MAX_ITERS: {}, RHO_ADMM: {}, MAX_ITER: {}, M: {}, TOTAL_TIME: {}, u_lims: {},{}'.format(ADMM_MAX_ITERS, RHO_ADMM, MAX_ITER, M, TOTAL_TIME, u_lims[0], u_lims[1]))
plt.legend()
plt.xlabel('ADMM iteration')
plt.ylabel('residual')
plt.show()
