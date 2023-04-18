import numpy as np

res_u = np.loadtxt('./metric_plots/plot_res_u.txt')
res_x = np.loadtxt('./metric_plots/plot_res_x.txt')
res_x_lambda = np.loadtxt('./metric_plots/plot_res_x_lambda.txt')
res_u_lambda = np.loadtxt('./metric_plots/plot_res_u_lambda.txt')
config_constants = np.loadtxt('./metric_plots/config_constants.txt')
rho_admm = np.loadtxt('./metric_plots/plot_rho_ADMM.txt')
timing_stats = np.loadtxt('./metric_plots/plot_times.txt')

# Timing Stats: tADMM, t_i_ADMM[0], tTime[0], fsimTime[0], 
# fsweepTime[0], bpTime[0], nisTime[0], initTime[0]
tADMM = timing_stats[0]
t_0_ADMM = timing_stats[1]
tTime = timing_stats[2]
fsimTime = timing_stats[3]
fsweepTime = timing_stats[4]
bpTime = timing_stats[5]
nisTime = timing_stats[6]
initTime = timing_stats[7]


# Experiment Setting Params: ADMM_MAX_ITERS, RHO_ADMM, MAX_ITER, M, TOTAL_TIME, u_lims[0], u_lims[1], NUM_TIME_STEPS, TIME_STEP
ADMM_MAX_ITERS = config_constants[0]
RHO_ADMM = config_constants[1]
MAX_ITER = config_constants[2]

TOTAL_TIME = config_constants[3]
u_min = config_constants[4]
u_max = config_constants[5]
NUM_TIME_STEPS = config_constants[6]
TIME_STEP = config_constants[7]

M_B = config_constants[8]
M_F = config_constants[9]

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 12))
ax = plt.axes()
plt.plot(res_u[:], 'r', label='res_u')
plt.plot(res_x[:], 'b', label='res_x')
plt.plot(res_x_lambda[:], 'g', label='res_x_lambda')
plt.plot(res_u_lambda[:], 'y', label='res_u_lambda')
plt.plot(rho_admm[:], 'm', label='rho_ADMM')
#plt.title('ADMM_MAX_ITERS: {}, RHO_ADMM: {}, MAX_ITER: {}, M: {}, TOTAL_TIME: {}, u_lims: {},{}'.format(ADMM_MAX_ITERS, RHO_ADMM, MAX_ITER, M, TOTAL_TIME, u_lims[0], u_lims[1]))

textstr = '\n'.join((
            r'ADMM_ITERS = %d' % (ADMM_MAX_ITERS, ),
            r'RHO_ADMM = %.2f' % (RHO_ADMM, ),
            r'DDP_ITERS = %d' % (MAX_ITER, ),
            r'M_B,M_F = %d | %d' % (M_B,M_F),
            r'Time_Horizon = %.1fs' % (TOTAL_TIME, ),
            r'u_lims = %.1f, %.1f' % (u_min, u_max, ),
            r'NUM_TIME_STEPS = %d' % (NUM_TIME_STEPS, ),
            r'TIME_STEP = %.2fs' % (TIME_STEP,),
            r'ADMM t0/T = %.2f / %.2fms' % (t_0_ADMM,tADMM,),
            r'tTime = %.2fms' % (tTime,),
            r'fsimTime = %.2fms' % (fsimTime,),
            r'fsweepTime = %.2fms' % (fsweepTime,),
            r'bpTime = %.2fms' % (bpTime,),
            r'nisTime = %.2fms' % (nisTime,),
            r'initTime = %.2fms' % (initTime,)))

props = dict(facecolor='gold', alpha=0.5, boxstyle='round')
ax.text(0.7, 0.65, textstr, transform=ax.transAxes, fontsize=7,
        verticalalignment='top', bbox=props, )
ax.set_xlim([-1, 30])
ax.set_ylim([-1, 50])
plt.legend()
plt.xlabel('ADMM iteration')
plt.ylabel('residual')
plt.show()
