import numpy as np

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# initialize 2D mat #M_F:1, 2, 4, 8, 16, 32 
plot_matrix = np.zeros((4,8))

MF_1 = np.loadtxt('./metric_plots/plot_times_s6_MF_256steps_1.txt')
MF_2 = np.loadtxt('./metric_plots/plot_times_s6_MF_256steps_2.txt')
MF_4 = np.loadtxt('./metric_plots/plot_times_s6_MF_256steps_4.txt')
MF_8 = np.loadtxt('./metric_plots/plot_times_s6_MF_256steps_8.txt')
MF_16 = np.loadtxt('./metric_plots/plot_times_s6_MF_256steps_16.txt')
MF_32 = np.loadtxt('./metric_plots/plot_times_s6_MF_256steps_32.txt')
MF_64 = np.loadtxt('./metric_plots/plot_times_s6_MF_256steps_64.txt')
MF_128 = np.loadtxt('./metric_plots/plot_times_s6_MF_256steps_128.txt')

# Timing Stats: tADMM, t_i_ADMM[0], tTime[0], fsimTime[0], 
# fsweepTime[0], bpTime[0], nisTime[0], initTime[0]
MFS = [MF_1, MF_2, MF_4, MF_8, MF_16, MF_32, MF_64, MF_128]

for i in range(len(MFS)):
    vals = np.array([MFS[i][3], MFS[i][4], MFS[i][5], MFS[i][1]])
    plot_matrix[:, i] = vals


plt.figure(figsize=(16, 12))
ax = plt.axes()
lns1 = ax.plot(plot_matrix.T[:,0], 'b', label='fsim')
lns2 = ax.plot(plot_matrix.T[:,1], 'g', label='fsweep')
lns3 = ax.plot(plot_matrix.T[:,2], 'm', label='bp')
#ax.legend(loc=0)
ax.set_xticklabels(['','1', '2', '4', '8', '16', '32', '64', '128'])

ax2 = ax.twinx()
lns4 = ax2.plot(plot_matrix.T[:,3], 'r', label='tADMM')
#ax2.legend(loc=0)
# added these three lines
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.set_xlabel('Forward Pass Blocks (M_F)')
ax.set_ylabel('Time (ms)')
ax2.set_ylabel('ADMM Time (ms)')

# plt.ylabel('residual')
# plt.title('ok title')
plt.show()



# plt.plot(res_x[:], 'b', label='res_x')
# plt.plot(res_x_lambda[:], 'g', label='res_x_lambda')
# plt.plot(res_u_lambda[:], 'y', label='res_u_lambda')
# plt.plot(rho_admm[:], 'm', label='rho_ADMM')

# textstr = '\n'.join((
#             r'ADMM_ITERS = %d' % (ADMM_MAX_ITERS, ),
#             r'RHO_ADMM = %.2f' % (RHO_ADMM, )))

# props = dict(facecolor='gold', alpha=0.5, boxstyle='round')
# ax.text(0.7, 0.65, textstr, transform=ax.transAxes, fontsize=7,
#         verticalalignment='top', bbox=props, )
# ax.set_xlim([-1, 30])
# ax.set_ylim([-1, 50])