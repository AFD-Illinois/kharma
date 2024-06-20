import h5py
import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
#                                               WRITING INTO TEXT FILE 
########################################################################################################################
# def write_hdf5_contents_to_txt(file, txt_file):
#     with open(txt_file, 'w') as output_file:
#         def visit(name, obj):
#             if isinstance(obj, h5py.Group):
#                 # Handle groups (folders)
#                 output_file.write(f'Group: {name}\n')
#             elif isinstance(obj, h5py.Dataset):
#                 # Handle datasets (data)
#                 output_file.write(f'Dataset: {name}\n')
#                 output_file.write(f'Data:\n{obj[()]}\n\n')
#         # Traverse the HDF5 file and write contents
#         file.visititems(visit)

########################################################################################################################
#                                               SETTING UP ARRAYS
########################################################################################################################

## FOR QUICK PLUGGING IN 
# DCS_DUMP : "path to dump"
# EDGB_DUMP : "path to dump"

dump_theory = "/scratch/bbgv/smajumdar/dev_merge_tests/dcs_test/dcs_gr_test/dumps_kharma/torus.out3.00000.phdf"
dump_gr = "/scratch/bbgv/smajumdar/dev_merge_tests/gr_for_thesis/dumps_kharma/torus.out3.00000.phdf"

dump_output = 'dcs_gr.txt'

# Acess theory dump
with h5py.File(dump_theory, 'r') as file1:
    # write_hdf5_contents_to_txt(file, EDGB01output)
    gcov_theory = file1['coords.gcov']
    theory_gcov = gcov_theory[0,:,:,0,:,:]
    X = file1['coords.Xsph']
    Xn = file1['coords.Xnative']

    x_slice = Xn[0, 1, :, :, :]
    z_slice = Xn[0, 2, :, :, :]

    r_plot = X[0, 1, 0, :, :]
    th_plot = X[0, 2, 0, :, :]

# Acess GR dump
with h5py.File(dump_gr, 'r') as file2:
    # write_hdf5_contents_to_txt(file, EDGB01output)
    gcov_gr = file2['coords.gcov']
    gr_gcov = gcov_gr[0,:,:,0,:,:]


# Setup arrays
diff = np.abs(gr_gcov-theory_gcov)
# print(r_plot.shape) 

########################################################################################################################
#                                            PLOTTING METRIC DIFFERENCES 
########################################################################################################################

fig, axs = plt.subplots(2, 2, figsize=(13, 10))
plt.set_cmap('cividis')
plt.rcParams['font.family'] = 'serif'

axs[0, 0].set_aspect('auto')
axs[0, 1].set_aspect('auto')
axs[1, 0].set_aspect('auto')
axs[1, 1].set_aspect('auto')

tt = axs[0, 0].pcolormesh(r_plot, th_plot, diff[0,0,:,:])
axs[0, 0].set_title('$g_{tt}$ diff')
axs[0, 0].set_xlabel('r')
axs[0, 0].set_ylabel('$\\theta$')
fig.colorbar(tt, ax=axs[0, 0])

rr = axs[0, 1].pcolormesh(r_plot, th_plot, diff[1,1,:,:])
axs[0, 1].set_title('$g_{rr}$ diff')
axs[0, 1].set_xlabel('r')
axs[0, 1].set_ylabel('$\\theta$')
fig.colorbar(tt, ax=axs[0, 1])

tp = axs[1, 0].pcolormesh(r_plot, th_plot, diff[0,3,:,:])
axs[1, 0].set_title('$g_{t\phi}$ diff')
axs[1, 0].set_xlabel('r')
axs[1, 0].set_ylabel('$\\theta$')
fig.colorbar(tt, ax=axs[1, 0])

pp = axs[1, 1].pcolormesh(r_plot, th_plot, diff[3,3,:,:])
axs[1, 1].set_title('$g_{\phi\phi}$ diff')
axs[1, 1].set_xlabel('r')
axs[1, 1].set_ylabel('$\\theta$')
fig.colorbar(tt, ax=axs[1, 1])

# for ax in axs.flat:
#     ax.set_xlim(1.7360295289859515, 10)

fig.suptitle('Metric Component Differences \n dCS [$\zeta$=0] - GR \n $a$=0.5')
plt.savefig("DCS_diff_PLOT.png")

########################################################################################################################