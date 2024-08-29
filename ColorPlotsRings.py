from RateAnalisys import succ_error_ring
import matplotlib.pyplot as plt
import numpy as np

errors = np.linspace(0, 0.07, 50)
transmissions = np.linspace(0.75, 0.99, 50)
photon_loss = [1 - trans for trans in transmissions]
Z = []
Z_detect = []
N = 10
max_error = 0
max_detect = 0
for eps in errors:
    z_inner = []
    z_detect_inner = []
    for eta_init_ring in transmissions:
        fusion_succ_ring, err_detect, log_succ_error = succ_error_ring(N, 2 * eps / 3, eta_init_ring)
        z_inner.append(log_succ_error)
        z_detect_inner.append(err_detect)
        print("eta = ", eta_init_ring, " , eps = ", eps, ", detect = ", err_detect, " , error = ", log_succ_error)
    Z.append(z_inner)
    Z_detect.append(z_detect_inner)
    new_max = max(z_inner)
    new_max_detect = max(z_detect_inner)
    if new_max > max_error:
        max_error = new_max
    if new_max_detect > max_detect:
        max_detect = new_max_detect

fig = plt.Figure()
ax = plt.axes()
levels = np.linspace(0, max_error, 100)
cont = ax.contourf(transmissions, errors, Z, cmap="plasma",  vmin=0, vmax=max_error, levels=levels)
CS2 = ax.contour(cont, levels=[0.01], colors='r')
plt.colorbar(cont)
plt.title('Logical error')
plt.ylabel("Physical error rate ($\epsilon$)")
plt.xlabel("Transmission ($\eta$)")
plt.show()


fig = plt.Figure()
ax = plt.axes()
levels = np.linspace(0, max_detect, 100)
cont = ax.contourf(transmissions, errors, Z_detect, cmap="plasma",  vmin=0, vmax=max_detect, levels=levels)
CS2 = ax.contour(cont, levels=[0.01], colors='r')
plt.colorbar(cont)
plt.title('Error detection')
plt.ylabel("Physical error rate ($\epsilon$)")
plt.xlabel("Transmission ($\eta$)")
plt.show()