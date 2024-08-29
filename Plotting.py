from Analyticformulas import*
import matplotlib.pyplot as plt
import numpy as np
import copy
from RateAnalisys import succ_error_ring, key_siphing
import os
from ConcatRingsOptimization import siphing_tree

def logical_fusion_plot():
    colors = plt.cm.magma(np.linspace(0.9, 0, 5))
    log_succ_plot = {"1": [], "2":[], "3":[], "4":[], "5":[]}
    fancy_log_succ = []
    Z_failures = []
    tranmissions = np.linspace(0.5, 1, 100)
    for eta in tranmissions:
        # Layer one
        log_succ = log_fusion_prob((1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), 1 - eta ** 2, eta)
        log_p_fail_x, log_p_fail_z = log_failure((1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), 1 - eta ** 2, eta)
        log_p_fail_y = 0
        sing_trans = eta
        log_succ_plot["1"].append(log_succ)
        # log_p_fail_x, log_p_fail_z = 0, 0
        for i in range(4):
            sing_trans = log_transmission(sing_trans)
            log_lost = 1 - log_succ - log_p_fail_z - log_p_fail_x
            log_succ_copy = copy.deepcopy(log_succ)
            log_succ = log_fusion_prob(log_succ, log_p_fail_x,
                                       log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
            log_p_fail_x, log_p_fail_z = log_failure(log_succ_copy, log_p_fail_x,
                                       log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
            log_succ_plot[str(i+2)].append(log_succ)


    fig, ax = plt.subplots(figsize=(9, 9))
    photon_loss = [1 - eta for eta in tranmissions]
    cnt = 0
    for key in log_succ_plot.keys():
        ax.plot(photon_loss, log_succ_plot[key], color=colors[cnt], linewidth=2.5, label="N = " + key)
        cnt += 1
    ax.set_xlabel("Photon loss ($1-\eta$)", fontsize=14)
    ax.set_ylabel("Fusion success probability", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(title="layers", fontsize=14, title_fontsize=18)
    plt.show()
    fig.savefig('FusionSuccProbLayers.png', dpi=600)


def logical_Pauli_measurement():
    colors = plt.cm.inferno(np.linspace(0.9, 0, 5))
    log_succ_plot = {"1": [], "2":[], "3":[], "4":[], "5":[]}
    tranmissions = np.linspace(0.5, 1, 100)
    for eta in tranmissions:
        # log_p_fail_x, log_p_fail_z = 0, 0
        sing_trans = eta
        for i in range(5):
            sing_trans = log_transmission(sing_trans)
            log_succ_plot[str(i+1)].append(1-sing_trans)


    fig, ax = plt.subplots(figsize=(9, 9))
    photon_loss = [1 - eta for eta in tranmissions]
    cnt = 0
    for key in log_succ_plot.keys():
        ax.plot(photon_loss, log_succ_plot[key], color=colors[cnt], linewidth=2.5, label="N = " + key)
        cnt += 1
    ax.set_xlabel("Physical photon loss ($1-\eta$)", fontsize=14)
    ax.set_ylabel("Logical loss ($1-\overline{\eta}$)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(title="layers", fontsize=14, title_fontsize=18)
    plt.show()
    # fig.savefig('LogicalLossLayers.png', dpi=600)


def logical_error_rate_no_loss_pauli_measurement():

    colors = plt.cm.magma(np.linspace(0.9, 0, 6))
    errors_plot = np.linspace(0, 0.3, 100)
    errors = [2 * eps / 3 for eps in errors_plot]
    detect = []
    log_errors = {"1":[], "2":[], "3":[], "4":[], "5":[], "6":[]}
    log_detections = {"1": [], "2": [], "3": [], "4": [], "5": [], "6":[]}
    for eps in errors:
        init_eps_f = 0
        for i in range(6):
            eps, init_eps_f = error_prop_layer(eps, init_eps_f)
            log_errors[str(i + 1)].append(eps)
            log_detections[str(i + 1)].append(init_eps_f)

    fig, ax = plt.subplots(figsize=(9, 9))
    cnt = 0
    line1, = ax.plot(errors_plot, log_errors["6"], color=colors[-1], linewidth=2.5, label="Logical error ($\overline{\epsilon}$)")
    line2, = ax.plot(errors_plot, log_detections["6"], color=colors[-1], linestyle="--", linewidth=2.5, label="Logical detection ($\overline{\epsilon}_d$)")
    # ax.legend(title="linstyles")
    first_legend = ax.legend(handles=[line1, line2], title="linestyles", bbox_to_anchor=(0, 1), loc="upper left", fontsize=12, title_fontsize=16)

    # Add the legend manually to the Axes.
    ax.add_artist(first_legend)
    handels = []
    for key in log_errors.keys():
        line1, = ax.plot(errors_plot, log_errors[key], color=colors[cnt], linewidth=2.5, label="N = " + key)
        line2, = ax.plot(errors_plot, log_detections[key], color=colors[cnt], linestyle="--", linewidth=2.5)
        handels.append(line1)
        handels.append(line2)
        cnt += 1

    ax.set_xlabel("Physical error rate ($\epsilon$)", fontsize=14)
    ax.set_ylabel("Logical error/detection rate", fontsize=14)
    ax.legend(handles=handels, title="layers", bbox_to_anchor=(0, 0.7), loc="upper left", fontsize=12, title_fontsize=16)
    # ax.legend(title="layers")
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
    fig.savefig('PauliMeasurementErrorDetect.png', dpi=600)


def loss_and_error_pauli_meas():
    errors_plot = np.linspace(0, 0.3, 50)
    errors = [2 * eps / 3 for eps in errors_plot]
    max_error = 0
    max_detect = 0
    transmissions = np.linspace(0.73, 0.999, 50)
    photon_loss = [1 - eta for eta in transmissions]
    Z = []
    Z_detect = []
    for eps_loop in errors:
        Z_inner = []
        Z_inner_detect = []
        for eta in transmissions:
            eps = copy.deepcopy(eps_loop)
            init_eps_f = 0  # intial_eps_f_with_loss(eps, eta)
            trans = eta # log_transmission(eta)
            for _ in range(6):
                eps, init_eps_f, trans = error_prop_layer_with_loss(eps, init_eps_f, trans)
            Z_inner.append(eps)
            Z_inner_detect.append(init_eps_f)
        Z.append(Z_inner)
        Z_detect.append(Z_inner_detect)
        new_max = max(Z_inner)
        new_max_detect = max(Z_inner_detect)
        if new_max > max_error:
            max_error = new_max
        if new_max_detect > max_detect:
            max_detect = new_max_detect

    # fig = plt.Figure()
    # ax = plt.axes()
    fig, ax = plt.subplots()
    levels = np.linspace(0, max_error, 100)
    cont = ax.contourf(photon_loss, errors_plot, Z, cmap="seismic", vmin=0, vmax=max_error, levels=levels)
    # cont = ax.contourf(transmissions, errors_plot, Z, cmap="seismic", vmin=0, vmax=max_error, levels=levels)
    cb = plt.colorbar(cont)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(12)
    # cb.set_label('Logical error rate ($\overline{\epsilon}$)', rotation=270, size='large')
    plt.title('6 Layers', fontsize=16)
    # plt.ylabel("Physical error rate ($\epsilon$)", fontsize=14)
    # plt.xlabel("Photon loss ($1-\eta$)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()
    # fig.savefig('PauliMeasurementErrorLoss6Layers.png', dpi=600)

    # fig = plt.Figure()
    # ax = plt.axes()
    fig, ax = plt.subplots()
    levels = np.linspace(0, max_detect, 100)
    # cont = ax.contourf(transmissions, errors_plot, Z_detect, cmap="magma", vmin=0, vmax=max_detect, levels=levels)
    # cont = ax.contourf(transmissions, errors_plot, Z_detect, cmap="RdYlBu", vmin=0, vmax=max_detect, levels=levels)
    cont = ax.contourf(photon_loss, errors_plot, Z_detect, cmap="seismic", vmin=0, vmax=max_detect, levels=levels)
    cb = plt.colorbar(cont)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(12)
    # cb.set_label('Logical detection rate ($\overline{\epsilon}_d$)', rotation=270, size='large')
    plt.title('6 Layers', fontsize=16)
    # plt.ylabel("Physical error rate ($\epsilon$)", fontsize=14)
    plt.xlabel("Photon loss ($1-\eta$)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()
    # fig.savefig('PauliMeasurementErrorDetectLoss6Layers.png', dpi=600)


def error_detect_log_fusions():
    errors = np.linspace(0, 0.07, 50)
    transmissions = np.linspace(0.75, 0.99, 50)
    photon_loss = [1 - trans for trans in transmissions]
    Z = []
    Z_detect = []
    N = 8
    max_error = 0
    max_detect = 0
    for eps in errors:
        z_inner = []
        z_detect_inner = []
        for eta_init_ring in transmissions:
            fusion_succ_ring, err_detect, log_succ_error = succ_error_ring(N, 2 * eps / 3, eta_init_ring)
            z_inner.append(abs(log_succ_error))
            z_detect_inner.append(err_detect)
        Z.append(z_inner)
        Z_detect.append(z_detect_inner)
        new_max = max(z_inner)
        new_max_detect = max(z_detect_inner)
        if new_max > max_error:
            max_error = new_max
        if new_max_detect > max_detect:
            max_detect = new_max_detect

    fig, ax = plt.subplots()
    levels = np.linspace(0, max_error, 100)
    cont = ax.contourf(photon_loss, errors, Z, cmap="inferno", vmin=0, vmax=max_error, levels=levels)
    # cont = ax.contourf(transmissions, errors_plot, Z, cmap="seismic", vmin=0, vmax=max_error, levels=levels)
    cb = plt.colorbar(cont)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(12)
    # cb.set_label('Logical error rate ($\overline{\epsilon}$)', rotation=270, size='large')
    plt.title(str(N) + ' Layers', fontsize=16)
    # plt.ylabel("Physical error rate ($\epsilon$)", fontsize=14)
    # plt.xlabel("Photon loss ($1-\eta$)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()
    # fig.savefig('FusionMeasurementErrorLoss8Layers.png', dpi=600)

    fig, ax = plt.subplots()
    levels = np.linspace(0, max_detect, 100)
    # cont = ax.contourf(transmissions, errors_plot, Z_detect, cmap="magma", vmin=0, vmax=max_detect, levels=levels)
    # cont = ax.contourf(transmissions, errors_plot, Z_detect, cmap="RdYlBu", vmin=0, vmax=max_detect, levels=levels)
    cont = ax.contourf(photon_loss, errors, Z_detect, cmap="seismic", vmin=0, vmax=max_detect, levels=levels)
    cb = plt.colorbar(cont)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(12)
    # cb.set_label('Logical detection rate ($\overline{\epsilon}_d$)', rotation=270, size='large')
    plt.title(str(N) + ' Layers', fontsize=16)
    # plt.ylabel("Physical error rate ($\epsilon$)", fontsize=14)
    plt.xlabel("Photon loss ($1-\eta$)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()
    # fig.savefig('FusionMeasurementErrorDetectLoss8Layers.png', dpi=600)


def plot_rates():
    # NOTES: Rember that the error rates here are following a depolarizing channel, it is not eps_r as in Johannes paper
    colors = plt.cm.magma(np.linspace(0.9, 0, 5))
    linestyles = ["solid", "dashed", "dashdot"]
    markers = ["o", "D", "s"]
    # path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\Rates100ns"
    path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\Rates20nspgen"

    fig, ax = plt.subplots(figsize=(9, 9))
    handles_dict = {"0":0, "1":0, "2":0, "3":0}
    for file in os.listdir(path):
        filename = os.path.join(path, file)
        first_char = file[:2]
        last_char = file[-9:-5]
        print(first_char)
        if first_char == "RG":
            m = markers[1]
            l = linestyles[1]
        elif first_char == "ri":
            m = markers[0]
            l = linestyles[0]
        else:
            m = markers[-1]
            l = linestyles[-1]

        if last_char == "e-05":
            lab="$\epsilon=5 x 10^{-5}$"
            idx = 0
        elif last_char == "0001":
            idx = 1
            lab = "$\epsilon=10^{-4}$"
        elif last_char == "0005":
            idx = 2
            lab = "$\epsilon=5 x 10^{-4}$"
        else:
            idx = 3
            lab = "$\epsilon=10^{-3}$"

        rates = np.loadtxt(filename)
        Ls = [x for x in range(100, 1000, 20)]
        plt_rate = []
        for r in rates:
            if r > 1:
                plt_rate.append(r)
            else:
                plt_rate.append(0)
        ax.plot(Ls, plt_rate, marker=m, linestyle=l, linewidth=2.5, markersize=7.5, color=colors[idx])
        if m == markers[0]:
            line1, = ax.plot(Ls[0], plt_rate[0], color=colors[idx], linewidth=2.5, markersize=7.5,label=lab)
            print(line1, idx)
            handles_dict[str(idx)] = line1
            if idx == 0:
                line1, = ax.plot(Ls[0], rates[0],marker=markers[0], linestyle=linestyles[0], color="black", linewidth=2.5,markersize=7.5,
                                 label="Rings")
                line2, = ax.plot(Ls[0], rates[0], marker=markers[1], linestyle=linestyles[1], color="black",
                                 linewidth=2.5,markersize=7.5,
                                 label="RGS")
                line3, = ax.plot(Ls[0], rates[0], marker=markers[2], linestyle=linestyles[2], color="black",
                                 linewidth=2.5,markersize=7.5,
                                 label="Trees")
                first_legend = ax.legend(handles=[line1, line2, line3], title="Linestyles", bbox_to_anchor=(0.2, 0.35),
                                         loc="lower left",
                                         fontsize=12, title_fontsize=12)
                ax.add_artist(first_legend)

    ax.set_xlabel("L (km)", fontsize=14)
    ax.set_ylabel("Rate (Hz)", fontsize=14)
    handles = []
    for key in handles_dict.keys():
        if handles_dict[key] == 0:
            continue
        else:
            handles.append(handles_dict[key])
    print(handles)
    ax.legend(handles=handles, title="Error rate", bbox_to_anchor=(0.2, 0.0), loc="lower left", fontsize=12,
              title_fontsize=12)
    ax.set_yscale("log")
    plt.show()
    fig.savefig('RatesFastGates.png', dpi=600)


if __name__ == '__main__':
    # logical_error_rate_no_loss_pauli_measurement()
    # loss_and_error_pauli_meas()
    # error_detect_log_fusions()
    # logical_fusion_plot()
    # plot_rates()


    # errors = np.linspace(0.001, 0.1, 100)
    # key_s = []
    # fid_key = []
    # for Q in errors:
    #     k_s = siphing_tree(Q)
    #     key_s.append(k_s)
    #     fid_key.append(key_siphing(1-Q))
    # plt.plot(errors, key_s)
    # plt.plot(errors, fid_key)
    # plt.show()
    print(np.exp(-0.18))