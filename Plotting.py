from Analyticformulas import*
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from AnalyticformulasUpdateLastStrat import succ_ring_with_individ_det_with_fail_traj

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
        for i in range(4):
            sing_trans = log_transmission(sing_trans)
            log_lost = 1 - log_succ - log_p_fail_z - log_p_fail_x
            log_succ_copy = copy.deepcopy(log_succ)
            log_succ = log_fusion_prob(log_succ, log_p_fail_x,
                                       log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
            log_p_fail_x, log_p_fail_z = log_failure(log_succ_copy, log_p_fail_x,
                                       log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
            log_succ_plot[str(i+2)].append(log_succ)

    fig, ax = plt.subplots(figsize=(11, 13))
    fig.set_size_inches(11, 5.5)
    photon_loss = [1 - eta for eta in tranmissions]
    cnt = 0
    standard_fusion = [(1 / 2) * (eta ** 2) for eta in tranmissions]
    for key in log_succ_plot.keys():
        ax.plot(photon_loss, log_succ_plot[key], color=colors[cnt], linewidth=3.5, label="N = " + key)
        cnt += 1
    ax.plot(photon_loss, standard_fusion, "--", color="black", linewidth=3.5, label="Standard fusion")
    # line1, = ax.plot(photon_loss[0], standard_fusion[0], "--", color="black", linewidth=2.5, label="standard fusion")

    # first_legend = ax.legend(handles=[line1], title="Standard fusion", bbox_to_anchor=(0.6, 0.67),
    #                                      loc="lower left",
    #                                      fontsize=12, title_fontsize=12)  # 0.6, 0.25, 0.57, 0.2, slow:0.57, 0.09
    # ax.add_artist(first_legend)
    # ax.set_xlabel("Photon loss ($1-\eta$)", fontsize=14)
    # ax.set_ylabel("Fusion success probability", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(title="Layers", fontsize=18, title_fontsize=20)
    fig.savefig('FusionSuccProbLayers.pdf', dpi=600)
    plt.show()
    # fig.savefig('FusionSuccProbLayers.pdf', dpi=600)


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





def loss_and_error_pauli_meas(N_layer=3, normalized=False):
    errors_plot = np.linspace(0, 0.3, 100)
    errors = [2 * eps / 3 for eps in errors_plot]
    max_error = 0
    max_detect = 0
    transmissions = np.linspace(0.75, 0.999, 100)
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
            for _ in range(N_layer):
                eps, init_eps_f, trans = error_prop_layer_with_loss(eps, init_eps_f, trans)
            if normalized:
                Z_inner.append(eps / (1 - init_eps_f))
            else:
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
    fig, ax = plt.subplots(figsize=(6, 4))
    # fig.figsize(6, 4)
    levels = np.linspace(0, max_error, 100)
    # cont = ax.contourf(photon_loss, errors_plot, Z, cmap="RdYlBu", vmin=0, vmax=max_error, levels=levels)
    cont = ax.contourf(photon_loss, errors_plot, Z, cmap="RdBu", vmin=0, vmax=max_error, levels=levels)
    # cb = plt.colorbar(cont, ticks=[0, 0.03, 0.06,  0.09, 0.12, 0.15, 0.18, 0.21])
    # cb = plt.colorbar(cont, ticks=[0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24])
    # cb = plt.colorbar(cont, ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
    cb = plt.colorbar(cont, ticks=[0, 0.06, 0.12, 0.18, 0.24, 0.3, 0.36, 0.42, 0.48])
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(17)
    plt.tick_params(axis='both', which='major', labelsize=17)
    # plt.show()
    fig.savefig('PauliMeasurementErrorLoss' + str(N_layer) + 'Layers.pdf', dpi=600)

    fig, ax = plt.subplots(figsize=(6, 4))
    # fig.figsize(6, 4)
    levels = np.linspace(0, max_detect, 100)
    cont = ax.contourf(photon_loss, errors_plot, Z_detect, cmap="RdBu", vmin=0, vmax=max_detect, levels=levels)
    # cb = plt.colorbar(cont, ticks=[0, 0.1, 0.2,  0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    cb = plt.colorbar(cont, ticks=[0, 0.08, 0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64])
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(17)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.show()
    # fig.savefig('PauliMeasurementErrorDetectLoss' + str(N_layer) + 'Layers.pdf', dpi=600)


def error_detect_log_fusions_updated_detect_individ_fail_traj(N=4, normalized=False):
    errors = np.linspace(0.001, 0.07, 100)  # np.linspace(0.025, 0.035, 10)
    transmissions = np.linspace(0.8, 0.99, 100)  # [0.75, 0.85]
    photon_loss = [1 - trans for trans in transmissions]
    Z = []
    Z_detect = []
    max_error = 0
    max_detect = 0
    for eps in errors:
        z_inner = []
        z_detect_inner = []
        for eta_init_ring in transmissions:
            fusion_succ_ring, err_detect, log_succ_error = succ_ring_with_individ_det_with_fail_traj(N, 2 * eps / 3, eta_init_ring)
            if normalized:
                z_inner.append(log_succ_error / (1 - err_detect))
            else:
                z_inner.append(log_succ_error)
            z_detect_inner.append(err_detect)
        Z.append(z_inner)
        Z_detect.append(z_detect_inner)
        new_max = max(z_inner)
        new_max_detect = max(z_detect_inner)
        if new_max > max_error:
            max_error = new_max
        if new_max_detect > max_detect:
            max_detect = new_max_detect

    fig, ax = plt.subplots(figsize=(6, 4))
    levels = np.linspace(0, max_error, 100)
    # cont = ax.contourf(photon_loss, errors, Z, cmap="inferno", vmin=0, vmax=max_error, levels=levels)
    # cont = ax.contourf(photon_loss, errors, Z, cmap="RdYlBu", vmin=0, vmax=max_error, levels=levels)
    cont = ax.contourf(photon_loss, errors, Z, cmap="RdBu", vmin=0, vmax=max_error, levels=levels)
    # cont = ax.contourf(photon_loss, errors, Z, cmap="RdBu")
    # cb = plt.colorbar(cont, ticks=[0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18])
    # cb = plt.colorbar(cont, ticks=[0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24])

    cb = plt.colorbar(cont, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    # cb = plt.colorbar(cont, ticks=[0.0, 0.12, 0.24, 0.36, 0.48, 0.6, 0.72, 0.84])
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(17)
    # cb.set_label('Logical error rate ($\overline{\epsilon}$)', rotation=270, size='large')
    # plt.ylabel("Physical error rate ($\epsilon$)", fontsize=14)
    # plt.xlabel("Photon loss ($1-\eta$)", fontsize=14)
    print(max_error)
    plt.tick_params(axis='both', which='major', labelsize=17)
    #plt.show()
    fig.savefig('FusionMeasurementErrorLoss' + str(N) + 'LayersNew.pdf')

    fig, ax = plt.subplots(figsize=(6, 4))
    levels = np.linspace(0, max_detect, 100)
    # cont = ax.contourf(transmissions, errors_plot, Z_detect, cmap="magma", vmin=0, vmax=max_detect, levels=levels)
    # cont = ax.contourf(transmissions, errors_plot, Z_detect, cmap="RdYlBu", vmin=0, vmax=max_detect, levels=levels)
    cont = ax.contourf(photon_loss, errors, Z_detect, cmap="RdBu", vmin=0, vmax=max_detect, levels=levels)
    cb = plt.colorbar(cont, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # cb = plt.colorbar(cont, ticks=[0, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.8])
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(17)
    # cb.set_label('Logical detection rate ($\overline{\epsilon}_d$)', rotation=270, size='large')
    # plt.ylabel("Physical error rate ($\epsilon$)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.show()
    fig.savefig('FusionMeasurementErrorDetectLoss' + str(N) + 'LayersNew.pdf')






def parse_all_info(path, L_s_init, sparse_flag=True):
    rates_dict = {"ri": {"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]},
                  "RG":{"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]},
                  "tr":{"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]}}
    stations_dict = {"ri": {"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]},
                     "RG":{"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]},
                     "tr":{"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]}}
    cost_dict = {"ri": {"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]},
                 "RG":{"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]},
                 "tr":{"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]}}
    lengt_dict = {"ri": {"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]},
                  "RG":{"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]},
                  "tr":{"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]}}
    lengt_dict_m_c = {"ri": {"e-05": [], "0001": [], "0005": [], ".003": [], ".001": []},
                  "RG": {"e-05": [], "0001": [], "0005": [], ".003": [], ".001": []},
                  "tr": {"e-05": [], "0001": [], "0005": [], ".003": [], ".001": []}}

    Ns_dict = {"ri": {"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]},
                  "RG":{"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]},
                  "tr":{"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]}}

    Ns_dict_plot = {"ri": {"e-05": [], "0001":[], "0005": [], ".003":[], ".001":[]}}

    for file in os.listdir(path):
        filename = os.path.join(path, file)
        first_char = file[:2]
        last_char = file[-9:-5]
        list_para = np.loadtxt(filename)
        rates = []
        ms = []
        Ns = []
        cs = []
        if first_char != "ri":
            for ix in range(len(list_para)):
                if ix % 2 == 0:
                    r, m1, c = list_para[ix]
                    rates.append(r)
                    ms.append(m1)
                    # ms.append(m1 * 3)
                    cs.append(c * 3)
        else:
            Ns = []
            for ix in range(len(list_para)):
                r, m1, N, c = list_para[ix]
                rates.append(r)
                ms.append(m1)
                # ms.append(m1 * (N + 1))
                Ns.append(N+1)
                cs.append(c)
            Ns_dict["ri"][last_char] = Ns
            Ns_dict["RG"][last_char] = Ns
            Ns_dict["tr"][last_char] = Ns
        Ls = []
        rates_plot = []
        cost_plot = []
        m_plot = []
        L_s_m_c = []
        Ns_plot = []
        if len(ms) > 0:
            if sparse_flag:
                for ix in range(len(ms)):
                    if ix == len(ms) - 1 and len(ms) < len(L_s_init):
                        m_plot.append(ms[ix])
                        rates_plot.append(rates[ix])
                        cost_plot.append(cs[ix])
                        Ls.append(L_s_init[ix])
                        L_s_m_c.append(L_s_init[ix])

                        # m_plot.append(ms[ix])
                        rates_plot.append(0)
                        # cost_plot.append(cs[ix])
                        Ls.append(L_s_init[ix + 3])
                        if first_char == "ri":
                            Ns_plot.append(Ns[ix])
                    if ix % 3 == 0:   # or ix == len(ms) - 1:
                        m_plot.append(ms[ix])
                        rates_plot.append(rates[ix])
                        cost_plot.append(cs[ix])
                        Ls.append(L_s_init[ix])
                        L_s_m_c.append(L_s_init[ix])
                        if first_char == "ri":
                            Ns_plot.append(Ns[ix])
            else:
                for ix in range(len(ms)):
                    if ms[ix] > 0:
                        m_plot.append(ms[ix])
                        cost_plot.append(cs[ix])
                        L_s_m_c.append(L_s_init[ix])
                        if first_char == "ri":
                            Ns_plot.append(Ns[ix])

                    rates_plot.append(rates[ix])
                    Ls.append(L_s_init[ix])
                if len(ms) < len(L_s_init):
                    # m_plot.append(ms[-1])
                    rates_plot.append(0)
                    # cost_plot.append(cs[-1])
                    Ls.append(L_s_init[len(ms) + 3])
        stations_dict[first_char][last_char] = m_plot
        rates_dict[first_char][last_char] = rates_plot
        cost_dict[first_char][last_char] = cost_plot
        lengt_dict[first_char][last_char] = Ls
        lengt_dict_m_c[first_char][last_char] = L_s_m_c
        if first_char == "ri":
            Ns_dict_plot[first_char][last_char] = Ns_plot
    for key in rates_dict.keys():
        if key != "ri":
            for error_key in rates_dict[key].keys():
                new_rate = []
                N = Ns_dict[key][error_key]
                rates = rates_dict[key][error_key]
                lengt = lengt_dict[key][error_key]
                for ix in range(len(lengt)):
                    idx = L_s_init.index(lengt[ix])
                    if N[idx] > 2:
                        new_rate.append(rates[ix] * N[idx] / 3)
                    else:
                        new_rate.append(rates[ix])
                print(len(new_rate), len(rates_dict[key][error_key]))
                rates_dict[key][error_key] = new_rate


    return rates_dict, stations_dict, cost_dict, lengt_dict, lengt_dict_m_c, Ns_dict_plot


def plot_quantitiy(plot_dict, lengt_dict, box_anc, box_anc_color, fig_name, legend_flag=False, log_plot_flag=True):
    colors = plt.cm.magma(np.linspace(0.9, 0, 5))
    linestyles = ["solid", "dashed", "dotted"]
    markers = ["o", "D", "s"]
    handles_dict = {"0": 0, "1": 0, "2": 0, "3": 0}
    fig, ax = plt.subplots(figsize=(9, 9))
    for key in plot_dict.keys():
        if key == "ri":
            idx = 0
        elif key == "RG":
            idx = 1
        else:
            idx = 2
        m = markers[idx]
        l = linestyles[idx]
        for error_key in plot_dict[key].keys():
            if error_key == "e-05":
                continue
                # lab = "$\epsilon=5 x 10^{-5}$"
            else:
                Ls = lengt_dict[key][error_key]
                plot_q = plot_dict[key][error_key]
                if error_key == "0001":
                    ix = 0
                    lab = "$\lambda=10^{-4}$"
                elif error_key == "0005":
                    ix = 1
                    lab = "$\lambda=5 x 10^{-4}$"
                elif error_key == ".003":
                    ix = 3
                    lab = "$\lambda=3x10^{-3}$"
                else:
                    ix = 2
                    lab = "$\lambda=10^{-3}$"
            ax.plot(Ls, plot_q, marker=m, linestyle=l, linewidth=2.5, markersize=7.5, color=colors[ix])
            if legend_flag:
                if m == markers[0]:
                    line1, = ax.plot(Ls[0], plot_q[0], color=colors[ix], linewidth=2.5, markersize=7.5, label=lab)
                    handles_dict[str(ix)] = line1
                    if idx == 0:
                        line1, = ax.plot(Ls[0], -10, marker=markers[0], linestyle=linestyles[0], markersize=5, color="black",
                                         label="Rings")
                        line2, = ax.plot(Ls[0], -10, marker=markers[1], linestyle=linestyles[1], markersize=4, color="black",
                                         label="RGS")
                        line3, = ax.plot(Ls[0], -10, marker=markers[2], linestyle=linestyles[2], markersize=4, color="black",
                                         label="Trees")
                        first_legend = ax.legend(handles=[line1, line2, line3], title="Linestyles", bbox_to_anchor=box_anc_color,
                                                 loc="lower left",
                                                 fontsize=12, title_fontsize=12)  # 0.6, 0.67
                        ax.add_artist(first_legend)

    if legend_flag:
        handles = []
        for key in handles_dict.keys():
            if handles_dict[key] == 0:
                continue
            else:
                handles.append(handles_dict[key])
        ax.legend(handles=handles, title="Error rate", bbox_to_anchor=box_anc, loc="lower left", fontsize=12,
                  title_fontsize=12)
    if log_plot_flag:
        ax.set_yscale("log")
    # ax.set_ylim((0, 350))
    plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.show()

    fig.savefig(fig_name, dpi=600)


def plot_emitters(plot_dict, lengt_dict, box_anc, box_anc_color, fig_name, legend_flag=False):
    colors = plt.cm.magma(np.linspace(0.9, 0, 5))
    linestyles = ["solid", "dashed", "dotted"]
    markers = ["o", "D", "s"]
    handles_dict = {"0": 0, "1": 0, "2": 0, "3": 0}
    fig, ax = plt.subplots(figsize=(9, 9))
    for key in plot_dict.keys():
        if key == "ri":
            idx = 0
        elif key == "RG":
            continue
        else:
            continue
        m = markers[idx]
        l = linestyles[idx]
        for error_key in plot_dict[key].keys():
            if error_key == "e-05":
                continue
                # lab = "$\epsilon=5 x 10^{-5}$"
            else:
                Ls = lengt_dict[key][error_key]
                plot_q = plot_dict[key][error_key]
                if error_key == "0001":
                    ix = 0
                    lab = "$\lambda=10^{-4}$"
                elif error_key == "0005":
                    ix = 1
                    lab = "$\lambda=5 x 10^{-4}$"
                elif error_key == ".003":
                    ix = 3
                    lab = "$\lambda=3x10^{-3}$"
                else:
                    ix = 2
                    lab = "$\lambda=10^{-3}$"
            ax.plot(Ls, plot_q, marker=m, linestyle=l, linewidth=2.5, markersize=7.5, color=colors[ix])
            if legend_flag:
                if m == markers[0]:
                    line1, = ax.plot(Ls[0], plot_q[0], color=colors[ix], linewidth=2.5, markersize=7.5, label=lab)
                    handles_dict[str(ix)] = line1

    if legend_flag:
        handles = []
        for key in handles_dict.keys():
            if handles_dict[key] == 0:
                continue
            else:
                handles.append(handles_dict[key])
        ax.legend(handles=handles, title="Error rate", bbox_to_anchor=box_anc, loc="lower left", fontsize=12,
                  title_fontsize=12)

    # ax.set_yscale("log")
    ax.set_ylim((3, 9))
    plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.show()

    fig.savefig(fig_name, dpi=600)


if __name__ == '__main__':
    # loss_and_error_pauli_meas(N_layer=6, normalized=True) # This is the one to use for log Pauli plots
    # logical_fusion_plot()
    # logical_Pauli_measurement()

    error_detect_log_fusions_updated_detect_individ_fail_traj(N=4, normalized=True) # This is the one to use for log fusion plots


    # path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\RepeaterPerformances\RatesMoreErrors\1ns_100_ns"
    # path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\RepeaterPerformances\RatesMoreErrors\1ns_100ns_long_distance"
    # path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\RepeaterPerformances\RatesMoreErrors\1ns_10ns_updated"
    # path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\RepeaterPerformances\RatesMoreErrors\1ns_10ns_long_distance_updated"



    # path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\RepeaterPerformances\RatesMoreErrors\1ns_100ns_last_iteration"
    # path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\RepeaterPerformances\RatesMoreErrors\1ns_10_ns_last_iteration"


    # path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\RepeaterPerformances\RatesMoreErrors\1ns_10_ns_update_last_layer"
    # path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\RepeaterPerformances\RatesMoreErrors\1ns_10_ns_long_distance_last_layer"
    # path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\RepeaterPerformances\RatesMoreErrors\1ns_100ns_last_layer"
    path = r"C:\Users\Admin\Desktop\ConcatenatedFusion\RepeaterPerformances\RatesMoreErrors\1ns_100ns_last_layer_long_distance"

    # L_s_init = [x for x in range(100, 1000, 20)]

    L_s_array = np.linspace(1000, 10000, 20)
    L_s_init = [x for x in L_s_array]

    # spares_flag = True  # Make it sparser for short distance
    spares_flag = False

    # length dict is for rates, and length_dict_m_c for everything else
    # rates_dict, stations_dict, cost_dict, lengt_dict, lengt_dict_m_c, Ns_dict = parse_all_info(path, L_s_init, spares_flag)

    # fig_name = 'CostFastGatesShortDistance.pdf'
    # fig_name = 'CostFastGatesLongDistance.pdf'
    # fig_name = 'CostSlowGatesShortDistance.pdf'
    fig_name = 'CostSlowGatesLongDistance.pdf'

    # Cost anchors
    box_anch = (0.61, 0.61)
    box_color_anch = (0.83, 0.68)

    # legend_flag = True
    legend_flag = False
    # plot_quantitiy(cost_dict, lengt_dict_m_c, box_anch, box_color_anch, fig_name, legend_flag)


    # fig_name = 'RatesFastGatesShortDistance.pdf'
    # fig_name = 'RatesFastGatesLongDistance.pdf'
    # fig_name = 'RatesSlowGatesShortDistance.pdf'
    fig_name = 'RatesSlowGatesLongDistance.pdf'


    # Rates anchors
    box_anch = (0.61, 0.0)
    box_color_anch = (0.83, 0.0)

    # legend_flag = True
    legend_flag = False

    # plot_quantitiy(rates_dict, lengt_dict, box_anch, box_color_anch, fig_name, legend_flag)



    # Stations anchors
    box_anch = (0.0, 0.61)
    box_color_anch = (0.22, 0.68)

    # fig_name = 'StationsFastGatesShortDistance.pdf'
    # fig_name = 'StationsFastGatesLongDistance.pdf'
    # fig_name = 'StationsSlowGatesShortDistance.pdf'
    fig_name = 'StationsSlowGatesLongDistance.pdf'

    # plot_quantitiy(stations_dict, lengt_dict_m_c, box_anch, box_color_anch, fig_name, legend_flag, log_plot_flag=False)

    # Emitters anchors
    box_anch = (0.7, 0.05)
    box_color_anch = (0.22, 0.68)

    # fig_name = 'EmittersFastGatesShortDistance.pdf'
    # fig_name = 'EmittersFastGatesLongDistance.pdf'
    # fig_name = 'EmittersSlowGatesShortDistance.pdf'
    fig_name = 'EmittersSlowGatesLongDistance.pdf'


    # plot_emitters(Ns_dict, lengt_dict_m_c, box_anch, box_color_anch, fig_name, legend_flag)
