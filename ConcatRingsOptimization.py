import numpy as np
from RepeaterEquations import generation_time, key_siphing
from RateAnalisys import succ_ring, succ_error_ring, generation_time_RGS, gen_time_tree
from Tree_analytics import RGS, p_succ_tree, fidelity_RGS
from Analyticformulas import parity_flip
import math

def cost_function(r_0, f, p_trans, m, tau_p, L, L_att=20 * (10**3)):
    # r_0 = generation time
    # f = binary fid. function
    # p_trans = p_succ**m
    # m = number of repeater stations
    # tau_p = generation time of a photonic qubit
    # L = Repeater distance in km
    # L_att = attneuation length in km
    L_m = L * (10 ** 3)
    C = (1 / (r_0 * f * p_trans)) * ((m * L_att) / (tau_p * L))
    return C


def cost_func_rate(r, m, L, tau_p, N=3, L_att = 20):
    C = (1 / r) * (m * L_att / (L * tau_p)) * N
    return C


def rate(r_0, p_trans):
    return p_trans / r_0


def photon_transmission(eta_d, fiber_trans):
    return eta_d * fiber_trans


def fiber_transmission(m, L, L_att=20):
    L_0 = (L / (m+1)) / 2  # Divided by two because thye meet in the middle
    return np.exp(-L_0 / L_att)


def fiber_transmission_tree(m, L, L_att=20):
    L_0 = L / (m+1)
    return np.exp(-L_0 / L_att)


def delay_loss_RGS(b0, b1, t_p, t_M, t_CZ, L_att = 20000):
    t_E = t_p + t_M
    c = 2 * (10 ** 8)
    photon_gen = (1 + b0 * b1) * t_gen
    E_and_CZ = b0 * (t_CZ + t_E)
    L = (photon_gen + E_and_CZ) * c
    return np.exp(-L / L_att)


def delay_loss_tree(b, t_gen, t_CZ, L_att=20000):
    c = 2 * (10 ** 8)
    delay = gen_time_tree(b[0], b[1], b[2], t_gen, t_CZ) / b[0]
    trans = np.exp(-c * delay / L_att)
    return trans



def binary_tree(x):
    return -x * math.log(x) - (1 - x) * math.log(1 - x)

def siphing_tree(Q):
    return 1 - binary_tree(Q) - Q - (1 - Q) * binary_tree((1 - 3 * Q / 2) / (1 - Q))



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # TODO: Add errors of single-qbuit measurements in trees.
    # DONE: Add delay loss for RGS and trees



    ################################################################################
    ###### Brute force test for L = 1000 km, no error, cutoff at L_0 = 1km #########
    ################################################################################

    n = 4
    N = 3
    eta_d = 0.95
    t_CZ = 1 / (10 ** 8)
    t_gen = 1 / (10 ** 9)
    t_meas = 10 * t_gen
    L = 1000
    ms = [n for n in range(int((L / 10) * 3), L)]
    t_0 = generation_time(N, n, t_CZ, t_meas, t_gen)
    cost_plot = []
    rates = []
    for m in ms:
        fiber_trans = fiber_transmission(m, L)
        eta = photon_transmission(eta_d, fiber_trans)
        p_succ = succ_ring(N, eta)
        p_trans = p_succ ** (m + 1)
        C = cost_function(t_0, 1, p_trans, m, t_gen, L)
        cost_plot.append(C)
        r = rate(t_0, p_trans)
        rates.append(r)


    idx = cost_plot.index(min(cost_plot))
    min_value = min(cost_plot)
    fiber_trans = fiber_transmission(ms[idx], L)
    eta = photon_transmission(eta_d, fiber_trans)
    print(min_value, idx, ms[idx], fiber_trans, eta, succ_ring(N, eta), succ_ring(N, eta) ** (ms[idx] + 1))
    plt.plot(ms, cost_plot)
    plt.plot(ms, rates)
    plt.yscale("log")
    plt.show()
    '''

    ####################################################################
    ############ Optimum ring vs RGS without errors ####################
    ####################################################################
    n = 4
    N = 3
    eta_d = 0.95
    t_CZ = 1 / (10 ** 8)
    t_gen = 1 / (10 ** 9)
    t_meas = 10 * t_gen
    t_0 = generation_time(N+1, n, t_CZ, t_meas, t_gen)
    Ls = [x for x in range(200, 1000, 10)]
    final_rate_plot = []
    final_rate_plot_RGS = []
    optimum_stations = []
    # N_qbts = 32
    N_qbts = [16, 24, 32]
    branch_list = [24, 7]
    # r_0_RGS = generation_time_RGS(branch_list[0], branch_list[1], t_CZ, t_meas, t_gen, N_qbts)
    for L in Ls:
        print("L = ", L)
        ms = [n for n in range(int((L / 10) * 3), L)]
        cost_plot = []
        rates = []
        rates_RGS = []
        for m in ms:
            fiber_trans = fiber_transmission(m, L)
            eta = photon_transmission(eta_d, fiber_trans)
            p_succ = succ_ring(N, eta)
            p_trans = p_succ ** (m + 1)
            C = cost_function(t_0, 1, p_trans, m, t_gen, L)
            cost_plot.append(C)
            r = rate(t_0, p_trans)
            rates.append(r)
            for N_qbt in N_qbts:
                r_0_RGS = generation_time_RGS(branch_list[0], branch_list[1], t_CZ, t_meas, t_gen, N_qbt)
                RGS_succ = RGS(eta, N_qbt, branch_list)
                p_trans_RGS = RGS_succ ** (m + 1)
                rates_RGS.append(rate(r_0_RGS, p_trans_RGS))
        final_rate_plot.append(max(rates))
        final_rate_plot_RGS.append(max(rates_RGS))
        idx = rates.index(max(rates))
        # idx = rates_RGS.index(max(rates_RGS))
        optimum_stations.append(ms[idx])
    plt.plot(Ls, final_rate_plot)
    plt.plot(Ls, final_rate_plot_RGS)
    plt.yscale("log")
    plt.show()


    ####################################################################
    ############ Optimum ring vs RGS different branch and tree #########
    ####################################################################
    colors = plt.cm.viridis(np.linspace(0.9, 0, 3))
    n = 4
    N = 3
    eta_d = 0.95
    t_CZ = 1 / (10 ** 8)
    # t_gen = 1 / (10 ** 9)
    # t_gens = [1 / (10 ** 9), 10 / (10 ** 9), 50 / (10 ** 9)]
    t_gens = [1 / (10 ** 9)]
    for ix, t_gen in enumerate(t_gens):
        t_meas = 10 * t_gen
        t_0 = generation_time(N+1, n, t_CZ, t_meas, t_gen)
        Ls = [x for x in range(200, 1000, 20)]
        final_rate_plot = []
        final_rate_plot_RGS = []
        final_rate_trees = []
        optimum_stations = []
        b = [4, 14, 4]
        N_qbts = [12, 16, 20, 24, 28, 32]
        branch_list = [24, 7]
        b_0s = [12, 14, 16, 18, 20, 22]
        b_1s = [4, 5, 6, 7, 8, 9]
        optimal_parameters = []
        # r_0_RGS = generation_time_RGS(branch_list[0], branch_list[1], t_CZ, t_meas, t_gen, N_qbts)
        for L in Ls:
            print("L = ", L)
            ms = [n for n in range(int((L / 10) * 3), L)]
            cost_plot = []
            rates = []
            rates_RGS = []
            rates_tree = []
            parameters_RGS = {}
            cnt = 0
            for m in ms:
                fiber_trans = fiber_transmission(m, L)
                fiber_trans_tree = fiber_transmission_tree(m, L)
                eta = photon_transmission(eta_d, fiber_trans)
                eta_tree = photon_transmission(eta_d, fiber_trans_tree)
                # print(fiber_trans_tree, eta_tree, b)
                succ_tree = p_succ_tree(eta_tree, b)
                p_succ = succ_ring(N, eta)
                p_trans = p_succ ** (m + 1)
                p_trans_tree = succ_tree ** (m + 1)
                r_tree = p_trans_tree / gen_time_tree(b[0], b[1], b[2], t_gen, t_CZ)
                rates_tree.append(r_tree)
                C = cost_function(t_0, 1, p_trans, m, t_gen, L)
                cost_plot.append(C)
                r = rate(t_0, p_trans)
                rates.append(r)

                for N_qbt in N_qbts:
                    for b0 in b_0s:
                        for b_1 in b_1s:
                            r_0_RGS = generation_time_RGS(b0, b_1, t_CZ, t_meas, t_gen, N_qbt)
                            RGS_succ = RGS(eta, N_qbt, [b0, b_1])
                            p_trans_RGS = RGS_succ ** (m + 1)
                            rates_RGS.append(rate(r_0_RGS, p_trans_RGS))
                            parameters_RGS[str(cnt)] = [N_qbt, b0, b_1, m]
                            cnt += 1
            final_rate_plot.append(max(rates))
            final_rate_plot_RGS.append(max(rates_RGS))
            final_rate_trees.append(max(rates_tree))
            idx = rates.index(max(rates))
            idx_RGS = rates_RGS.index(max(rates_RGS))
            optimal_parameters.append(parameters_RGS[str(idx)])
            optimum_stations.append(ms[idx])
        print(optimal_parameters)
        plt.plot(Ls, final_rate_plot, "-o",color=colors[ix])
        plt.plot(Ls, final_rate_plot_RGS, "--D", color=colors[ix+1])
        plt.plot(Ls, final_rate_trees, color=colors[ix+2], marker="s", linestyle="dotted")

    plt.legend()
    plt.yscale("log")
    plt.show()
    '''
    ###################################################################################
    ############ Optimum ring vs RGS different branch and tree  with errors ###########
    ###################################################################################

    eps = 0.00005  # Depolarizing error rate
    colors = plt.cm.viridis(np.linspace(0.9, 0, 3))
    # epsilons = [0.00005, 0.0001, 0.0005, 0.001]
    epsilons = [
        0.00004]
    for ix, eps in enumerate(epsilons):
        n = 4
        # N = 4
        Ns = [4, 5, 6, 7]
        eta_d = 0.95
        t_CZ = 1 / (10 ** 8)
        t_gen = 1 / (10 ** 9)
        t_meas = 10 * t_gen
        Ls = [x for x in range(760, 1000, 20)]
        final_rate_plot = []
        final_rate_plot_RGS = []
        final_rate_trees = []
        optimum_stations = []
        b = [4, 14, 4]
        N_qbts = [12, 16, 20, 24, 28, 32, 36]
        branch_list = [24, 7]
        b_0s = [12, 14, 16, 18, 20, 22, 24, 28, 32]
        b_1s = [4, 5, 6, 7, 8, 9, 10]
        optimal_parameters = []
        plt_C_best = []
        # r_0_RGS = generation_time_RGS(branch_list[0], branch_list[1], t_CZ, t_meas, t_gen, N_qbts)
        parameters_ring = []
        parameters_tree = []
        for L in Ls:
            print("L = ", L)
            ms = [n for n in range(int((L / 10)), L)]
            cost_plot = []
            rates = []
            rates_RGS = []
            rates_tree = []

            parameters_RGS = {}
            cnt = 0
            max_N = 0
            max_rate = 0
            best_C = 10 ** 10
            best_m_C = 0
            best_N_C = 0
            best_m_C_RGS = 0
            best_C_RGS = 10 ** 10
            cost_f_tree = []
            for m in ms:
                fiber_trans = fiber_transmission(m, L)
                fiber_trans_tree = fiber_transmission_tree(m, L)
                delay_transsmission_tree = delay_loss_tree(b, t_gen, t_CZ)
                eta = photon_transmission(eta_d, fiber_trans)
                eta_tree = photon_transmission(eta_d, fiber_trans_tree) * delay_transsmission_tree
                succ_tree = p_succ_tree(eta_tree, b)
                if ix < 3:
                    error_tree = eps * 3
                else:
                    error_tree = eps * 5
                # error_RGS = parity_flip(2 * eps / 3)
                # fid_RGS = (1 - error_RGS) ** (m + 1)
                fid_tree = (1 - error_tree) ** (m + 1)

                p_trans_tree = succ_tree ** (m + 1)


                r_tree = key_siphing(fid_tree) * p_trans_tree / gen_time_tree(b[0], b[1], b[2], t_gen, t_CZ)
                rates_tree.append(r_tree)
                # C = cost_function(gen_time_tree(b[0], b[1], b[2], t_gen, t_CZ), key_siphing(fid_tree), p_trans_tree, m, t_gen, L)
                C = cost_func_rate(r_tree, m, L, t_gen)
                cost_f_tree.append(C)
                for N_qbt in N_qbts:
                    for b0 in b_0s:
                        for b_1 in b_1s:
                            r_0_RGS = generation_time_RGS(b0, b_1, t_CZ, t_meas, t_gen, N_qbt)
                            RGS_delay_transmission = delay_loss_RGS(b0, b_1, t_gen, t_meas, t_CZ)
                            RGS_succ = RGS(eta * RGS_delay_transmission, N_qbt, [b0, b_1])
                            p_trans_RGS = RGS_succ ** (m + 1)
                            # if p_trans_RGS > 0.99:
                            fid_RGS = fidelity_RGS(m, N_qbt, 2 * eps / 3, eta * RGS_delay_transmission, b0, b_1)
                            f = key_siphing(fid_RGS)
                            rates_RGS.append(f * rate(r_0_RGS, p_trans_RGS))
                            parameters_RGS[str(cnt)] = [N_qbt, b0, b_1, m, RGS_succ, p_trans_RGS,f * rate(r_0_RGS, p_trans_RGS)]
                            C = cost_func_rate(f * rate(r_0_RGS, p_trans_RGS), m, L, t_gen)
                            cnt += 1
                            if C < best_C_RGS:
                                best_C_RGS = C
                                best_rate_RGS = f * rate(r_0_RGS, p_trans_RGS)
                                best_m_C_RGS = m
                for N in Ns:
                    t_0 = generation_time(N, n, t_CZ, t_meas, t_gen)
                    p_succ, err_detect, log_error = succ_error_ring(N, 2 * eps / 3, eta)
                    fid_ring = (1 - log_error) ** (m + 1)
                    detect_abort = (1 - err_detect) ** (m + 1)
                    p_trans = p_succ ** (m + 1)
                    r = key_siphing(fid_ring) * rate(t_0, p_trans) * detect_abort
                    rates.append(r)
                    C = cost_func_rate(r, m, L, t_gen, N)
                    if r > max_rate:
                        max_rate = r
                        max_N = N
                        max_reap = m
                    if C < best_C:
                        best_C = C
                        best_rate = r
                        best_m_C = m
                        best_N_C = N
            idx_tree = rates_tree.index(max(rates_tree))
            idx_min = cost_f_tree.index(min(cost_f_tree))
            print(max(rates), max(rates_RGS), max(rates_tree), max_N, max_reap, ms[idx_tree])
            print(ms[idx_min], rates_tree[idx_min], best_rate, best_m_C, best_N_C, best_rate_RGS, best_m_C_RGS)
            print("Cost optimal: ", cost_func_rate(best_rate, best_m_C, L, t_gen))
            print("Rate optimal: ", cost_func_rate(max(rates), max_reap, L, t_gen))
            parameters_ring.append([max_reap, max_N, max(rates), best_m_C, best_N_C, best_rate])
            parameters_tree.append([ms[idx_tree], max(rates_tree), ms[idx_min], rates_tree[idx_min]])
            final_rate_plot.append(max(rates))
            final_rate_plot_RGS.append(max(rates_RGS))
            max_tree = max(rates_tree)
            if max_tree < 1:
                max_tree = 0
            final_rate_trees.append(max_tree)
            idx = rates.index(max(rates))
            idx_RGS = rates_RGS.index(max(rates_RGS))
            optimal_parameters.append(parameters_RGS[str(idx_RGS)])
            print(parameters_RGS[str(idx_RGS)])
            plt_C_best.append(best_rate)

            # optimum_stations.append(ms[idx])

        tree_rates = np.array(final_rate_trees)
        np.savetxt('tree_rate_' + str(eps) + '_' + '.txt', tree_rates)
        RGS_rates = np.array(final_rate_plot_RGS)
        np.savetxt('RGS_rate_' + str(eps) + '_' + '.txt', RGS_rates)
        ring_rates = np.array(final_rate_plot)
        np.savetxt('ring_rate_' + str(eps) + '_' + '.txt', ring_rates)

        ring_p = np.array(parameters_ring)
        np.savetxt('ring_p.txt', ring_p)
        tree_p = np.array(parameters_tree)
        np.savetxt('tree_p.txt', tree_p)
        print(optimal_parameters)

        plt.plot(Ls, plt_C_best, color="red")
        plt.plot(Ls, final_rate_plot, color="black")
        plt.show()
    # plt.plot(Ls, final_rate_plot, "-o", color=colors[0])
    # plt.plot(Ls, final_rate_plot_RGS, "--D", color=colors[1])
    # plt.plot(Ls, final_rate_trees, color=colors[2], marker="s", linestyle="dotted")

    # plt.legend()
    # plt.yscale("log")
    # plt.show()






