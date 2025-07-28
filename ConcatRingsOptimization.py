import numpy as np
from RepeaterEquations import *
from Tree_analytics import RGS, p_succ_tree, fidelity_RGS
import math
from AnalyticformulasUpdateLastStrat import succ_ring_with_individ_det_with_fail_traj



def cost_func_rate(r, m, L, tau_p, N=3, L_att = 20):
    C = (1 / r) * (m * L_att / (L * tau_p)) * (N)
    return C


def cost_func_rate_tree_RGS(r, m, L, tau_p, L_att = 20):
    C = (1 / r) * (m * L_att / (L * tau_p))
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
    photon_gen = (1 + b0 * b1) * t_p
    E_and_CZ = b0 * (t_CZ + t_E)
    L = (photon_gen + E_and_CZ) * c
    return np.exp(-L / L_att)


def delay_loss_tree(b, t_gen, t_CZ, L_att=20000):
    c = 2 * (10 ** 8)
    delay = gen_time_tree(b[0], b[1], b[2], t_gen, t_CZ) / b[0]
    trans = np.exp(-c * delay / L_att)
    return trans



def optimize_cost_function(eps):

    # Parameters
    eta_d = 0.95
    t_CZ = 1 / (10 ** 8)
    t_gen = 1 / (10 ** 9)
    t_meas = 10 * t_gen
    Ls = [x for x in range(100, 1000, 20)]
    # Ls = np.linspace(1000, 10000, 20)
    # Ring parameters
    n = 4
    Ns = [4, 5, 6, 7]


    # Tree parameters
    b_t_0 = [3, 4, 5, 6, 7]
    b_t_1 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 25, 30, 35, 40, 45]
    b_t_2 = [3, 4, 5, 6, 7, 8]


    # RGS parameters
    N_qbts = [12, 16, 20, 24, 28, 32, 36, 40]
    b_0s = [12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40]
    b_1s = [4, 5, 6, 7, 8, 9, 10]

    # best parameters according to cost function
    parameters_ring = []
    parameters_tree = []
    parameters_rgs = []

    parameters_ring_r = []
    parameters_tree_r = []
    parameters_rgs_r = []
    for L in Ls:
        print("L = ", L)
        ms = [n for n in range(int((L / 10)), int(L / 2))]
        rates = []
        rates_RGS = []
        rates_tree = []


        max_rate = 0
        best_C = 10 ** 10
        best_m_C = 0
        best_N_C = 0

        best_r = 0
        best_r_m = 0
        best_r_N = 0

        best_C_tree = 10 ** 10
        best_m_C_tree = 0
        best_rate_C_tree = 0
        best_b_tree = 0

        best_r_tree = 0
        best_r_m_tree = 0
        best_r_graph_tree = 0

        best_m_C_RGS = 0
        best_C_RGS = 10 ** 10
        best_graph_RGS = 0
        best_rate_RGS = 0

        best_r_RGS = 0
        best_r_m_RGS = 0
        best_r_graph_RGS = 0

        for m in ms:
            fiber_trans = fiber_transmission(m, L)
            fiber_trans_tree = fiber_transmission_tree(m, L)
            eta = photon_transmission(eta_d, fiber_trans)


            error_tree = eps * 3
            fid_tree = (1 - error_tree) ** (m + 1)

            for b0 in b_t_0:
                for b1 in b_t_1:
                    for b2 in b_t_2:
                        if b0 * b1 * b2 > 30000:
                            continue
                        else:
                            b = [b0, b1, b2]
                            delay_transsmission_tree = delay_loss_tree(b, t_gen, t_CZ)
                            eta_tree = photon_transmission(eta_d, fiber_trans_tree) * delay_transsmission_tree
                            succ_tree = p_succ_tree(eta_tree, b)
                            p_trans_tree = succ_tree ** (m + 1)

                            r_tree = key_siphing(fid_tree) * p_trans_tree / gen_time_tree(b[0], b[1], b[2], t_gen, t_CZ)
                            rates_tree.append(r_tree)
                            C = cost_func_rate_tree_RGS(r_tree, m, L, t_gen)

                            if C < best_C_tree:
                                best_C_tree = C
                                best_rate_C_tree = r_tree
                                best_m_C_tree = m
                                best_b_tree = b
                            if r_tree > best_r_tree:
                                best_r_tree = r_tree
                                best_r_m_tree = m
                                best_r_graph_tree = b

            for N_qbt in N_qbts:
                for b0 in b_0s:
                    for b_1 in b_1s:
                        r_0_RGS = generation_time_RGS(b0, b_1, t_CZ, t_meas, t_gen, N_qbt)
                        RGS_delay_transmission = delay_loss_RGS(b0, b_1, t_gen, t_meas, t_CZ)
                        RGS_succ = RGS(eta * RGS_delay_transmission, N_qbt, [b0, b_1])
                        p_trans_RGS = RGS_succ ** (m + 1)

                        fid_RGS = fidelity_RGS(m, N_qbt, 2 * eps / 3, eta * RGS_delay_transmission, b0, b_1)
                        f = key_siphing(fid_RGS)
                        rates_RGS.append(f * rate(r_0_RGS, p_trans_RGS))
                        r = f * rate(r_0_RGS, p_trans_RGS)
                        C = cost_func_rate_tree_RGS(f * rate(r_0_RGS, p_trans_RGS), m, L, t_gen)
                        if C < best_C_RGS:
                            best_C_RGS = C
                            best_rate_RGS = f * rate(r_0_RGS, p_trans_RGS)
                            best_m_C_RGS = m
                            best_graph_RGS = [N_qbt, b0, b_1]
                        if r > best_r_RGS:
                            best_r_RGS = r
                            best_r_m_RGS = m
                            best_r_graph_RGS = [N_qbt, b0, b_1]

            r_ring = 0
            for N in Ns:
                t_0 = generation_time(N, n, t_CZ, t_meas, t_gen)
                p_succ, err_detect, log_error = succ_ring_with_individ_det_with_fail_traj(N, 2 * eps / 3, eta) 
                fid_ring = (1 - log_error) ** (m + 1)
                detect_abort = (1 - err_detect) ** (m + 1)
                p_trans = p_succ ** (m + 1)
                r = key_siphing(fid_ring) * rate(t_0, p_trans) * detect_abort
                rates.append(r)
                C = cost_func_rate(r, m, L, t_gen, N)

                if C < best_C:
                    best_C = C
                    best_rate = r
                    best_m_C = m
                    best_N_C = N
                if r > best_r:
                    best_r = r
                    best_r_m = m
                    best_r_N = N

        print("Best parameters ring: ", best_rate, best_m_C, best_N_C, best_C)
        print("Best parameters tree: ", best_rate_C_tree, best_m_C_tree, best_C_tree, best_b_tree)
        print("Best parameter RGS: ", best_rate_RGS, best_m_C_RGS, best_C_RGS, best_graph_RGS)
        print()
        print("Best parameters ring: ", best_r, best_r_m, best_r_N)
        print("Best parameters tree: ", best_r_tree, best_r_m_tree, best_r_graph_tree)
        print("Best parameter RGS: ", best_r_RGS, best_r_m_RGS, best_r_graph_RGS)
        parameters_ring.append([best_rate, best_m_C, best_N_C, best_C])
        parameters_ring_r.append([best_r, best_r_m, best_r_N])
        if best_rate_C_tree > 1:
            parameters_tree.append([best_rate_C_tree, best_m_C_tree, best_C_tree])
            parameters_tree.append(best_b_tree)
        if best_r_tree > 1:
            parameters_tree_r.append([best_r_tree, best_r_m_tree, 0])
            parameters_tree_r.append(best_r_graph_tree)
        if best_rate_RGS > 1:
            parameters_rgs.append([best_rate_RGS, best_m_C_RGS, best_C_RGS])
            parameters_rgs.append(best_graph_RGS)
        if best_r_RGS > 1:
            parameters_rgs_r.append([best_r_RGS, best_r_m_RGS, 0])
            parameters_rgs_r.append(best_r_graph_RGS)




    tree_rates = np.array(parameters_tree)
    np.savetxt('tree_rate_' + str(eps) + '_' + '.txt', tree_rates)
    RGS_rates = np.array(parameters_rgs)
    np.savetxt('RGS_rate_' + str(eps) + '_' + '.txt', RGS_rates)
    ring_rates = np.array(parameters_ring)
    np.savetxt('ring_rate_' + str(eps) + '_' + '.txt', ring_rates)

    tree_rates = np.array(parameters_tree_r)
    np.savetxt('tree_rate_rate_' + str(eps) + '_' + '.txt', tree_rates)
    RGS_rates = np.array(parameters_rgs_r)
    np.savetxt('RGS_rate_rate_' + str(eps) + '_' + '.txt', RGS_rates)
    ring_rates = np.array(parameters_ring_r)
    np.savetxt('ring_rate_rate_' + str(eps) + '_' + '.txt', ring_rates)


def optimize_cost_function_ring(eps):
    print("Eps: ", eps)
    # Parameters
    eta_d = 0.95
    t_CZ = 10 / (10 ** 8)
    t_gen = 1 / (10 ** 9)
    t_meas = 10 * t_gen
    # Ls = [x for x in range(100, 1000, 20)]
    Ls = np.linspace(1000, 10000, 20)
    # Ring parameters
    n = 4
    Ns = [4, 5, 6, 7]




    # best parameters according to cost function
    parameters_ring = []

    parameters_ring_r = []
    for L in Ls:
        print("L = ", L)
        ms = [n for n in range(int((L / 10)), int(L / 2))]
        rates = []
        rates_RGS = []
        rates_tree = []


        max_rate = 0
        best_C = 10 ** 10
        best_m_C = 0
        best_N_C = 0

        best_r = 0
        best_r_m = 0
        best_r_N = 0
        best_rate = 0


        for m in ms:
            fiber_trans = fiber_transmission(m, L)
            eta = photon_transmission(eta_d, fiber_trans)



            r_ring = 0
            for N in Ns:
                t_0 = generation_time(N, n, t_CZ, t_meas, t_gen)
                p_succ, err_detect, log_error = succ_ring_with_individ_det_with_fail_traj(N, 2 * eps / 3, eta) 
                fid_ring = (1 - log_error) ** (m + 1)
                detect_abort = (1 - err_detect) ** (m + 1)
                p_trans = p_succ ** (m + 1)
                r = key_siphing(fid_ring) * rate(t_0, p_trans) * detect_abort
                rates.append(r)
                C = cost_func_rate(r, m, L, t_gen, N)

                if C < best_C:
                    best_C = C
                    best_rate = r
                    best_m_C = m
                    best_N_C = N
                if r > best_r:
                    best_r = r
                    best_r_m = m
                    best_r_N = N
        print("Best parameters ring: ", best_rate, best_m_C, best_N_C, best_C)
        print()
        print("Best parameters ring: ", best_r, best_r_m, best_r_N)
        parameters_ring.append([best_rate, best_m_C, best_N_C, best_C])
        parameters_ring_r.append([best_r, best_r_m, best_r_N])





    ring_rates = np.array(parameters_ring)
    np.savetxt('ring_rate_' + str(eps) + '_' + '.txt', ring_rates)


    ring_rates = np.array(parameters_ring_r)
    np.savetxt('ring_rate_rate_' + str(eps) + '_' + '.txt', ring_rates)

def optimize_cost_function_ring_opt_error_lay(eps):
    print("Eps: ", eps)
    # Parameters
    eta_d = 0.95
    t_CZ = 1 / (10 ** 8)
    t_gen = 1 / (10 ** 9)
    t_meas = 10 * t_gen
    Ls = [x for x in range(100, 1000, 20)]
    # Ls = np.linspace(1000, 10000, 20)
    # Ring parameters
    n = 4
    Ns = [4, 5, 6, 7]




    # best parameters according to cost function
    parameters_ring = []

    parameters_ring_r = []
    for L in Ls:
        print("L = ", L)
        ms = [n for n in range(int((L / 10)), int(L / 2))]
        rates = []
        rates_RGS = []
        rates_tree = []


        max_rate = 0
        best_C = 10 ** 10
        best_m_C = 0
        best_N_C = 0

        best_r = 0
        best_r_m = 0
        best_r_N = 0
        best_rate = 0
        best_N_E = 0


        for m in ms:
            fiber_trans = fiber_transmission(m, L)
            eta = photon_transmission(eta_d, fiber_trans)



            r_ring = 0
            for N in Ns:
                for N_E in range(1, N):
                    t_0 = generation_time(N, n, t_CZ, t_meas, t_gen)
                    p_succ, err_detect, log_error = succ_ring_with_individ_det_with_fail_traj(N, 2 * eps / 3, eta, N_first_layers=N_E) # succ_ring_with_individ_det(N, 2 * eps / 3, eta)
                    fid_ring = (1 - log_error) ** (m + 1)
                    detect_abort = (1 - err_detect) ** (m + 1)
                    p_trans = p_succ ** (m + 1)
                    # print("fidelity: ", fid_ring, m, log_error, 2 * eps / 3, eta, N)
                    r = key_siphing(fid_ring) * rate(t_0, p_trans) * detect_abort
                    rates.append(r)
                    C = cost_func_rate(r, m, L, t_gen, N)

                    # if C < best_C and r >= r_ring:
                    if C < best_C:
                        best_C = C
                        best_rate = r
                        best_m_C = m
                        best_N_C = N
                        best_N_E = N_E
                    if r > best_r:
                        best_r = r
                        best_r_m = m
                        best_r_N = N
        print("Best parameters ring: ", best_rate, best_m_C, best_N_C, best_C, best_N_E)
        print()
        # print("Best rate ring: ", best_r, best_r_m, best_r_N)
        parameters_ring.append([best_rate, best_m_C, best_N_C, best_C, best_N_E])
        # parameters_ring_r.append([best_r, best_r_m, best_r_N])





    ring_rates = np.array(parameters_ring)  # Saving best cost func
    np.savetxt('ring_rate_' + str(eps) + '_' + '.txt', ring_rates)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    epsilons = [0.003, 0.0001, 0.0005, 0.001]
    for eps in epsilons:
        # optimize_cost_function(eps)
        optimize_cost_function_ring(eps)
