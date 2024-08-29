from Analyticformulas import *
from RepeaterEquations import *
import numpy as np
import matplotlib.pyplot as plt
from Tree_analytics import *

# Tree's assumes 100 times longer decay rate for ceration photons

def log_fusion_spin_gates_modified(N, eta_init):
    sing_trans = log_transmission(eta_init)
    sing_trans = log_transmission(sing_trans)
    spin_log_succ, spin_fail, spin_lost = log_fusion_prob_spin_gates(eta_init, 0, 0, 0, 1 - eta_init, sing_trans)
    spin_log_succ, spin_fail, spin_lost = log_fusion_prob_spin_gates(spin_log_succ, 0, 0, 0, spin_lost, sing_trans)
    print("Sing trans: ", sing_trans)
    return spin_log_succ

def log_fusion_prob_spin_gates(p_s, p_f_x, p_f_y, p_f_z, p_l, sing_trans):
    term_one = p_s * ((sing_trans ** 3 + 3 * (1 - sing_trans) * (sing_trans ** 2)))
    term_two = p_f_x * p_s * ((sing_trans ** 2 + 2 * (1 - sing_trans) * sing_trans))
    term_three = p_l * p_s * (sing_trans ** 2 + p_f_y * (sing_trans))
    term_four = (p_f_x ** 2) * p_s * (sing_trans + p_f_z)
    log_success = term_one + term_two + term_three + term_four
    p_failure = p_s * (1-sing_trans) * sing_trans * sing_trans
    return log_success, 0, 1 - log_success
    # return log_success, p_failure, 1- log_success - 3 * p_failure


def delay_loss(b, t_gen, t_CZ):
    speed_of_light = 2 * (10 ** 8)
    delay = gen_time_tree(b[0], b[1], b[2], t_gen, t_CZ) / b[0]
    trans = np.exp(-speed_of_light * delay / 20000)
    return trans

def succ_ring_spin(N, eta_init):
    spin_log_succ, spin_fail, spin_lost = log_fusion_prob_spin_gates(eta_init, 0, 0, 0, 1 - eta_init, eta_init)
    sing_trans = log_transmission(eta_init)
    for _ in range(N-1):
        spin_log_succ, spin_fail, spin_lost = log_fusion_prob_spin_gates(spin_log_succ, spin_fail, spin_fail, spin_fail, spin_lost, sing_trans)
        sing_trans = log_transmission(sing_trans)
    return spin_log_succ


def succ_error_ring(N, eps, eta_init):
    log_succ_error, error_detection_prob, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f_up, eta_up = \
        log_fusion_error_prob(eps, 0, failed_flip(eps), failed_flip(eps), failed_flip(eps), parity_flip(eps),
                              (1 / 2) * (eta_init ** 2), (1 / 2) * (eta_init ** 2), (1 / 2) * (eta_init ** 2),
                              (1 / 2) * (eta_init ** 2), 1 - eta_init ** 2, eta_init)

    # log_succ_error, error_detection_prob, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f_up, eta_up = \
    #     log_fusion_error_prob(eps, 0, failed_flip(eps), failed_flip(eps), failed_flip(eps), parity_flip(eps),
    #                           eta_init, 0, 0,
    #                          0, 1 - eta_init, eta_init)
    init_eps_f = intial_eps_f_with_loss(eps, eta_init)
    sing_trans = log_transmission(eta_init)
    epsilon_up = epsilon_up
    epsilon_f = init_eps_f
    err_detect = 0
    log_lost = 1 - log_succ_this_layer - log_fail_x_this_layer - log_fail_y_this_layer - log_fail_z_this_layer


    for _ in range(2):
        log_succ_error, error_detection_prob, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f, sing_trans = \
            log_fusion_error_prob(epsilon_up, epsilon_f, log_p_fail_x_this_layer, log_p_fail_y_this_layer,
                                  log_p_fail_z_this_layer, log_succ_error, log_succ_this_layer, log_fail_x_this_layer,
                                  log_fail_y_this_layer, log_fail_z_this_layer, log_lost, sing_trans)
        err_detect += error_detection_prob
        log_lost = 1 - log_succ_this_layer - log_fail_x_this_layer - log_fail_y_this_layer - log_fail_z_this_layer

    log_lost = 1 - log_succ_this_layer
    for _ in range(N - 3):
        log_succ_error, err_detect, log_succ_this_layer, log_lost, epsilon_up, epsilon_f, sing_trans = fault_tolerant_fusion_layers_error(
            log_succ_this_layer, log_lost, sing_trans, log_succ_error, err_detect, epsilon_up, epsilon_f)
    return log_succ_this_layer, err_detect, log_succ_error


def gen_time_tree(b0, b1, b2, t_gen, t_CZ):
    generation_time = b0 * (100 + b1 * (1 + b2)) * t_gen
    control_phase_time = b0 * (3 + b1) * t_CZ
    return generation_time + control_phase_time

def succ_ring(N, eta):
    log_succ = log_fusion_prob((1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2),
                               (1 / 2) * (eta ** 2), 1 - eta ** 2, eta)
    log_p_fail_x, log_p_fail_z = log_failure((1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2),
                                             (1 / 2) * (eta ** 2), 1 - eta ** 2, eta)
    log_p_fail_y = 0
    sing_trans = eta
    # log_p_fail_x, log_p_fail_z = 0, 0
    for _ in range(N):
        sing_trans = log_transmission(sing_trans)
        log_lost = 1 - log_succ - log_p_fail_z - log_p_fail_x
        log_succ_copy = copy.deepcopy(log_succ)
        log_succ = log_fusion_prob(log_succ, log_p_fail_x,
                                   log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
        log_p_fail_x, log_p_fail_z = log_failure(log_succ_copy, log_p_fail_x,
                                                 log_p_fail_y, log_p_fail_z, log_lost, sing_trans)

    return log_succ


def generation_time_RGS(b_0, b_1, t_CZ, t_meas, t_gen, N):
    gen_t = (1 + b_0 * b_1) * t_gen
    CZ_t = (2 + b_0) * t_CZ
    meas_E_t = (b_0 + 2) * t_meas
    t = N * (gen_t + CZ_t + meas_E_t) + t_meas
    return t



def generation_time_RGS_no_meas(b_0, b_1, t_CZ, t_meas, t_gen, N):
    gen_t = (1 + b_0 * b_1) * t_gen
    CZ_t = (2 + b_0) * t_CZ
    meas_E_t = (b_0 + 2) * t_meas
    t = N * (gen_t + CZ_t)
    return t


def RGS_rough_estimates(n_stations, eps, log_succ, t_CZ, t_meas, t_gen, b_0=24, b_1=7, N=32):
    p_s = log_succ ** n_stations
    fusion_error = parity_flip(eps)
    # numb_CZ = 768
    # num_meas = 768
    # numb_photons = 23000
    # t = numb_CZ * t_CZ + num_meas * t_meas + numb_photons * t_gen
    t = generation_time_RGS(b_0, b_1, t_CZ, t_meas, t_gen, N)
    fid = (1 - fusion_error) ** n_stations
    rate = key_siphing(fid) * p_s * (1 / t)
    return rate


def compare_RGS():
    Ls = np.linspace(500, 12000, 100)
    L_att = 20
    N = 4
    n = 4
    eta_init = 0.9 # 0.86
    t_CZ = 10 / (10**9)
    t_meas = 10 / (10 ** 9)
    t_gen = 1 / (10 ** 9)
    eps = 2 * 0.00003 / 3
    # eps_borr_red = (2 * (0.005 / 10) / 5) / 3
    detection_inefficiency = 0.05
    rate = []
    RGS_trans = 0.86
    N_qbts = 32
    branch_list = [24, 7]
    RGS_succ = RGS(RGS_trans, N_qbts, branch_list)
    RGS_rate = []
    for L in Ls:
        n_stations = number_of_stations(L, eta_init + detection_inefficiency)
        n_stations_RGS = number_of_stations(L, RGS_trans + detection_inefficiency)
        log_succ_error, error_detection_prob, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f_up, eta_up = \
            log_fusion_error_prob(eps, 0, failed_flip(eps), failed_flip(eps), failed_flip(eps), parity_flip(eps), (1 / 2) * (eta_init ** 2), (1 / 2) * (eta_init ** 2) , (1 / 2) * (eta_init ** 2), (1 / 2) * (eta_init ** 2), 1 - eta_init ** 2, eta_init)
        init_eps_f = intial_eps_f_with_loss(eps, eta_init)
        sing_trans = log_transmission(eta_init)
        epsilon_up = epsilon_up
        epsilon_f = init_eps_f
        err_detect = 0
        log_lost = 1 - log_succ_this_layer - log_fail_x_this_layer - log_fail_y_this_layer - log_fail_z_this_layer
        for _ in range(3):
            log_succ_error, error_detection_prob, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f, sing_trans = \
                log_fusion_error_prob(epsilon_up, epsilon_f, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_error, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, log_lost, sing_trans)
            err_detect += error_detection_prob
            log_lost = 1 - log_succ_this_layer - log_fail_x_this_layer - log_fail_y_this_layer - log_fail_z_this_layer

        log_lost = 1 - log_succ_this_layer
        for _ in range(N-4):
            log_succ_error, err_detect, log_succ_this_layer, log_lost, epsilon_up, epsilon_f, sing_trans = fault_tolerant_fusion_layers_error(
                log_succ_this_layer, log_lost, sing_trans, log_succ_error, err_detect, epsilon_up, epsilon_f)

        p_s = log_succ_this_layer ** n_stations
        fid = (1 - log_succ_error) ** n_stations
        detect_abort = (1 - err_detect) ** n_stations
        print("Outputs: ", fid, log_succ_error, n_stations, p_s, err_detect, detect_abort, n_stations_RGS)
        if fid > 0.99999:
            fid -= 0.00001
        eff_rate = detect_abort * effective_rate(N, n, L, L_att, eta_init, t_CZ, t_meas, t_gen, fid, p_s)
        rate.append(eff_rate)
        RGS_r = RGS_rough_estimates(n_stations_RGS, eps, RGS_succ, t_CZ, t_meas, t_gen)
        RGS_rate.append(RGS_r)

    plt.plot(Ls, rate, color="red", label="Concat. rings")
    plt.plot(Ls, RGS_rate, "k:", label="RGS")
    plt.yscale("log")
    plt.xlabel("L (km)")
    plt.ylabel("R (Hz)")
    plt.title("$\epsilon = $" + str(3 * eps / 2) + ", $\eta_d = $" + str(1-detection_inefficiency))
    plt.legend()
    plt.show()




def compare_tree(eta, t_CZ, t_gen_times, N=3, n=4):
    eps = (1 / 10000) / 3
    detection_inefficiency = 0.05
    eta_init_ring = 0.9
    eta_init_tree = 0.9
    b = [4, 14, 4]
    fusion_succ_ring = succ_ring(N, eta_init_ring)
    print(fusion_succ_ring)
    Ls = np.linspace(500, 1000, 100)
    transmissions = np.linspace(0.9, 1, 100)
    N_qbts = 32
    branch_list = [24, 7]
    RGS_succ = RGS(0.9, N_qbts, branch_list)
    colors = ["red", "black", "purple"]
    for i_x, t_gen in enumerate(t_gen_times):
        rate_tree = []
        rate_ring = []
        rate_RGS = []
        for L in Ls:
            delay_trans = delay_loss(b, t_gen, t_CZ)
            print(delay_trans)
            # succ_tree = p_succ_tree(delay_trans * eta_init_tree, b)
            if eta_init_tree/ delay_trans / (1-detection_inefficiency) > 0.99:
                eta_init_tree = 0.7
            succ_tree = p_succ_tree(eta_init_tree, b)
            n_stations_ring = number_of_stations(L, eta_init_ring/ (1-detection_inefficiency))
            print("eff: ", eta_init_ring/ (1-detection_inefficiency))
            # n_stations_tree = number_of_stations_one_way(L, eta_init_tree + detection_inefficiency)
            n_stations_tree = number_of_stations_one_way(L, eta_init_tree/ delay_trans / (1-detection_inefficiency))
            print(n_stations_tree, eta_init_tree/ delay_trans / (1-detection_inefficiency))
            p_s = fusion_succ_ring ** n_stations_ring
            p_s_tree = succ_tree ** n_stations_tree
            t_tree = p_s_tree / gen_time_tree(b[0], b[1], b[2], t_gen, t_CZ)
            # t_ring = p_s / generation_time_no_meas(N+1, n, t_CZ, t_gen)
            # t_RGS = (RGS_succ ** n_stations_ring) / generation_time_RGS_no_meas(branch_list[0], branch_list[1], t_CZ, 1 / (10 ** 12),
            #                                                                     t_gen, N_qbts)
            # t_RGS = (RGS_succ ** n_stations_ring) / generation_time_RGS_no_meas(branch_list[0], branch_list[1], t_CZ,
            #                                                                     t_CZ,
            #                                                                     t_gen, N_qbts)
            t_RGS = (RGS_succ ** n_stations_ring)/ generation_time_RGS_no_meas(branch_list[0], branch_list[1], t_CZ, t_CZ, t_gen, N_qbts)
            # t_ring = p_s / generation_time(N+1, n, t_CZ, 1 / (10 ** 12), t_gen)
            t_ring = p_s / generation_time(N + 1, n, t_CZ, t_CZ, t_gen)
            rate_ring.append(t_ring)
            rate_tree.append(t_tree)
            rate_RGS.append(t_RGS)
        # for et in transmissions:
        #    log_succ.append(succ_ring(N, et))
        plt.plot(Ls, rate_ring, color=colors[i_x], label="Ring, "+ "$t_{gen}=$"+str(t_gen * (10 ** 9)) + " ns")
        plt.plot(Ls, rate_tree, color=colors[i_x], linestyle="--", label="Tree, " + "$t_{gen}=$"+str(t_gen * (10 ** 9)) + " ns")
        plt.plot(Ls, rate_RGS, color=colors[i_x], linestyle=":",
                 label="RGS, " + "$t_{gen}=$" + str(t_gen * (10 ** 9)) + " ns")
    plt.yscale("log")
    plt.xlabel("L (km)")
    plt.ylabel("R (Hz)")
    plt.title("$t_{CZ} = $" + str(t_CZ))
    plt.legend()
    plt.show()
    log_succ = []
    log_succ_RGS = []
    log_succ_Tree = []
    for et in transmissions:
        # log_succ.append(succ_ring_spin(3, et))
        log_succ.append(log_fusion_spin_gates_modified(3, et))
        log_succ_RGS.append(RGS(et, N_qbts, branch_list))
        log_succ_Tree.append(p_succ_tree(et, b))
    plt.plot(transmissions, log_succ)
    plt.plot(transmissions, log_succ_RGS)
    plt.plot(transmissions, log_succ_Tree)
    plt.show()



def compare_tree_spin_gates(eta, t_CZ, t_gen_times, N=3, n=4):
    eps = (1 / 10000) / 3
    detection_inefficiency = 0.05
    eta_init_ring = 0.92
    eta_init_tree = 0.9
    b = [4, 14, 4]
    fusion_succ_ring = log_fusion_spin_gates_modified(N, eta_init_ring) # succ_ring(N, eta_init_ring)
    print(fusion_succ_ring)
    Ls = np.linspace(500, 1000, 100)
    transmissions = np.linspace(0.9, 1, 100)
    colors = ["red", "black", "purple"]
    for i_x, t_gen in enumerate(t_gen_times):
        rate_tree = []
        rate_ring = []
        for L in Ls:
            delay_trans = delay_loss(b, t_gen, t_CZ)
            print(delay_trans)
            # succ_tree = p_succ_tree(delay_trans * eta_init_tree, b)
            if eta_init_tree/ delay_trans / (1-detection_inefficiency) > 0.99:
                eta_init_tree = 0.7
            succ_tree = p_succ_tree(eta_init_tree, b)
            n_stations_ring = number_of_stations_one_way(L, eta_init_ring/ (1-detection_inefficiency))
            print("eff: ", eta_init_ring/ (1-detection_inefficiency))
            # n_stations_tree = number_of_stations_one_way(L, eta_init_tree + detection_inefficiency)
            n_stations_tree = number_of_stations_one_way(L, eta_init_tree/ delay_trans / (1-detection_inefficiency))
            print(n_stations_tree, eta_init_tree/ delay_trans / (1-detection_inefficiency))
            p_s = fusion_succ_ring ** n_stations_ring
            p_s_tree = succ_tree ** n_stations_tree
            t_tree = p_s_tree / gen_time_tree(b[0], b[1], b[2], t_gen, t_CZ)
            t_ring = p_s / ((67 + 58) * t_CZ + 96 * t_gen + 4 * 100 * t_gen)# generation_time(N + 1, n, t_CZ, t_CZ, t_gen)
            rate_ring.append(t_ring)
            rate_tree.append(t_tree)
        # for et in transmissions:
        #    log_succ.append(succ_ring(N, et))
        plt.plot(Ls, rate_ring, color=colors[i_x], label="Ring, "+ "$t_{gen}=$"+str(t_gen * (10 ** 9)) + " ns")
        plt.plot(Ls, rate_tree, color=colors[i_x], linestyle="--", label="Tree, " + "$t_{gen}=$"+str(t_gen * (10 ** 9)) + " ns")
    plt.yscale("log")
    plt.xlabel("L (km)")
    plt.ylabel("R (Hz)")
    plt.title("$t_{CZ} = $" + str(t_CZ))
    plt.legend()
    plt.show()
    log_succ = []
    log_succ_Tree = []
    for et in transmissions:
        # log_succ.append(succ_ring_spin(3, et))
        log_succ.append(log_fusion_spin_gates_modified(3, et))
        log_succ_Tree.append(p_succ_tree(et, b))
    plt.plot(transmissions, log_succ)
    plt.plot(transmissions, log_succ_Tree)
    plt.show()




def compare_tree_error(eta, t_CZ, t_gen_times, N=3, n=4):
    eps = (1 / 1000) / 3
    detection_inefficiency = 0.05
    eta_init_ring = 0.9
    eta_init_tree = 0.75
    b = [4, 14, 4]
    succ_tree = p_succ_tree(0.98* eta_init_tree, b)
    fusion_succ_ring, err_detect, log_succ_error = succ_error_ring(N, 2 * eps / 3, eta_init_ring)
    print(fusion_succ_ring, err_detect, log_succ_error)
    Ls = np.linspace(500, 1000, 100)
    N_qbts = 32
    branch_list = [24, 7]
    RGS_succ = RGS(0.9, N_qbts, branch_list)
    colors = ["red", "black", "purple"]
    for i_x, t_gen in enumerate(t_gen_times):
        rate_tree = []
        rate_ring = []
        rate_RGS = []
        for L in Ls:
            delay_trans = delay_loss(b, t_gen, t_CZ)
            n_stations_ring = number_of_stations(L, (eta_init_ring) / (1- detection_inefficiency))
            n_stations_tree = number_of_stations_one_way(L, eta_init_tree/(1- detection_inefficiency) / delay_trans)
            p_s = fusion_succ_ring ** n_stations_ring
            fid_ring = (1 - log_succ_error) ** n_stations_ring
            detect_abort = (1 - err_detect) ** n_stations_ring
            print(n_stations_tree)
            p_s_tree = succ_tree ** n_stations_tree
            fid = (1- 3 * eps) ** n_stations_tree
            t_tree = key_siphing(fid) * p_s_tree / gen_time_tree(b[0], b[1], b[2], t_gen, t_CZ)
            # t_ring = detect_abort * key_siphing(fid_ring) * p_s / generation_time_no_meas(N, n, t_CZ, t_gen)
            # t_RGS = (RGS_succ ** n_stations_ring) / generation_time_RGS_no_meas(branch_list[0], branch_list[1], t_CZ, 1 / (10 ** 12),
            #                                                                     t_gen, N_qbts)
            # t_RGS = (RGS_succ ** n_stations_ring) / generation_time_RGS_no_meas(branch_list[0], branch_list[1], t_CZ,
            #                                                                     t_CZ,
            #                                                                     t_gen, N_qbts)
            t_RGS = RGS_rough_estimates(n_stations_ring, 2 * eps / 3, RGS_succ, t_CZ, t_CZ, t_gen) # (RGS_succ ** n_stations_ring)/ generation_time_RGS_no_meas(branch_list[0], branch_list[1], t_CZ, 1, t_gen, N_qbts)
            # t_ring = p_s / generation_time(N+1, n, t_CZ, 1 / (10 ** 12), t_gen)
            t_ring = detect_abort * key_siphing(fid_ring) * p_s / generation_time(N, n, t_CZ, t_CZ, t_gen)
            rate_ring.append(t_ring)
            rate_tree.append(t_tree)
            rate_RGS.append(t_RGS)
        # for et in transmissions:
        #    log_succ.append(succ_ring(N, et))
        plt.plot(Ls, rate_ring, color=colors[i_x], label="Ring, "+ "$t_{gen}=$"+str(t_gen * (10 ** 9)) + " ns")
        plt.plot(Ls, rate_tree, color=colors[i_x], linestyle="--", label="Tree, " + "$t_{gen}=$"+str(t_gen * (10 ** 9)) + " ns")
        plt.plot(Ls, rate_RGS, color=colors[i_x], linestyle=":",
                 label="RGS, " + "$t_{gen}=$" + str(t_gen * (10 ** 9)) + " ns")
    plt.yscale("log")
    plt.xlabel("L (km)")
    plt.ylabel("R (Hz)")
    plt.title("$\epsilon = $"+ str(1 / 1000) + ", $t_{CZ} = $" + str(t_CZ))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # gen_time_tree(4, 14, 4, 1, 1)
    t_CZ = 10 / (10 ** 9)
    # t_CZ = 1 / (10**12)
    # compare_tree_spin_gates(1, t_CZ, [10 / (10 ** 9)], N=3)
    compare_tree(1, t_CZ, [1000 / (10 ** 9)], N=3)
    compare_tree_error(1, t_CZ, [1/ (10 ** 9)], N=5)