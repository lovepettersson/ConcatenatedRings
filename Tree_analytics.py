import numpy as np
import math


def Z_ind_tree(t, k, branch_list):
    max_layer_depth = len(branch_list)
    #     print(branch_list, k)
    if k < max_layer_depth:
        this_b = branch_list[k]
    else:
        return 0
    if k == (max_layer_depth - 1):
        b_next = 0
    else:
        b_next = branch_list[k + 1]
    return 1 - ((1 - (t * ((t + (1 - t) * Z_ind_tree(t, k + 2, branch_list)) ** b_next))) ** this_b)


# Z_tree(t, k, branch_list) is the probability of Z measurement (direct or indirect) on a qubit at layer k
# of a tree-graph described by branchings branch_list, in case the trasmittivity is t.
def Z_tree(t, k, branch_list):
    return t + (1 - t) * Z_ind_tree(t, k, branch_list)

# p_succ_tree(t, branch_list) is the total probability for decoding a tree-graph
#
def p_succ_tree(t, branch_list):
    if not branch_list:
        return 0
    elif len(branch_list) == 1:
        b0 = branch_list[0]
        p = t ** b0
    else:
        z_inds = [Z_ind_tree(t, k, branch_list) for k in (1, 2)]
        b0 = branch_list[0]
        b1 = branch_list[1]
        p = ((t + (1 - t) * z_inds[0]) ** b0 - ((1 - t) * z_inds[0]) ** b0) * (t + (1 - t) * z_inds[1]) ** b1
    return p


def p_X_succ_tree(t, branch_list):
    if not branch_list:
        return 0
    elif len(branch_list) == 1:
        b0 = branch_list[0]
        p = t ** b0
    else:
        z_inds = [Z_ind_tree(t, k, branch_list) for k in (1, 2)]
        b0 = branch_list[0]
        p = (t + (1 - t) * z_inds[0]) ** b0
    return p


def tree_q_num(b_list):
    tot = 1
    num_last_layer = 1
    for b in b_list:
        num_last_layer *= b
        tot += num_last_layer
    return tot - 1

def RGS(eta, N, branch_list):
    Z = Z_tree(eta, 0, branch_list)
    X = p_X_succ_tree(eta, branch_list)
    fuse_succ = (1 - (1-(eta**2)/2)**(N/2)) * ((X**2) * (Z ** (N - 2)))
    # fuse_succ = (1 - ((1 - eta) ** 2 / 2) ** (N / 2)) * ((X ** 2) * (Z ** (N - 2)))
    return fuse_succ

def binom_coeff(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))




def error_trans_one_log_Z_op(eps, b1, eta):
    error = 0
    for i in range(1, b1 + 2, 2):
        error += binom_coeff(b1 + 1, i) * (eps ** i) * ((1 - eps) ** (b1 + 1 - i))
    return error, eta ** (b1 + 1)

def majority_vote(eps, mk):
    error = 0
    if mk == 1:
        error = eps
    elif mk % 2 == 1:
        for m in range(int(mk / 2), mk + 1):
            error += binom_coeff(mk, m) * (eps ** m) * ((1 - eps) ** (mk - m))
    else:
        for m in range(int(mk / 2), mk):
            error += binom_coeff(mk- 1, m) * (eps ** m) * ((1 - eps) ** (mk - 1 - m))
    return error

def error_log_Z(eps, b0, b1, trans):
    tot_error = 0
    tot_trans = 0
    eps, eta = error_trans_one_log_Z_op(eps, b1, trans)
    for j in range(1, b0 + 1):
        prob_meas = binom_coeff(b0, j) * (eta ** j) * ((1 - eta) ** (b0 - j))
        tot_trans += prob_meas
        err = majority_vote(eps, j)
        tot_error += err * prob_meas
    return tot_trans, tot_error / tot_trans

def error_single_branch(eps, b1, eta):
    tot_trans = 1 - (1 - eta) ** (b1 + 1)
    tot_error = 0
    direct = eta * ((1 - eta) ** b1) * eps
    tot_error += direct
    for j in range(1, b1 + 1):
        prob_meas = binom_coeff(b1, j) * (eta ** j) * ((1 - eta) ** (b1 - j))
        err = majority_vote(eps, j)
        tot_error += err * prob_meas
    return tot_trans, tot_error / tot_trans


def error_log_X(eps, b0, b1, trans):
    eta, err = error_single_branch(eps, b1, trans)
    error = 0
    log_X_trans = eta ** (b0)
    for i in range(1, b0, 2):
        error += binom_coeff(b0, i) * (err ** i) * ((1 - err) ** (b0 - i))
    return log_X_trans, error / log_X_trans


def fidelity_RGS(N_stations, N_nodes, eps, eta, b0, b1):
    trans_Z, error_Z = error_log_Z(eps, b0, b1, eta)
    trans_X, error_X = error_log_X(eps, b0, b1, eta)
    E_x_z = (1 / 4) * (1 - ((1 - 2 * eps) ** ((2 * N_stations + 1))) * ((1 - 2 * error_X) ** (2 * N_stations)))
    E_y = (1 / 4) * (1 + ((1 - 2 * eps) ** ((2 * N_stations + 1))) * ((1 - 2 * error_X) ** (2 * N_stations)) - \
                     2 * ((1 - 2 * eps) ** ((2 * N_stations + 1))) * ((1 - 2 * error_X) ** (2 * N_stations)) * (1 - 2 * error_Z) ** (N_stations * (N_nodes - 2)))

    fid = 1 - 2 * E_x_z - E_y
    return fid



##########################################################
############ GENERALIZING TO DEPTH D TREES ###############
##########################################################


def indirect_Z_error(eps_X, eps_Z, eta_Z, eta_X, b_n, b_n_plus_one):
    tot_error = 0
    error = 0
    tot_trans = 0
    eta = eta_X * (eta_Z ** (b_n_plus_one))
    for i in range(1, b_n_plus_one + 1, 2):
        error += binom_coeff(b1 + 1, i) * (eps_Z ** i) * ((1 - eps_Z) ** (b1 + 1 - i))
    eps = error * (1 - eps_X) + eps_X * (1 - error)
    for j in range(1, b_n + 1):
        prob_meas = binom_coeff(b0, j) * (eta ** j) * ((1 - eta) ** (b0 - j))
        tot_trans += prob_meas
        err = majority_vote(eps, j)
        tot_error += err * prob_meas
    return tot_trans, tot_error / tot_trans


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    eps = 0.000005
    eta = 0.88
    trans_Z, error_Z = error_log_Z(eps, 20, 7, 0.9)
    print(error_single_branch(eps, 7, 0.9))
    print(error_log_X(eps, 24, 7, 0.9))
    print(fidelity_RGS(50, 32, eps, 0.9, 24, 7))

    N_stations = 50
    N_nodes = 32
    b0 = 24
    b1 = 7
    errors = np.linspace(0.0000005, 0.0005, 100)
    RGS_error = []
    approx_error = [(1 - 4 * (epsilon * (1 - epsilon) + (epsilon ** 2))) ** (N_stations + 1) for epsilon in errors]
    for error in errors:
        err_RGS = fidelity_RGS(N_stations, N_nodes, error, eta, b0, b1)
        RGS_error.append(err_RGS)
    plt.plot(errors, RGS_error, color="red")
    plt.plot(errors, approx_error, color="black")
    plt.show()