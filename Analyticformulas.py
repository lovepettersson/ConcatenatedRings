import numpy as np
import math
import copy


def binom_coeff(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))


def failed_flip(epsilon):
    eps = epsilon / 2
    return 4 * (eps * (1 - 3 * eps) + (eps ** 2))



def log_fusion_prob(p_s, p_f_x, p_f_y, p_f_z, p_l, sing_trans):
    term_one = p_s * ((sing_trans ** 3 + 3 * (1 - sing_trans) * (sing_trans ** 2)) ** 2)
    term_two = p_f_x * p_s * ((sing_trans ** 2 + 2 * (1 - sing_trans) * sing_trans) ** 2)
    term_three = p_l * p_s * (sing_trans ** 4 + p_f_y * (sing_trans ** 2))
    term_four = (p_f_x ** 2) * p_s * (sing_trans ** 2 + p_f_z)
    log_success = term_one + term_two + term_three + term_four
    return log_success



def log_failure(p_s, p_f_x, p_f_y, p_f_z, p_l, sing_trans):
    p_failure_x = ((1 - sing_trans) ** 2) * p_l * p_s * (sing_trans ** 2 + p_f_y) + (p_f_x ** 2) * (p_f_z ** 2)
    p_failure_z = p_s * (sing_trans ** 2) * ((1 - sing_trans) ** 4) + p_f_x * p_s * ((1 - sing_trans) ** 4) + p_f_x * p_l * (sing_trans ** 4)
    return p_failure_x, p_failure_z



def log_transmission(eta):
    eta_up = eta ** 4 + 4 * (eta ** 3) * (1 - eta) + 2 * ((1 - eta) ** 2) * (eta ** 2)
    return eta_up

def error_prop_layer(epsilon, epsilon_f):
    p_s = 1 - epsilon - epsilon_f
    epsilon_up = 4 * epsilon * epsilon * (p_s ** 2) + (2 * (1 - (1 - epsilon_f) ** 2)) * 2 * epsilon * p_s
    epsilon_f_up = 4 * epsilon * (p_s ** 3) + 4 * (epsilon ** 3) * p_s + 4 * epsilon_f * epsilon_f * ((1 - epsilon_f)**2) + 4 * epsilon_f * epsilon_f * epsilon_f * (1- epsilon_f) + epsilon_f ** 4
    return epsilon_up, epsilon_f_up





def error_prop_layer_fusion(epsilon_x, epsilon_f_x, epsilon_z, epsilon_f_z):
    # Use this for propagating fusion errors fault tolerantly, for logical X.
    p_s_x = 1 - epsilon_x - epsilon_f_x
    p_s_z = 1 - epsilon_z - epsilon_f_z
    eps_up = 4 * epsilon_x * epsilon_z * p_s_x * p_s_z + (1 - (1 - epsilon_f_x) ** 2) * 2 * epsilon_z * p_s_z + (1 - (1 - epsilon_f_z) ** 2) * 2 * epsilon_x * p_s_x
    eps_f_up = 2 * epsilon_x * p_s_x * p_s_z * p_s_z + 2 * epsilon_z * p_s_x * p_s_x * p_s_z + 2 * (epsilon_x ** 2) * epsilon_z * p_s_z + 2 * (epsilon_z ** 2) * epsilon_x * p_s_x + \
        4 * epsilon_f_x * epsilon_f_z * (1 - epsilon_f_x) * (1 - epsilon_f_z) + 2 * epsilon_f_x * epsilon_f_x * epsilon_f_z * (1 - epsilon_f_z) + 2 * epsilon_f_z * epsilon_f_z * epsilon_f_x * (1 - epsilon_f_x) + \
        epsilon_f_x * epsilon_f_x * epsilon_f_z * epsilon_f_z
    return eps_up, eps_f_up


def error_prop_layer_fusion_ZY(epsilon_x, epsilon_f_x, epsilon_z, epsilon_f_z):
    # For logical Z and Y we are matching different paritites, i.e., for logical Z it is X_1 x Z_2 or Z_3 x X_4
    p_s_x = 1 - epsilon_x - epsilon_f_x
    p_s_z = 1 - epsilon_z - epsilon_f_z
    eps_up = 2 * epsilon_x * epsilon_x * p_s_z * p_s_z + 2 * epsilon_z * epsilon_z * p_s_x * p_s_x + 2 * (1 - (1 - epsilon_f_x) * (1 - epsilon_f_z)) * (epsilon_z * p_s_x + epsilon_x * p_s_z)
    eps_f_up = 2 * epsilon_x * p_s_x * p_s_z * p_s_z + 2 * epsilon_z * p_s_x * p_s_x * p_s_z + 2 * (epsilon_x * epsilon_z) * epsilon_x * p_s_z + 2 * (epsilon_z * epsilon_x) * epsilon_z * p_s_x + \
        2 * epsilon_f_x * epsilon_f_x * (1 - epsilon_f_z) * (1 - epsilon_f_z) + 2 * epsilon_f_z * epsilon_f_z * (1 - epsilon_f_x) * (1 - epsilon_f_x) + 2 * epsilon_f_x * epsilon_f_x * epsilon_f_z * (1 - epsilon_f_z) + 2 * epsilon_f_z * epsilon_f_z * epsilon_f_x * (1 - epsilon_f_x) + \
        epsilon_f_x * epsilon_f_x * epsilon_f_z * epsilon_f_z
    return eps_up, eps_f_up


def intial_eps_f(epsilon):
    return 4 * epsilon * ((1-epsilon) ** 3) + 4 * (epsilon ** 3) * (1 - epsilon)



def error_prop_layer_with_loss(epsilon, epsilon_f, eta):
    eta_four = eta ** 4
    p_s = 1 - epsilon - epsilon_f
    eta_up = log_transmission(eta)
    epsilon_up = (eta_four * (4 * epsilon * epsilon * (p_s ** 2) + (2 * (2 * (p_s+epsilon) * epsilon_f + epsilon_f * epsilon_f)) * 2 * epsilon * p_s) + (eta_up - eta_four) * (2 * epsilon * (p_s))) / eta_up
    epsilon_f_up = (eta_four * (4 * epsilon * (p_s ** 3) + 4 * (epsilon ** 3) * p_s + 4 * epsilon_f * epsilon_f * ((1 - epsilon_f)**2) + 4 * epsilon_f * epsilon_f * epsilon_f * (1- epsilon_f) + epsilon_f ** 4) + (
                eta_up - eta_four) * (2 * epsilon_f * (1 - epsilon_f) + epsilon_f ** 2)) / eta_up
    return epsilon_up, epsilon_f_up, eta_up


def intial_eps_f_with_loss(epsilon, eta):
    return (eta ** 4) * (4 * epsilon * ((1-epsilon) ** 3) + 4 * (epsilon ** 3) * (1 - epsilon))




if __name__ == '__main__':
    from Tree_analytics import *


