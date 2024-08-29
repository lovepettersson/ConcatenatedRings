import numpy as np
import matplotlib.pyplot as plt
import math
import copy



def binary_entropy_function(x):
    return -x * math.log2(x) - (1 - x) * math.log2(1-x)

def key_siphing(fid):
    return fid - binary_entropy_function(1 - fid) - fid * binary_entropy_function((3 * fid - 1) / (2 * fid))


def binom_coeff(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))


def parity_flip_old(epsilon):
    # This is the probability that one parity is flipped!
    return 4 * (epsilon * (1 - epsilon) + (epsilon ** 2))

def failed_flip_old(epsilon):
    return 2 * epsilon * (1 - epsilon)

def parity_flip(epsilon):
    eps = epsilon / 2
    p_flip = 4 * (eps * (1 - 3 * eps) + (eps ** 2))
    fusion_error = 2 * p_flip * (1 - p_flip) + p_flip ** 2
    return fusion_error

def failed_flip(epsilon):
    eps = epsilon / 2
    return 4 * (eps * (1 - 3 * eps) + (eps ** 2))
    # return 2 * epsilon * (1 - epsilon)


def log_fusion_prob(p_s, p_f_x, p_f_y, p_f_z, p_l, sing_trans):
    term_one = p_s * ((sing_trans ** 3 + 3 * (1 - sing_trans) * (sing_trans ** 2)) ** 2)
    term_two = p_f_x * p_s * ((sing_trans ** 2 + 2 * (1 - sing_trans) * sing_trans) ** 2)
    term_three = p_l * p_s * (sing_trans ** 4 + p_f_y * (sing_trans ** 2))
    term_four = (p_f_x ** 2) * p_s * (sing_trans ** 2 + p_f_z)
    log_success = term_one + term_two + term_three + term_four
    return log_success

def log_failure(p_s, p_f_x, p_f_y, p_f_z, p_l, sing_trans):
    p_failure_x = ((1-sing_trans) ** 2) * p_l * p_s * (sing_trans ** 2 + p_f_y) + p_f_x ** 4
    p_failure_z = p_s * (sing_trans ** 2) * ((1-sing_trans) ** 4) + p_f_x * p_s * ((1 - sing_trans) ** 4) + p_f_x * p_l * (sing_trans ** 4)
    return p_failure_x, p_failure_z

def log_fusion_error_prob_old(eps, eps_f, eps_p_fail_x, eps_p_fail_y, eps_p_fail_z, eps_p_succ, log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans):

    # Missing error in failure mode !!! But neglect it for now as it is generally very low
    fail_and_succ_error_x = eps_p_succ * (1 - eps_p_fail_x) + eps_p_fail_x * (1 - eps_p_succ)
    fail_and_succ_no_error_x = (1 - eps_p_fail_x) * (1 - eps_p_succ)

    fail_and_succ_error_y = eps_p_succ * (1 - eps_p_fail_y) + eps_p_fail_y * (1 - eps_p_succ)
    fail_and_succ_no_error_y = (1 - eps_p_fail_y) * (1 - eps_p_succ)

    traj_one_error = (eps_p_succ * ((1 - eps) ** 5) + binom_coeff(5, 2) * eps_p_succ * ((1 - eps) ** 3) * (eps ** 2) + binom_coeff(5, 4) * eps_p_succ * (1 - eps) * (eps ** 4) + \
                      binom_coeff(5, 1) * ((1 - eps) ** 4) * eps * (1-eps_p_succ) + binom_coeff(5, 3) * ((1 - eps) ** 2) * (eps ** 3) * (1-eps_p_succ)  + (eps ** 5) * (1-eps_p_succ)) * log_succ * (sing_trans ** 6)
    # traj_one_error = (eps_p_succ * ((1 - eps) ** 6) + binom_coeff(6, 2) * eps_p_succ * ((1 - eps) ** 4) * (eps ** 2) + binom_coeff(6, 4) * eps_p_succ * ((1 - eps) ** 2) * (eps ** 4) + eps_p_succ * (eps ** 6) + \
                      # binom_coeff(6, 1) * ((1 - eps) ** 5) * eps * (1-eps_p_succ) + binom_coeff(6, 3) * ((1 - eps) ** 3) * (eps ** 3) * (1-eps_p_succ)  + binom_coeff(6, 5) * (1 - eps) * (eps ** 5) * (1-eps_p_succ)) * log_succ * (sing_trans ** 6)
    traj_one_detection = ((binom_coeff(5, 1) * (1 - eps_f) ** 4) * eps_f + binom_coeff(5, 2) * (1 - eps_f) ** 3 * (
                eps_f ** 2) + binom_coeff(5, 3) * (1 - eps_f) ** 2 * (eps_f ** 3) + binom_coeff(5, 4) * (
                                      1 - eps_f) * (eps_f ** 4) + \
                           eps_f ** 5) * log_succ * (sing_trans ** 6)
    # traj_one_detection = ((binom_coeff(6, 1) * (1 - eps_f) ** 5) * eps_f + binom_coeff(6, 2) * (1 - eps_f) ** 4 * (eps_f ** 2) + binom_coeff(6, 3) * (1 - eps_f) ** 3 * (eps_f ** 3) + binom_coeff(6, 4) * (1 - eps_f) ** 2 * (eps_f ** 4) + \
                         # binom_coeff(6, 5) * (1 - eps_f) * (eps_f ** 5) + eps_f ** 6) * log_succ * (sing_trans ** 6)

    traj_two_error = 4 * (eps_p_succ * ((1 - eps) ** 5) + binom_coeff(5, 2) * eps_p_succ * ((1 - eps) ** 3) * (eps ** 2) + binom_coeff(5, 4) * eps_p_succ * (1 - eps) * (eps ** 4) + \
                      binom_coeff(5, 1) * ((1 - eps) ** 4) * eps * (1-eps_p_succ) + binom_coeff(5, 3) * ((1 - eps) ** 2) * (eps ** 3) * (1-eps_p_succ)  + (eps ** 5) * (1-eps_p_succ)) * log_succ * (sing_trans ** 5) * (1 - sing_trans)

    traj_two_detection = 4 * ((binom_coeff(5, 1) * (1 - eps_f) ** 4) * eps_f + binom_coeff(5, 2) * (1 - eps_f) ** 3 * (
                eps_f ** 2) + binom_coeff(5, 3) * (1 - eps_f) ** 2 * (eps_f ** 3) + binom_coeff(5, 4) * (
                                      1 - eps_f) * (eps_f ** 4) + \
                           eps_f ** 5) * log_succ * (sing_trans ** 5) * (1 - sing_trans)

    traj_three_error = (eps_p_succ * ((1 - eps) ** 4) + binom_coeff(4, 2) * eps_p_succ * ((1 - eps) ** 2) * (eps ** 2) +  eps_p_succ  * (eps ** 4) + \
                      binom_coeff(4, 1) * ((1 - eps) ** 3) * eps * (1-eps_p_succ) + binom_coeff(4, 3) * (1 - eps) * (eps ** 3) * (1-eps_p_succ)) * log_succ * ((sing_trans ** 3 + 3 * (1 - sing_trans) * sing_trans) ** 2) # * log_succ * (sing_trans ** 4) * ((1 - sing_trans) * 2)

    traj_three_detection = ((binom_coeff(4, 1) * (1 - eps_f) ** 3) * eps_f + binom_coeff(4, 2) * (1 - eps_f) ** 2 * (
            eps_f ** 2) + binom_coeff(4, 3) * (1 - eps_f) * (eps_f ** 3) + \
                              eps_f ** 4) * log_succ * ((sing_trans ** 3 + 3 * (1 - sing_trans) * sing_trans) ** 2)# * log_succ * (sing_trans ** 4) * ((1 - sing_trans) * 2)

    ## p_fail_x * p_s * eta ** 4 ##
    traj_four_error = (fail_and_succ_error_x * ((1 - eps) ** 4) + binom_coeff(4, 2) * fail_and_succ_error_x  * ((1 - eps) ** 2) * (eps ** 2) + fail_and_succ_error_x * (1 - eps) * (eps ** 4) + \
                      binom_coeff(4, 1) * ((1 - eps) ** 3) * eps * fail_and_succ_no_error_x + binom_coeff(4, 3) * (1 - eps) * (eps ** 3) * fail_and_succ_no_error_x) * log_succ * log_p_fail_x * (sing_trans ** 4)

    traj_four_detection = (
                (binom_coeff(4, 1) * (1 - eps_f) ** 3) * eps_f + binom_coeff(4, 2) * (1 - eps_f) ** 2 * (
                eps_f ** 2) + binom_coeff(4, 3) * (1 - eps_f) * (eps_f ** 3) + \
                eps_f ** 4) * log_succ * log_p_fail_x * (sing_trans ** 4)

    ## p_fail_x * p_s * eta ** 3 * (1 - eta) ##
    traj_five_error = 4 * (fail_and_succ_error_x * ((1 - eps) ** 3) + binom_coeff(3, 2) * fail_and_succ_error_x * (1 - eps) * (
                eps ** 2) + \
                       binom_coeff(3, 1) * ((1 - eps) ** 2) * eps * fail_and_succ_no_error_x + (
                                   eps ** 3) * fail_and_succ_no_error_x) * log_succ * log_p_fail_x * (sing_trans ** 3) * (1 - sing_trans)

    traj_five_detection = ((binom_coeff(3, 1) * (1 - eps_f) ** 2) * eps_f + binom_coeff(3, 2) * (
                                      1 - eps_f) * (
                                          eps_f ** 2) + (eps_f ** 3)) * log_succ * log_p_fail_x * (sing_trans ** 3) * (1 - sing_trans)

    ## p_fail_x * p_s * eta ** 2 * (1 - eta) * (1 - eta) ##

    traj_six_error = 4 * (fail_and_succ_error_x * ((1 - eps) ** 2) + fail_and_succ_error_x * (
            eps ** 2) + binom_coeff(2, 1) * (1 - eps) * eps * fail_and_succ_no_error_x) * log_succ * log_p_fail_x * (sing_trans ** 2) * ((
                                  1 - sing_trans) ** 2)

    traj_six_detection = ((binom_coeff(2, 1) * (1 - eps_f)) * eps_f + (
                                   eps_f ** 2)) * log_succ * log_p_fail_x * (sing_trans ** 2) * ((
                                  1 - sing_trans) ** 2)

    ## p_l * p_s * eta ** 4  ##

    traj_seven_error = (fail_and_succ_error_x * ((1 - eps) ** 4) + binom_coeff(4, 2) * fail_and_succ_error_x * ((1 - eps) ** 2) * (eps ** 2) + fail_and_succ_error_x * (eps ** 4) + \
                      binom_coeff(4, 1) * ((1 - eps) ** 3) * eps * fail_and_succ_no_error_x + binom_coeff(4, 3) * (eps ** 3) * fail_and_succ_no_error_x) * log_lost * log_succ * (sing_trans ** 4)

    traj_seven_detection = (
                                  (binom_coeff(4, 1) * (1 - eps_f) ** 3) * eps_f + binom_coeff(4, 2) * (
                                      1 - eps_f) ** 2 * (
                                          eps_f ** 2) + binom_coeff(4, 3) * (1 - eps_f) * (eps_f ** 3) + \
                                  eps_f ** 4) * log_lost * log_succ * (sing_trans ** 4)

    ## p_l * p_s * p_fail_y * eta ** 2

    traj_eigth_error = (fail_and_succ_error_y * ((1 - eps) ** 2) + fail_and_succ_error_y * (eps ** 2) + \
                       binom_coeff(2, 1) * ((1 - eps)) * eps * fail_and_succ_no_error_y) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y

    traj_eigth_detection = ((binom_coeff(2, 1) * (1 - eps_f)) * eps_f + (
            eps_f ** 2)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y

    ## p_fail_x * p_fail_x * p_s * eta ** 2

    two_fail_and_succ_error = eps_p_succ * ((1- eps_p_fail_x) ** 2) + 2 * eps_p_fail_x * (1 - eps_p_succ) * (1 - eps_p_fail_x) + eps_p_succ * eps_p_fail_x * eps_p_fail_x
    two_fail_and_succ_no_error = 1 - two_fail_and_succ_error

    traj_nine_error = (two_fail_and_succ_error * ((1 - eps) ** 2) + two_fail_and_succ_error * (eps ** 2) + \
                        binom_coeff(2, 1) * ((1 - eps)) * eps * two_fail_and_succ_no_error) * log_succ * (
                                   sing_trans ** 2) * log_p_fail_x * log_p_fail_x
    traj_nine_detection = ((binom_coeff(2, 1) * (1 - eps_f)) * eps_f + (
            eps_f ** 2)) * log_succ * (
                                   sing_trans ** 2) * log_p_fail_x * log_p_fail_x

    ## p_fail_x * p_fail_x * p_s * p_fail_z
    three_fail_and_succ_error = eps_p_succ * ((1 - eps_p_fail_x) ** 2) * (1 - eps_p_fail_z) + 2 * eps_p_fail_x * (1 - eps_p_succ) * (
                1 - eps_p_fail_x) * (1 - eps_p_fail_z) + eps_p_succ * eps_p_fail_x * eps_p_fail_x * (1 - eps_p_fail_z) + 2 * eps_p_succ * eps_p_fail_z * eps_p_fail_x * (1 - eps_p_fail_x) + (1 - eps_p_succ) * eps_p_fail_z * (eps_p_fail_x ** 2)
    traj_ten_error = three_fail_and_succ_error * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ

    log_succ_this_layer = log_fusion_prob(log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
    epsilon_up, epsilon_f_up, eta_up =  error_prop_layer_with_loss(eps, eps_f, eta) # log_transmission(sing_trans)
    log_fail_x_this_layer, log_fail_z_this_layer = log_failure(log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
    # log_fail_x_this_layer, log_fail_z_this_layer = 0, 0
    log_fail_y_this_layer = 0

    log_succ_error = (traj_three_error + traj_four_error + traj_five_error + traj_six_error + traj_seven_error + traj_eigth_error + traj_nine_error + traj_ten_error) / log_succ_this_layer
    error_detection_prob = (traj_three_detection + traj_four_detection + traj_five_detection + traj_six_detection + traj_seven_detection + traj_eigth_detection + traj_nine_detection)
    # log_succ_error = (eps_p_succ * ((1 - eps) ** 6) + binom_coeff(6, 2) * eps_p_succ * ((1 - eps) ** 4) * (eps ** 2) + binom_coeff(6, 4) * eps_p_succ * ((1 - eps) ** 2) * (eps ** 4) + eps_p_succ * (eps ** 6) + \
    #                   binom_coeff(6, 1) * ((1 - eps) ** 5) * eps * (1-eps_p_succ) + binom_coeff(6, 3) * ((1 - eps) ** 3) * (eps ** 3) * (1-eps_p_succ)  + binom_coeff(6, 5) * (1 - eps) * (eps ** 5) * (1-eps_p_succ))
    # error_detection_prob = ((binom_coeff(6, 1) * (1 - eps_f) ** 5) * eps_f + binom_coeff(6, 2) * (1 - eps_f) ** 4 * (eps_f ** 2) + binom_coeff(6, 3) * (1 - eps_f) ** 3 * (eps_f ** 3) + binom_coeff(6, 4) * (1 - eps_f) ** 2 * (eps_f ** 4) + \
    #                      binom_coeff(6, 5) * (1 - eps_f) * (eps_f ** 5) + eps_f ** 6)
    # Fix errors for failure rates
    log_p_fail_x_this_layer = 0
    log_p_fail_y_this_layer = 0
    log_p_fail_z_this_layer = 0
    print("here:", traj_three_error, traj_four_error, traj_five_error, traj_six_error, traj_seven_error, traj_eigth_error, traj_nine_error, traj_ten_error)
    return log_succ_error, error_detection_prob, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f_up, eta_up







def log_fusion_error_prob(eps, eps_f, eps_p_fail_x, eps_p_fail_y, eps_p_fail_z, eps_p_succ, log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans):

    p_s = 1 - eps - eps_f
    # Missing error in failure mode !!! But neglect it for now as it is generally very low
    fail_and_succ_error_x = eps_p_succ * (1 - eps_p_fail_x) + eps_p_fail_x * (1 - eps_p_succ)
    fail_and_succ_no_error_x = (1 - eps_p_fail_x) * (1 - eps_p_succ)

    fail_and_succ_error_y = eps_p_succ * (1 - eps_p_fail_y) + eps_p_fail_y * (1 - eps_p_succ)
    fail_and_succ_no_error_y = (1 - eps_p_fail_y) * (1 - eps_p_succ)


    traj_one_error = (eps_p_succ * (p_s ** 4) + binom_coeff(4, 2) * eps_p_succ * (p_s ** 2) * (eps ** 2)  + \
                      binom_coeff(4, 1) * (p_s ** 4) * eps * (1-eps_p_succ) + binom_coeff(4, 3) * (p_s) * (eps ** 3) * (1-eps_p_succ)) * log_succ * (sing_trans ** 6)
    # traj_one_error = (eps_p_succ * ((p_s) ** 6) + binom_coeff(6, 2) * eps_p_succ * ((p_s) ** 4) * (eps ** 2) + binom_coeff(6, 4) * eps_p_succ * ((p_s) ** 2) * (eps ** 4) + eps_p_succ * (eps ** 6) + \
    #                   binom_coeff(6, 1) * ((p_s) ** 5) * eps * (1-eps_p_succ) + binom_coeff(6, 3) * ((p_s) ** 3) * (eps ** 3) * (1-eps_p_succ)  + binom_coeff(6, 5) * (p_s) * (eps ** 5) * (1-eps_p_succ)) * log_succ * (sing_trans ** 6)
    # traj_one_detection = ((binom_coeff(5, 1) * (1 - eps_f) ** 4) * eps_f + binom_coeff(5, 2) * (1 - eps_f) ** 3 * (
    #             eps_f ** 2) + binom_coeff(5, 3) * (1 - eps_f) ** 2 * (eps_f ** 3) + binom_coeff(5, 4) * (
    #                                   1 - eps_f) * (eps_f ** 4) + \
    #                        eps_f ** 5) * log_succ * (sing_trans ** 6)

    #### Since you only care about two single qubit measurements, you can correct one detection event ####
    traj_one_detection = (binom_coeff(6, 2) * (1 - eps_f) ** 4 * (eps_f ** 2) + binom_coeff(6, 3) * (1 - eps_f) ** 3 * (eps_f ** 3) + binom_coeff(6, 4) * (1 - eps_f) ** 2 * (eps_f ** 4) + \
                         binom_coeff(6, 5) * (1 - eps_f) * (eps_f ** 5) + eps_f ** 6) * log_succ * (sing_trans ** 6)

    traj_two_error = 6 * (eps_p_succ * (p_s ** 5) + binom_coeff(5, 2) * eps_p_succ * (p_s ** 3) * (eps ** 2) + \
                      binom_coeff(5, 1) * (p_s ** 4) * eps * (1-eps_p_succ) + binom_coeff(5, 3) * (p_s ** 2) * (eps ** 3) * (1-eps_p_succ)) * log_succ * (sing_trans ** 5) * (1 - sing_trans)

    # traj_two_detection = 6 * ((binom_coeff(5, 1) * (1 - eps_f) ** 4) * eps_f + binom_coeff(5, 2) * (1 - eps_f) ** 3 * (
    #             eps_f ** 2) + binom_coeff(5, 3) * (1 - eps_f) ** 2 * (eps_f ** 3) + binom_coeff(5, 4) * (
    #                                   1 - eps_f) * (eps_f ** 4) + \
    #                        eps_f ** 5) * log_succ * (sing_trans ** 5) * (1 - sing_trans)
    #### Only the single qubit detection events from one code matters
    # traj_two_detection = 6 * ((binom_coeff(2, 1) * (1 - eps_f)) * eps_f + (
    #                     eps_f ** 2) + binom_coeff(3, 2) * (1 - eps_f) * (eps_f ** 2) +  (eps_f ** 4)) * log_succ * (sing_trans ** 5) * (1 - sing_trans)
    traj_two_detection = 6 * ((binom_coeff(4, 1) * (1 - eps_f) ** 3) * eps_f + binom_coeff(4, 2) * (1 - eps_f) ** 2 * (
                    eps_f ** 2) + binom_coeff(4, 3) * (1 - eps_f) * (eps_f ** 3) +  (eps_f ** 4)) * log_succ * (sing_trans ** 5) * (1 - sing_trans)

    traj_three_error = (eps_p_succ * ((p_s) ** 4) + binom_coeff(4, 2) * eps_p_succ * ((p_s) ** 2) * (eps ** 2) + \
                      binom_coeff(4, 1) * ((p_s) ** 3) * eps * (1-eps_p_succ) + binom_coeff(4, 3) * (p_s) * (eps ** 3) * (1-eps_p_succ)) * log_succ * ((3 * (1 - sing_trans) * sing_trans) ** 2) # * log_succ * (sing_trans ** 4) * ((1 - sing_trans) * 2)

    traj_three_detection = ((binom_coeff(4, 1) * (1 - eps_f) ** 3) * eps_f + binom_coeff(4, 2) * (1 - eps_f) ** 2 * (
            eps_f ** 2) + binom_coeff(4, 3) * (1 - eps_f) * (eps_f ** 3) + \
                              eps_f ** 4) * log_succ * ((3 * (1 - sing_trans) * sing_trans) ** 2)# * log_succ * (sing_trans ** 4) * ((1 - sing_trans) * 2)

    ## p_fail_x * p_s * eta ** 4 ##
    traj_four_error = (fail_and_succ_error_x * ((p_s) ** 4) + binom_coeff(4, 2) * fail_and_succ_error_x  * ((p_s) ** 2) * (eps ** 2) + eps_p_fail_x * eps_p_succ * binom_coeff(4, 1) * eps * (p_s ** 3) + \
                      binom_coeff(4, 1) * ((p_s) ** 3) * eps * fail_and_succ_no_error_x + binom_coeff(4, 3) * (p_s) * (eps ** 3) * fail_and_succ_no_error_x) * log_succ * log_p_fail_x * (sing_trans ** 4)

    traj_four_detection = (
                (binom_coeff(4, 1) * (1 - eps_f) ** 3) * eps_f + binom_coeff(4, 2) * (1 - eps_f) ** 2 * (
                eps_f ** 2) + binom_coeff(4, 3) * (1 - eps_f) * (eps_f ** 3) + \
                eps_f ** 4) * log_succ * log_p_fail_x * (sing_trans ** 4)

    ## p_fail_x * p_s * eta ** 3 * (1 - eta) ##
    traj_five_error = 4 * (fail_and_succ_error_x * ((p_s) ** 3) + binom_coeff(3, 2) * fail_and_succ_error_x * (p_s) * (
                eps ** 2) + eps_p_fail_x * eps_p_succ * binom_coeff(3, 1) * eps * (p_s ** 2) + \
                       binom_coeff(3, 1) * ((p_s) ** 2) * eps * fail_and_succ_no_error_x + (
                                   eps ** 3) * fail_and_succ_no_error_x) * log_succ * log_p_fail_x * (sing_trans ** 3) * (1 - sing_trans)

    traj_five_detection = ((binom_coeff(3, 1) * (1 - eps_f) ** 2) * eps_f + binom_coeff(3, 2) * (
                                      1 - eps_f) * (
                                          eps_f ** 2) + (eps_f ** 3)) * log_succ * log_p_fail_x * (sing_trans ** 3) * (1 - sing_trans)

    ## p_fail_x * p_s * eta ** 2 * (1 - eta) * (1 - eta) ##

    traj_six_error = 4 * (fail_and_succ_error_x * ((p_s) ** 2) + fail_and_succ_error_x * (
            eps ** 2) + eps_p_fail_x * eps_p_succ * 2 * eps * p_s + binom_coeff(2, 1) * (p_s) * eps * fail_and_succ_no_error_x) * log_succ * log_p_fail_x * (sing_trans ** 2) * ((
                                  1 - sing_trans) ** 2)

    traj_six_detection = ((binom_coeff(2, 1) * (1 - eps_f)) * eps_f + (
                                   eps_f ** 2)) * log_succ * log_p_fail_x * (sing_trans ** 2) * ((
                                  1 - sing_trans) ** 2)

    ## p_l * p_s * eta ** 4  ##

    traj_seven_error = (eps_p_succ * ((p_s) ** 4) + binom_coeff(4, 2) * eps_p_succ * ((p_s) ** 2) * (eps ** 2) + \
                      binom_coeff(4, 1) * ((p_s) ** 3) * eps * (1-eps_p_succ) + binom_coeff(4, 3) * (eps ** 3) * (1 - eps_p_succ)) * log_lost * log_succ * (sing_trans ** 4)

    traj_seven_detection = (
                                  (binom_coeff(4, 1) * (1 - eps_f) ** 3) * eps_f + binom_coeff(4, 2) * (
                                      1 - eps_f) ** 2 * (
                                          eps_f ** 2) + binom_coeff(4, 3) * (1 - eps_f) * (eps_f ** 3) + \
                                  eps_f ** 4) * log_lost * log_succ * (sing_trans ** 4)

    ## p_l * p_s * p_fail_y * eta ** 2

    traj_eigth_error = (fail_and_succ_error_y * ((p_s) ** 2) + fail_and_succ_error_y * (eps ** 2) + eps_p_fail_y * eps_p_succ * 2 * eps * p_s + \
                       binom_coeff(2, 1) * ((p_s)) * eps * fail_and_succ_no_error_y) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y

    traj_eigth_detection = ((binom_coeff(2, 1) * (1 - eps_f)) * eps_f + (
            eps_f ** 2)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y

    ## p_fail_x * p_fail_x * p_s * eta ** 2

    traj_nine_error = (eps_p_succ * ((1- eps_p_fail_x) ** 2) * p_s * p_s + 2 * eps_p_fail_x * (1 - eps_p_succ) * (1 - eps_p_fail_x) * (p_s ** 2) + eps_p_succ * eps_p_fail_x * eps_p_fail_x * (p_s ** 2) + \
                              2 * eps_p_fail_x * eps_p_succ * (1 - eps_p_fail_x) * 2 * eps * p_s + 2 * eps_p_fail_x * eps_p_fail_x * eps * p_s * (1 - eps_p_succ) + 2 * ((1- eps_p_fail_x) ** 2) * (1 - eps_p_succ) * p_s * eps)  * log_succ * (
                                   sing_trans ** 2) * log_p_fail_x * log_p_fail_x

    traj_nine_detection = ((binom_coeff(2, 1) * (1 - eps_f)) * eps_f + (
            eps_f ** 2)) * log_succ * (
                                   sing_trans ** 2) * log_p_fail_x * log_p_fail_x

    ## p_fail_x * p_fail_x * p_s * p_fail_z
    traj_ten_error = (2 * eps_p_fail_x * (1 - eps_p_fail_x) * (1 - eps_p_succ) * (1 -eps_p_fail_z) + ((1 - eps_p_fail_x) ** 2) * (1 - eps_p_succ) * eps_p_fail_z + \
                     (eps_p_fail_x ** 2) * (eps_p_fail_z * (1 - eps_p_succ) + eps_p_succ * (1 - eps_p_fail_z)) + 2 * eps_p_fail_x * (1 - eps_p_fail_x) * eps_p_succ * eps_p_fail_z) * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ


    log_succ_this_layer = log_fusion_prob(log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
    epsilon_up, epsilon_f_up, eta_up = error_prop_layer_with_loss(eps, eps_f, sing_trans) # log_transmission(sing_trans)
    log_fail_x_this_layer, log_fail_z_this_layer = log_failure(log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
    # log_fail_x_this_layer, log_fail_z_this_layer = 0, 0
    log_fail_y_this_layer = 0

    log_succ_error = (traj_one_error + traj_two_error + traj_three_error + traj_four_error + traj_five_error + traj_six_error + traj_seven_error + traj_eigth_error + traj_nine_error + traj_ten_error) / log_succ_this_layer
    error_detection_prob = (traj_one_detection + traj_two_detection + traj_three_detection + traj_four_detection + traj_five_detection + traj_six_detection + traj_seven_detection + traj_eigth_detection + traj_nine_detection)

    log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer = log_failure_errors(eps, eps_f, eps_p_fail_x, eps_p_fail_y, eps_p_fail_z, eps_p_succ, log_succ, log_p_fail_x,
                       log_p_fail_y, log_p_fail_z, log_lost, sing_trans)

    # print("here:", traj_one_error, traj_two_error,traj_three_error, traj_four_error, traj_five_error, traj_six_error, traj_seven_error,
    #       traj_eigth_error, traj_nine_error, traj_ten_error)
    # print("here detect:", traj_one_detection, traj_two_detection, traj_three_detection, traj_four_detection, traj_five_detection, traj_six_detection, traj_seven_detection, traj_eigth_detection, traj_nine_detection)
    # print("epsilon detect: ", eps_f)
    return log_succ_error, error_detection_prob, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f_up, eta_up



def log_failure_errors(eps, eps_f, eps_p_fail_x, eps_p_fail_y, eps_p_fail_z, eps_p_succ, log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans):
    p_s = 1 - eps - eps_f
    term_one = (eps_p_succ * (p_s ** 2) + eps_p_succ * eps * eps + 2 * eps * p_s * (1 - eps_p_succ)) * (log_lost * log_succ * sing_trans * sing_trans) * ((1 - sing_trans) ** 2)
    term_two = (eps_p_succ * (1 - eps_p_fail_y) + eps_p_fail_y * (1 - eps_p_succ)) * (log_lost * log_succ * log_p_fail_y) * ((1 - sing_trans) ** 2)
    term_three = (binom_coeff(4, 1) * eps_p_fail_x * ((1 - eps_p_fail_x) ** 3) + binom_coeff(4, 3) * (eps_p_fail_x ** 3) * (1 - eps_p_fail_x)) * (log_p_fail_x ** 4)
    log_p_fail_x_this_layer = (term_one + term_two + term_three) / ((log_p_fail_x ** 4) + (log_lost * log_succ * log_p_fail_y) * ((1 - sing_trans) ** 2) + (log_lost * log_succ * sing_trans * sing_trans) * ((1 - sing_trans) ** 2))

    log_p_fail_y_this_layer = 0

    term_one = (eps_p_succ * (p_s ** 2) + (1 - eps_p_succ) * binom_coeff(2, 1) * eps * p_s + eps_p_succ * (eps ** 2)) * (log_succ * sing_trans * sing_trans) * ((1 - sing_trans) ** 4)
    term_two = (eps_p_succ * (1 - eps_p_fail_x) + eps_p_fail_x * (1 - eps_p_succ)) * (log_p_fail_x * log_succ)* ((1 - sing_trans) ** 4)
    term_three = (eps_p_fail_x * (p_s ** 4) + eps_p_fail_x * binom_coeff(4, 2) * eps * eps * p_s * p_s + (1 - eps_p_fail_x) * binom_coeff(4, 1) * eps * (p_s ** 3) + (1 - eps_p_fail_x) * binom_coeff(4, 3) * (eps ** 3) * p_s) * log_p_fail_x * log_lost * (sing_trans ** 4)
    log_p_fail_z_this_layer = (term_one + term_two + term_three) / (log_p_fail_x * log_lost * (sing_trans ** 4) + (log_p_fail_x * log_succ)* ((1 - sing_trans) ** 4) + (log_succ * sing_trans * sing_trans) * ((1 - sing_trans) ** 4))

    return log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer

def log_transmission(eta):
    eta_up = eta ** 4 + 4 * (eta ** 3) * (1 - eta) + 2 * ((1 - eta) ** 2) * (eta ** 2)
    return eta_up

def error_prop_layer(epsilon, epsilon_f):
    p_s = 1 - epsilon - epsilon_f
    # epsilon_up = 4 * epsilon * epsilon * (1 - epsilon) * (1 - epsilon) + epsilon_f * 2 * epsilon * (1-epsilon)
    # epsilon_up = 4 * epsilon * epsilon * (p_s ** 2) + (2 * (2 * (p_s + epsilon) * epsilon_f + epsilon_f * epsilon_f)) * 2 * epsilon * p_s
    epsilon_up = 4 * epsilon * epsilon * (p_s ** 2) + (
                2 * (1-(1 - epsilon_f) ** 2)) * 2 * epsilon * p_s
    # epsilon_up = 4 * epsilon * epsilon * (p_s ** 2) + (
    #                 2 * ((epsilon_f) ** 2)) * 2 * epsilon * p_s
    # epsilon_f_up = 4 * epsilon * ((1-epsilon) ** 3) + 4 * (epsilon ** 3) * (1 - epsilon)
    epsilon_f_up = 4 * epsilon * (p_s ** 3) + 4 * (epsilon ** 3) * p_s + 4 * epsilon_f * epsilon_f * ((1 - epsilon_f)**2) + 4 * epsilon_f * epsilon_f * epsilon_f * (1- epsilon_f) + epsilon_f ** 4
    return epsilon_up, epsilon_f_up

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


def final_correction_layer(p_s, p_l, sing_trans):
    term_one = p_s ** 4  ##  All succed
    term_two = p_s * p_l * (sing_trans ** 4) ## first succeed and second is lost
    term_three = p_s * p_s * p_l * (sing_trans ** 2)  ## two first succed and third is lost
    term_four = (p_s ** 3) * p_l  ## first three succed and last is lost
    term_five = p_l * p_s * (sing_trans ** 4)  ## first is lost
    return term_one + term_two + term_three + term_four + term_five

def final_correction_layer_error_detection(eps, eps_f):
    return 4 * eps * ((1 - eps) ** 3)

def final_correction_layer_error(eps, eps_f):
    p_s = 1 - eps - eps_f
    return 4 * (eps ** 2) * (p_s ** 2)


def fault_tolerant_fusion_layers_error(p_s, p_l, sing_trans, eps, eps_f, eps_sing, eps_f_sing):
    logical_fusion_succes = final_correction_layer(p_s, p_l, sing_trans)
    no_error_or_detect = 1 - eps - eps_f
    # print(logical_fusion_succes)
    term_one_error, term_one_detect = error_prop_layer(eps, eps_f) # 4 * eps * (no_error_or_detect ** 3) * (p_s ** 4)
    # term_one_detect = 4 * (eps * (no_error_or_detect ** 3) + (eps**3) * no_error_or_detect) * (p_s ** 4)
    term_one_error = term_one_error * (p_s ** 4)
    term_one_detect = term_one_detect * (p_s ** 4)
    term_two_error = p_s * p_l * (sing_trans ** 4) * (eps * (p_s ** 4) + eps * binom_coeff(4, 2) * (eps_sing ** 2) * (p_s ** 2) + binom_coeff(4, 1) * eps * (p_s ** 3) * no_error_or_detect + binom_coeff(4, 3) * (eps_sing ** 3) * p_s * no_error_or_detect )
    term_two_detect = p_s * p_l * (sing_trans ** 4) * (binom_coeff(4, 1) * eps_f_sing * ((1- eps_f_sing) ** 3) + binom_coeff(4, 2) * eps_f_sing * eps_f_sing * ((1- eps_f_sing) ** 2) + binom_coeff(4, 3) * eps_f_sing * eps_f_sing* eps_f_sing * ((1- eps_f_sing)) + eps_f_sing ** 4 + eps_f)

    term_three_error = p_s * p_s * p_l * (sing_trans ** 2) * (2 * eps * no_error_or_detect * p_s * p_s + 2 * eps * no_error_or_detect * (eps_sing ** 2) + 2 * eps * eps * eps_sing * p_s)
    term_three_detect = p_s * p_s * p_l * (sing_trans ** 2) * (2 * eps_f * (1 - eps_f) + eps_f ** 2 + 2 * eps_f_sing * (1 - eps_f_sing) + eps_f_sing ** 2)

    term_four_error = p_l * (p_s ** 3) * (binom_coeff(3, 1) * eps * (no_error_or_detect ** 2) + eps ** 3)
    term_four_detect = p_l * (p_s ** 3) * (binom_coeff(3, 1) * eps_f * ((1 - eps_f) ** 2) + binom_coeff(3, 2) * eps_f * eps_f * ((1 - eps_f)) + eps_f ** 3)

    term_five_error = p_l * p_s * (sing_trans ** 4) * (eps * (p_s ** 4) + eps * binom_coeff(4, 2) * (eps_sing ** 2) * (p_s ** 2) + binom_coeff(4, 1) * eps_sing * (p_s ** 3) + binom_coeff(4, 3) * (eps_sing ** 3) * p_s)
    term_five_detect = p_l * p_s * (sing_trans ** 4) * (eps_f + binom_coeff(4, 1) * eps_f_sing * ((1- eps_f_sing) ** 3) + binom_coeff(4, 2) * (eps_f_sing ** 2) * ((1 - eps_f_sing) ** 2) + binom_coeff(4, 3) * (eps_f_sing ** 3) * (1 - eps_f_sing))

    error_rate = (term_one_error + term_two_error + term_three_error + term_four_error + term_five_error) / logical_fusion_succes
    detection_rate = (term_one_detect + term_two_detect + term_three_detect + term_four_detect + term_five_detect) / logical_fusion_succes
    eps_sing_up, eps_f_sing_up, eta_up = error_prop_layer_with_loss(eps_sing, eps_f_sing, sing_trans)
    # print(term_one_error, term_two_error, term_three_error, term_four_error, term_five_error)
    # print(eps, eps_f, eps_sing, eps_f_sing)
    return error_rate, detection_rate, logical_fusion_succes, 1 - logical_fusion_succes, eps_sing_up, eps_f_sing_up ,eta_up

if __name__ == '__main__':
    from Tree_analytics import *

    # TODO: Probably need to normalize error detection in logical fusion.... Or not
    # fidelities = np.linspace(0.87, 0.999, 100)
    # key_rates = [key_siphing(fid) for fid in fidelities]
    # plt.plot(fidelities, key_rates, color="black")
    # plt.show()

    errors_plot = np.linspace(0, 0.5, 100)
    errors = [2  * eps / 3 for eps in errors_plot]
    detect = []
    log_error = []

    for eps in errors:
        init_eps_f = 0 # intial_eps_f(eps)
        # eps = 4 * eps * eps * (1 - eps) * (1 - eps)
        for _ in range(10):
            eps, init_eps_f = error_prop_layer(eps, init_eps_f)
            # eps, init_eps_f = error_prop_layer(eps, init_eps_f)
        log_error.append(eps)
        detect.append(init_eps_f)


    plt.plot(errors_plot, errors, "k:", label="physical error rate")
    plt.plot(errors_plot, detect, color="red", label="detect")
    plt.plot(errors_plot, log_error, color="black", label="log error")
    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate")
    # plt.yscale("log")
    plt.legend()
    plt.show()


    errors_plot = np.linspace(0, 0.01, 100)
    errors = [2  * eps / 3 for eps in errors_plot]
    detect = []
    log_error = []
    log_trans = []
    eta = 0.8
    for eps in errors:
        init_eps_f = 0 # intial_eps_f_with_loss(eps, eta)
        trans = log_transmission(eta)
        print(trans)
        for _ in range(5):
            eps, init_eps_f, trans = error_prop_layer_with_loss(eps, init_eps_f, trans)
            print("Transmission: ", trans, " , at layer: ", _ + 1)
        log_error.append(eps)
        detect.append(init_eps_f)
        log_trans.append(trans)

    plt.plot(errors_plot, errors, "k:", label="physical error rate")
    plt.plot(errors_plot, detect, label="detect")
    plt.plot(errors_plot, log_error, label="log error")
    plt.yscale("log")
    plt.ylabel("Log error rate")
    plt.xlabel("Physical error rate")
    plt.legend()
    plt.show()


    log_succ_plot = []
    fancy_log_succ = []
    Z_failures = []
    tranmissions = np.linspace(0.5, 1, 100)
    N = 32
    branch_list = [24, 7]
    RGS_succ = [RGS(eta, N, branch_list) for eta in tranmissions]
    for eta in tranmissions:
        # Layer one
        log_succ = log_fusion_prob((1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), 1 - eta ** 2, eta)
        log_p_fail_x, log_p_fail_z = log_failure((1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), 1 - eta ** 2, eta)
        log_p_fail_y = 0
        sing_trans = eta
        print("log succ: ", log_fusion_prob((1 / 2) * (0.86 ** 2), (1 / 2) * (0.86 ** 2), (1 / 2) * (0.86 ** 2), (1 / 2) * (0.86 ** 2), 1 - 0.86 ** 2, 0.86))
        # log_p_fail_x, log_p_fail_z = 0, 0
        for _ in range(3):
            sing_trans = log_transmission(sing_trans)
            log_lost = 1 - log_succ - log_p_fail_z - log_p_fail_x
            log_succ_copy = copy.deepcopy(log_succ)
            log_succ = log_fusion_prob(log_succ, log_p_fail_x,
                                       log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
            log_p_fail_x, log_p_fail_z = log_failure(log_succ_copy, log_p_fail_x,
                                       log_p_fail_y, log_p_fail_z, log_lost, sing_trans)

            if _ == 2:
                Z_fail = (log_lost ** 2) * (sing_trans ** 2)
                Z_failures.append(Z_fail)
                l_succ = final_correction_layer(log_succ, 1 - log_succ, sing_trans)

        log_succ_plot.append(log_succ)
        sing_trans = log_transmission(sing_trans)
        # log_succ_fancy = final_correction_layer(l_succ, 1 - l_succ, sing_trans)  # All fusions in two layers !!
        log_succ_fancy = final_correction_layer(log_succ, 1 - log_succ, sing_trans)
        print("log succ: ", log_succ, ", fancy log succ:", final_correction_layer(log_succ, 1 - log_succ, sing_trans))
        fancy_log_succ.append(log_succ_fancy)
    plt.plot(tranmissions, RGS_succ, "--", color="red", label="RGS")
    plt.plot(tranmissions, log_succ_plot, color="black", label="Concat. rings")
    plt.plot(tranmissions, fancy_log_succ, color="purple", label="Fancy error correcting layer")
    plt.plot(tranmissions, Z_failures, color="orange", label="Failure in Z")
    plt.xlabel("Transmission $\eta$")
    plt.ylabel("Fusion success prob.")
    plt.legend()
    plt.show()


    #######################################################################################################
    ############ TESTING ERROR AND DETECTION RATE FOR LOGICAL FUSION WITH 14 % LOSS  ######################
    #######################################################################################################

    eta_init = 0.88
    err = np.linspace(0, 0.5, 100)
    errors = [2 * eps / 3 for eps in err] # Adjusting the error rate to a depolarizing channel
    log_error_plot = []
    error_detect_plot = []
    fancy_detection_prob = []
    fancy_error_prob = []
    for eps in errors:
        log_succ_error, error_detection_prob, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f_up, eta_up = \
            log_fusion_error_prob(eps, 0, failed_flip(eps), failed_flip(eps), failed_flip(eps), parity_flip(eps), (1 / 2) * (eta_init ** 2), (1 / 2) * (eta_init ** 2) , (1 / 2) * (eta_init ** 2), (1 / 2) * (eta_init ** 2), 1 - eta_init ** 2, eta_init)
        print("Errors first layer: ", log_succ_error, error_detection_prob)
        init_eps_f = intial_eps_f_with_loss(eps, eta_init)
        sing_trans = log_transmission(eta_init)
        epsilon_up = epsilon_up
        epsilon_f = init_eps_f
        err_detect = 0
        log_lost = 1 - log_succ_this_layer - log_fail_x_this_layer - log_fail_y_this_layer - log_fail_z_this_layer
        print(log_succ_this_layer)
        print("eps: ", eps, " , log fusion error: ", log_succ_error)
        for _ in range(3):
            log_succ_error, error_detection_prob, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f, sing_trans = \
                log_fusion_error_prob(epsilon_up, epsilon_f, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_error, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, log_lost, sing_trans)
            print("Error detection: ", err_detect, "layer: ", _, "from function: ", error_detection_prob, " error: ", log_succ_error)
            err_detect += error_detection_prob * (1- err_detect)  # stop if fusion succeds and an error is detected
            log_lost = 1 - log_succ_this_layer - log_fail_x_this_layer - log_fail_y_this_layer - log_fail_z_this_layer

        log_error_plot.append(log_succ_error)
        error_detect_plot.append(err_detect)

        log_lost = 1 - log_succ_this_layer
        print("log succ", log_succ_this_layer)
        for _ in range(1):
            log_succ_error, error_detection_prob, log_succ_this_layer, log_lost, epsilon_up, epsilon_f, sing_trans = fault_tolerant_fusion_layers_error(log_succ_this_layer, log_lost, sing_trans, log_succ_error,error_detection_prob, epsilon_up, epsilon_f)
            # log_succ_error, err_detect = error_prop_layer(log_succ_error, err_detect)
            print("log_succ_error", log_succ_error, ", error detection", error_detection_prob, ", sum ", error_detection_prob+ log_succ_error)
        print("log succ", log_succ_this_layer)
        fancy_error_prob.append(log_succ_error)
        fancy_detection_prob.append(error_detection_prob)
        # fancy_detection_prob.append(final_correction_layer_error_detection(log_succ_error, err_detect))
        # fancy_error_prob.append(final_correction_layer_error(log_succ_error, err_detect))
    detection_abort = [1-(1-eps_d)**330 for eps_d in fancy_detection_prob]
    plt.plot(err, log_error_plot, color="red", label="Log error second last layer")
    plt.plot(err, error_detect_plot, color="black", label="Detection second last layer")
    plt.plot(err, fancy_detection_prob, color="purple", label="Fancy detection prob. last layer")
    # plt.plot(err, detection_abort, color="orange", label="Fancy detection prob. abort")
    plt.plot(err, fancy_error_prob, color="orange", label="Fancy error prob. last layer")
    plt.plot(err, err, "k:", label="Physical error rate")
    plt.xlabel("Physical error rate")
    plt.legend()
    plt.show()


