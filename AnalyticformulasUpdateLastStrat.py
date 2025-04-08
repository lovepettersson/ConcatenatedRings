import numpy as np
import math
import copy
from Analyticformulas import *


def log_fusion_and_fail_error_prob_individ_parities_with_detection(eps, eps_f, eps_p_fail_x, eps_p_fail_y, eps_p_fail_z, log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans, ZZ_par, XX_par, YY_par, ZZ_par_det, XX_par_detect, YY_par_detect):


    # This is the first layer of logical fusion, i.e., called once in the beginning of the concatenation
    p_s = 1 - eps - eps_f
    no_err_detect_XX = 1 - XX_par - XX_par_detect
    no_err_detect_YY = 1 - YY_par - YY_par_detect
    no_err_detect_ZZ = 1 - ZZ_par - ZZ_par_det


    ### p_s (eta ** 3 + 3 * (1 - eta) * eta ** 2) ** 2

    sing_error = binom_coeff(4, 1) * eps * (p_s ** 3) + binom_coeff(4, 3) * (
            eps ** 3) * p_s
    traj_one_error_ZZ = (2 * eps * p_s * no_err_detect_XX + (p_s ** 2) * XX_par) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                        (binom_coeff(3, 1) * eps * p_s * no_err_detect_XX + XX_par * (p_s ** 3)) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                        (sing_error * no_err_detect_XX + XX_par * (1 - sing_error)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)

    traj_one_error_XX = (2 * eps * p_s * no_err_detect_ZZ + (p_s ** 2) * ZZ_par) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                        (binom_coeff(3, 1) * eps * p_s * no_err_detect_ZZ + ZZ_par * (p_s ** 3)) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                        (sing_error * no_err_detect_ZZ + ZZ_par * (1 - sing_error)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)

    traj_one_error_YY = (2 * eps * p_s * no_err_detect_YY + (p_s ** 2) * YY_par) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                        (binom_coeff(3, 1) * eps * p_s * no_err_detect_YY + YY_par * (p_s ** 3)) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                        (sing_error * no_err_detect_YY + YY_par * (1 - sing_error)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)



    traj_one_detect_ZZ = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * log_succ * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                         (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 3)) * log_succ * 2 * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                         (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 4)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)

    traj_one_detect_XX = (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 2)) * log_succ * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                         (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 3)) * log_succ * 2 * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                         (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 4)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)

    traj_one_detect_YY = (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 2)) * log_succ * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                         (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 3)) * log_succ * 2 * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                         (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 4)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)

    ## p_fail_x * p_s * (eta ** 2 + 2 * eta * (1 - eta))**2

    traj_two_error_ZZ = (ZZ_par * (1 - eps_p_fail_x) + eps_p_fail_x * no_err_detect_ZZ) * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1- sing_trans) * sing_trans) ** 2)
    traj_sing_error = 2 * eps * p_s
    traj_two_error_XX = (traj_sing_error * no_err_detect_YY + YY_par * p_s * p_s) * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1- sing_trans) * sing_trans) ** 2)
    traj_two_error_YY = (traj_sing_error * no_err_detect_XX + XX_par * p_s * p_s) * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1 - sing_trans) * sing_trans) ** 2)


    traj_two_detect_XX = (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 2)) * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1- sing_trans) * sing_trans) ** 2)
    traj_two_detect_ZZ = ZZ_par_det * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1- sing_trans) * sing_trans) ** 2)  # No detection clicks in fusion failures in the first layer
    traj_two_detect_YY = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1 - sing_trans) * sing_trans) ** 2)

    ## p_l * p_s * eta ** 4  ##

    traj_sing_error = binom_coeff(4, 1) * (p_s ** 3) * eps + binom_coeff(4, 1) * p_s * (eps ** 3)

    traj_three_error_ZZ = (traj_sing_error * no_err_detect_ZZ + ZZ_par * (1 - traj_sing_error)) * log_lost * log_succ * (sing_trans ** 4)
    traj_three_error_XX = (YY_par * (p_s ** 2) + 2 * no_err_detect_YY * eps * p_s) * log_lost * log_succ * (sing_trans ** 4)
    traj_three_error_YY = (XX_par * (p_s ** 2) + 2 * no_err_detect_XX * p_s * eps) * log_lost * log_succ * (sing_trans ** 4)

    traj_three_detect_ZZ = (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 4)) * log_lost * log_succ * (sing_trans ** 4)
    traj_three_detect_XX = (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 2)) * log_lost * log_succ * (sing_trans ** 4)
    traj_three_detect_YY = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * log_lost * log_succ * (sing_trans ** 4)

    ## p_l * p_s * p_fail_y * eta ** 2

    sing_error = 2 * eps * p_s
    traj_four_error_ZZ = (ZZ_par * p_s * p_s + sing_error * no_err_detect_ZZ) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y
    traj_four_error_XX = (YY_par * (1 - eps_p_fail_y) + eps_p_fail_y * (no_err_detect_YY)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y
    traj_four_error_YY = (XX_par * (1 - eps_p_fail_y) * p_s * p_s + eps_p_fail_y * no_err_detect_XX * p_s * p_s + sing_error * no_err_detect_XX * (1 - eps_p_fail_y)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y


    traj_four_detect_ZZ = (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 2)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y
    traj_four_detect_XX = YY_par_detect * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y
    traj_four_detect_YY = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y


    ## p_f_x**2 * p_s * eta ** 2

    traj_five_error_ZZ = (XX_par * (1 - eps_p_fail_x) * p_s * p_s + eps_p_fail_x * (no_err_detect_XX) * p_s * p_s + sing_error * (1 - eps_p_fail_x) * no_err_detect_XX) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_five_error_XX = (ZZ_par * (1 - eps_p_fail_x) * p_s * p_s + eps_p_fail_x * (no_err_detect_ZZ) * p_s * p_s + sing_error * (1 - eps_p_fail_x) * no_err_detect_ZZ) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_five_error_YY = (YY_par * ((1 - eps_p_fail_x) ** 2) + 2 * (1 - eps_p_fail_x) * eps_p_fail_x * (no_err_detect_YY) * p_s * p_s) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)



    traj_five_detect_ZZ = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_five_detect_XX = (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 2)) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_five_detect_YY = (1 - (1 - YY_par_detect)) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)

    ## p_f_x ** 2 * p_f_z * p_s

    traj_six_error_ZZ = (XX_par * (1 - eps_p_fail_z) + eps_p_fail_z * (no_err_detect_XX)) * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ
    traj_six_error_XX = (ZZ_par * (1 - eps_p_fail_x) * (1 - eps_p_fail_z) + eps_p_fail_x * (no_err_detect_ZZ) * (1 - eps_p_fail_z) + eps_p_fail_z * (1- eps_p_fail_x) * (no_err_detect_ZZ)) * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ
    traj_six_error_YY = (YY_par * (1 - eps_p_fail_x) + eps_p_fail_x * (no_err_detect_YY)) * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ

    traj_six_detect_ZZ = XX_par_detect * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ
    traj_six_detect_XX = ZZ_par_det * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ
    traj_six_detect_YY = YY_par_detect * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ

    log_succ_this_layer = log_fusion_prob(log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
    epsilon_up, epsilon_f_up, eta_up = error_prop_layer_with_loss(eps, eps_f, sing_trans)
    log_fail_x_this_layer, log_fail_z_this_layer = log_failure(log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans)

    log_fail_y_this_layer = 0


    log_succ_error_ZZ = (traj_one_error_ZZ + traj_two_error_ZZ + traj_three_error_ZZ + traj_four_error_ZZ + traj_five_error_ZZ + traj_six_error_ZZ) / log_succ_this_layer
    log_succ_error_XX = (traj_one_error_XX + traj_two_error_XX + traj_three_error_XX + traj_four_error_XX + traj_five_error_XX + traj_six_error_XX) / log_succ_this_layer
    log_succ_error_YY = (traj_one_error_YY + traj_two_error_YY + traj_three_error_YY + traj_four_error_YY + traj_five_error_YY + traj_six_error_YY) / log_succ_this_layer
    error_detection_prob_ZZ = (traj_one_detect_ZZ + traj_two_detect_ZZ + traj_three_detect_ZZ +\
                               traj_four_detect_ZZ + traj_five_detect_ZZ + traj_six_detect_ZZ) / log_succ_this_layer
    error_detection_prob_XX = (traj_one_detect_XX + traj_two_detect_XX + traj_three_detect_XX + \
                               traj_four_detect_XX + traj_five_detect_XX + traj_six_detect_XX) / log_succ_this_layer
    error_detection_prob_YY = (traj_one_detect_YY + traj_two_detect_YY + traj_three_detect_YY + \
                               traj_four_detect_YY + traj_five_detect_YY + traj_six_detect_YY) / log_succ_this_layer


    log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_p_fail_x_detect_this_layer, log_p_fail_y_detect_this_layer, log_p_fail_z_detect_this_layer \
        = log_failure_errors_with_fail(eps, eps_f, eps_p_fail_x, eps_p_fail_y, eps_p_fail_z, log_succ, log_p_fail_x,
                                       0, log_p_fail_y, 0, log_p_fail_z,
                                       0,
                                       log_lost, sing_trans, ZZ_par, XX_par, YY_par, ZZ_par_det, XX_par_detect,
                                       YY_par_detect)

    return log_succ_error_ZZ, log_succ_error_XX, log_succ_error_YY, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f_up, eta_up, \
           error_detection_prob_ZZ, error_detection_prob_XX, error_detection_prob_YY, log_p_fail_x_detect_this_layer, log_p_fail_y_detect_this_layer, log_p_fail_z_detect_this_layer




def log_fusion_error_prob_individ_parities_with_detection_with_fail(eps, eps_f, eps_p_fail_x, eps_p_fail_y, eps_p_fail_z, log_succ, log_p_fail_x, log_p_fail_x_detect, log_p_fail_y, log_p_fail_y_detect, log_p_fail_z, log_p_fail_z_detect,
                                                                    log_lost, sing_trans, ZZ_par, XX_par, YY_par, ZZ_par_det, XX_par_detect, YY_par_detect):

    # eps = pauli meas error rate, eps_f = pauli meas detect rate, eps_p_fail_i = fusion failure error rate in basis "i",
    # log_succ = fusion success probability, log_lost = fusion lost, , log_p_fail_i = fusion failure rate probability in basis "i",
    # log_p_fail_i_detect = fusion failure detection rate in basis "i", sing_trans = Photon transmission, ZZ/XX/YY_par = error rate in parites from successfull fusions,
    # ZZ/XX/YY_par_det = detection rate in parities from successfull fusions.

    # Logical fusion for the loss tolerant layers.

    p_s = 1 - eps - eps_f
    no_err_detect_XX = 1 - XX_par - XX_par_detect
    no_err_detect_YY = 1 - YY_par - YY_par_detect
    no_err_detect_ZZ = 1 - ZZ_par - ZZ_par_det

    no_error_or_detect_p_z = 1 - eps_p_fail_z - log_p_fail_z_detect
    no_error_or_detect_p_x = 1 - eps_p_fail_x - log_p_fail_x_detect
    no_error_or_detect_p_y = 1 - eps_p_fail_y - log_p_fail_y_detect

    ### p_s (eta ** 3 + 3 * (1 - eta) * eta ** 2) ** 2

    sing_error = binom_coeff(4, 1) * eps * (p_s ** 3) + binom_coeff(4, 3) * (eps ** 3) * p_s
    traj_one_error_ZZ = (2 * eps * p_s * no_err_detect_XX + (p_s ** 2) * XX_par) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                      (binom_coeff(3, 1) * eps * p_s * no_err_detect_XX + XX_par * (p_s ** 3)) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                      (sing_error * no_err_detect_XX + XX_par * (1 - sing_error)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)

    traj_one_error_XX = (2 * eps * p_s * no_err_detect_ZZ + (p_s ** 2) * ZZ_par) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                        (binom_coeff(3, 1) * eps * p_s * no_err_detect_ZZ + ZZ_par * (p_s ** 3)) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                        (sing_error * no_err_detect_ZZ + ZZ_par * (1 - sing_error)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)

    traj_one_error_YY = (2 * eps * p_s * no_err_detect_YY + (p_s ** 2) * YY_par) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                        (binom_coeff(3, 1) * eps * p_s * no_err_detect_YY + YY_par * (p_s ** 3)) * log_succ * ((sing_trans * (sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                        (sing_error * no_err_detect_YY + YY_par * (1 - sing_error)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)




    traj_one_detect_ZZ = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * log_succ * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                         (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 3)) * log_succ * 2 * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                         (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 4)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)

    traj_one_detect_XX = (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 2)) * log_succ * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                         (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 3)) * log_succ * 2 * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                         (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 4)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)

    traj_one_detect_YY = (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 2)) * log_succ * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) ** 2) + \
                         (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 3)) * log_succ * 2 * ((sing_trans * (2 * sing_trans * (1 - sing_trans) + sing_trans ** 2)) * (1 - sing_trans) * (sing_trans ** 2)) + \
                         (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 4)) * log_succ * (((1 - sing_trans) * (sing_trans ** 2)) ** 2)

    ## p_fail_x * p_s * (eta ** 2 + 2 * eta * (1 - eta)) ** 2

    traj_sing_error = 2 * eps * p_s
    traj_two_error_ZZ = (ZZ_par * no_error_or_detect_p_x + eps_p_fail_x * (no_err_detect_ZZ)) * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1- sing_trans) * sing_trans) ** 2)
    traj_two_error_XX = (traj_sing_error * (no_err_detect_YY) + YY_par * p_s * p_s) * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1- sing_trans) * sing_trans) ** 2)
    traj_two_error_YY = (traj_sing_error * (no_err_detect_XX) + XX_par * p_s * p_s) * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1 - sing_trans) * sing_trans) ** 2)

    traj_two_detect_XX = (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 2)) * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1- sing_trans) * sing_trans) ** 2)
    traj_two_detect_ZZ = (1 - (1 - ZZ_par_det) * (1 - log_p_fail_x_detect)) * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1- sing_trans) * sing_trans) ** 2)
    traj_two_detect_YY = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * log_succ * log_p_fail_x * ((sing_trans ** 2 + 2 * (1 - sing_trans) * sing_trans) ** 2)

    ## p_l * p_s * eta ** 4  ##

    traj_sing_error = binom_coeff(4, 1) * (p_s ** 3) * eps + binom_coeff(4, 1) * p_s * (eps ** 3)

    traj_three_error_ZZ = (traj_sing_error * (no_err_detect_ZZ) + ZZ_par * (1 - traj_sing_error)) * log_lost * log_succ * (sing_trans ** 4)
    traj_three_error_XX = (YY_par * (p_s ** 2) + 2 * (no_err_detect_YY) * eps * p_s) * log_lost * log_succ * (sing_trans ** 4)
    traj_three_error_YY = (XX_par * (p_s ** 2) + 2 * (no_err_detect_XX) * eps * p_s) * log_lost * log_succ * (sing_trans ** 4)

    traj_three_detect_ZZ = (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 4)) * log_lost * log_succ * (sing_trans ** 4)
    traj_three_detect_XX = (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 2)) * log_lost * log_succ * (sing_trans ** 4)
    traj_three_detect_YY = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * log_lost * log_succ * (sing_trans ** 4)

    ## p_l * p_s * p_fail_y * eta ** 2

    sing_error = 2 * eps * p_s
    traj_four_error_ZZ = (ZZ_par * p_s * p_s + sing_error * (no_err_detect_ZZ)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y
    traj_four_error_XX = (YY_par * no_error_or_detect_p_y + eps_p_fail_y * (no_err_detect_YY)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y
    traj_four_error_YY = (XX_par * no_error_or_detect_p_y * p_s * p_s + eps_p_fail_y * no_err_detect_XX * p_s * p_s + sing_error * no_err_detect_XX * (1 - eps_p_fail_y)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y


    traj_four_detect_ZZ = (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 2)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y
    traj_four_detect_XX = (1 - (1 - YY_par_detect) * (1 - log_p_fail_y_detect)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y
    traj_four_detect_YY = (1 - (1 - XX_par_detect) * (1 - log_p_fail_y_detect) * ((1 - eps_f) ** 2)) * log_lost * log_succ * (sing_trans ** 2) * log_p_fail_y


    ## p_f_x**2 * p_s * eta ** 2

    traj_five_error_ZZ = (XX_par * no_error_or_detect_p_x * p_s * p_s + eps_p_fail_x * (no_err_detect_XX) * p_s * p_s + sing_error * no_error_or_detect_p_x * (no_err_detect_XX)) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_five_error_XX = (ZZ_par * no_error_or_detect_p_x * p_s * p_s + eps_p_fail_x * (no_err_detect_ZZ) * p_s * p_s + sing_error * no_error_or_detect_p_x * (no_err_detect_ZZ)) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_five_error_YY = (YY_par * (no_error_or_detect_p_x ** 2) + 2 * eps_p_fail_x * (no_err_detect_XX) * no_error_or_detect_p_x) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)


    traj_five_detect_ZZ = (1 - (1 - XX_par_detect) * (1 - log_p_fail_x_detect) * ((1 - eps_f) ** 2)) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_five_detect_XX = (1 - (1 - ZZ_par_det) * (1 - log_p_fail_x_detect) * ((1 - eps_f) ** 2)) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_five_detect_YY = (1 - (1 - YY_par_detect) * ((1 - log_p_fail_x_detect) ** 2)) * log_p_fail_x * log_p_fail_x * log_succ * (sing_trans ** 2)

    ## p_f_x**2 * p_f_z * p_s

    traj_six_error_ZZ = (XX_par * no_error_or_detect_p_z + eps_p_fail_z * (no_err_detect_XX)) * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ
    traj_six_error_XX = (ZZ_par * no_error_or_detect_p_x * no_error_or_detect_p_z + eps_p_fail_x * no_err_detect_ZZ * no_error_or_detect_p_z + eps_p_fail_z * no_error_or_detect_p_x * no_err_detect_ZZ)* log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ
    traj_six_error_YY = (YY_par * no_error_or_detect_p_x + eps_p_fail_x * no_err_detect_YY) * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ

    traj_six_detect_ZZ = (1 - (1 - XX_par_detect) * (1 - log_p_fail_z_detect)) * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ
    traj_six_detect_XX = (1 - (1 - ZZ_par_det) * (1 - log_p_fail_x_detect) * (1 - log_p_fail_z_detect)) * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ
    traj_six_detect_YY = (1 - (1 - YY_par_detect) * (1 - log_p_fail_x_detect)) * log_p_fail_x * log_p_fail_x * log_p_fail_z * log_succ


    # p_l * p_x * eta ** 2 * p_s

    traj_seven_error_ZZ = (XX_par * (p_s ** 2) + 2 * p_s * eps * no_err_detect_XX) * log_lost * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_seven_error_XX = (ZZ_par * (p_s ** 2) * no_error_or_detect_p_x + 2 * p_s * eps * no_err_detect_ZZ * no_error_or_detect_p_x + eps_p_fail_x * no_err_detect_ZZ * (p_s ** 2)) * log_lost * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_seven_error_YY = (YY_par * no_error_or_detect_p_x + eps_p_fail_x * no_err_detect_XX) * log_lost * log_p_fail_x * log_succ * (sing_trans ** 2)

    traj_seven_detect_ZZ = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * log_lost * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_seven_detect_XX = (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 2) * (1 - log_p_fail_x_detect)) * log_lost * log_p_fail_x * log_succ * (sing_trans ** 2)
    traj_seven_detect_YY = (1 - (1 - YY_par_detect) * (1 - log_p_fail_x_detect)) * log_lost * log_p_fail_x * log_succ * (sing_trans ** 2)

    # p_z * p_s * eta ** 4

    sing_error = binom_coeff(4, 1) * eps * (p_s ** 3) + binom_coeff(4, 3) * (eps ** 3) * p_s

    traj_eigth_error_ZZ = (ZZ_par * (1 - sing_error) + no_err_detect_ZZ * sing_error) * log_p_fail_z * log_succ * (sing_trans ** 4)
    traj_eigth_error_XX = (YY_par * (p_s ** 2) + 2 * eps * p_s * no_err_detect_YY) * log_p_fail_z * log_succ * (sing_trans ** 4)
    traj_eigth_error_YY = (XX_par * (p_s ** 2) + 2 * eps * p_s * no_err_detect_XX) * log_p_fail_z * log_succ * (sing_trans ** 4)

    traj_eigth_detect_ZZ = (1 - (1 - ZZ_par_det) * ((1 - eps_f) ** 4)) * log_p_fail_z * log_succ * (sing_trans ** 4)
    traj_eigth_detect_XX = (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 2)) * log_p_fail_z * log_succ * (sing_trans ** 4)
    traj_eigth_detect_YY = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * log_p_fail_z * log_succ * (sing_trans ** 4)

    # p_z * p_y * p_s * eta ** 2

    traj_nine_error_ZZ = (ZZ_par * (p_s ** 2) + 2 * p_s * eps * no_err_detect_ZZ) * log_p_fail_z * log_p_fail_y * log_succ * (sing_trans ** 2)
    traj_nine_error_XX = (YY_par * no_error_or_detect_p_y + eps_p_fail_y * no_err_detect_YY) * log_p_fail_z * log_p_fail_y * log_succ * (sing_trans ** 2)
    traj_nine_error_YY = (XX_par * (p_s ** 2) * no_error_or_detect_p_y + 2 * eps * p_s * no_err_detect_XX * no_error_or_detect_p_y + no_err_detect_XX * (p_s ** 2) * eps_p_fail_y) * log_p_fail_z * log_p_fail_y * log_succ * (sing_trans ** 2)

    traj_nine_detect_ZZ = (1 - (1 - ZZ_par_det) * (1 - eps_f) ** 2) * log_p_fail_z * log_p_fail_y * log_succ * (sing_trans ** 2)
    traj_nine_detect_XX = (1 - (1 - YY_par_detect) * (1 - log_p_fail_y_detect)) * log_p_fail_z * log_p_fail_y * log_succ * (sing_trans ** 2)
    traj_nine_detect_YY = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2) * (1 - log_p_fail_y_detect)) * log_p_fail_z * log_p_fail_y * log_succ * (sing_trans ** 2)

    # log_succ_this_layer = log_fusion_prob(log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans)

    log_succ_this_layer = log_fusion_prob_above_layers(log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
    epsilon_up, epsilon_f_up, eta_up = error_prop_layer_with_loss(eps, eps_f, sing_trans)
    log_fail_x_this_layer, log_fail_z_this_layer = log_failure(log_succ, log_p_fail_x, log_p_fail_y, log_p_fail_z, log_lost, sing_trans)

    log_fail_y_this_layer = 0


    # log_succ_error_ZZ = (traj_one_error_ZZ + traj_two_error_ZZ + traj_three_error_ZZ + traj_four_error_ZZ + traj_five_error_ZZ + traj_six_error_ZZ) / log_succ_this_layer
    # log_succ_error_XX = (traj_one_error_XX + traj_two_error_XX + traj_three_error_XX + traj_four_error_XX + traj_five_error_XX + traj_six_error_XX) / log_succ_this_layer
    # log_succ_error_YY = (traj_one_error_YY + traj_two_error_YY + traj_three_error_YY + traj_four_error_YY + traj_five_error_YY + traj_six_error_YY) / log_succ_this_layer

    # error_detection_prob_ZZ = (traj_one_detect_ZZ + traj_two_detect_ZZ + traj_three_detect_ZZ +traj_four_detect_ZZ + traj_five_detect_ZZ + traj_six_detect_ZZ) / log_succ_this_layer
    # error_detection_prob_XX = (traj_one_detect_XX + traj_two_detect_XX + traj_three_detect_XX + traj_four_detect_XX + traj_five_detect_XX + traj_six_detect_XX) / log_succ_this_layer
    # error_detection_prob_YY = (traj_one_detect_YY + traj_two_detect_YY + traj_three_detect_YY + traj_four_detect_YY + traj_five_detect_YY + traj_six_detect_YY) / log_succ_this_layer

    log_succ_error_ZZ = (traj_one_error_ZZ + traj_two_error_ZZ + traj_three_error_ZZ + traj_four_error_ZZ + traj_five_error_ZZ + traj_six_error_ZZ + traj_seven_error_ZZ + traj_eigth_error_ZZ + traj_nine_error_ZZ) / log_succ_this_layer
    log_succ_error_XX = (traj_one_error_XX + traj_two_error_XX + traj_three_error_XX + traj_four_error_XX + traj_five_error_XX + traj_six_error_XX + traj_seven_error_XX + traj_eigth_error_XX + traj_nine_error_XX) / log_succ_this_layer
    log_succ_error_YY = (traj_one_error_YY + traj_two_error_YY + traj_three_error_YY + traj_four_error_YY + traj_five_error_YY + traj_six_error_YY + traj_seven_error_YY + traj_eigth_error_YY + traj_nine_error_YY) / log_succ_this_layer

    error_detection_prob_ZZ = (traj_one_detect_ZZ + traj_two_detect_ZZ + traj_three_detect_ZZ + traj_four_detect_ZZ + traj_five_detect_ZZ + traj_six_detect_ZZ + traj_seven_detect_ZZ + traj_eigth_detect_ZZ + traj_nine_detect_ZZ) / log_succ_this_layer
    error_detection_prob_XX = (traj_one_detect_XX + traj_two_detect_XX + traj_three_detect_XX + traj_four_detect_XX + traj_five_detect_XX + traj_six_detect_XX + traj_seven_detect_XX + traj_eigth_detect_XX + traj_nine_detect_XX) / log_succ_this_layer
    error_detection_prob_YY = (traj_one_detect_YY + traj_two_detect_YY + traj_three_detect_YY + traj_four_detect_YY + traj_five_detect_YY + traj_six_detect_YY + traj_seven_detect_YY + traj_eigth_detect_YY + traj_nine_detect_YY) / log_succ_this_layer

    log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_p_fail_x_detect_this_layer, log_p_fail_y_detect_this_layer, log_p_fail_z_detect_this_layer \
        = log_failure_errors_with_fail(eps, eps_f, eps_p_fail_x, eps_p_fail_y, eps_p_fail_z, log_succ, log_p_fail_x, log_p_fail_x_detect, log_p_fail_y, log_p_fail_y_detect, log_p_fail_z, log_p_fail_z_detect,
                                 log_lost, sing_trans, ZZ_par, XX_par, YY_par, ZZ_par_det, XX_par_detect, YY_par_detect)

    return log_succ_error_ZZ, log_succ_error_XX, log_succ_error_YY, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f_up, eta_up, \
           error_detection_prob_ZZ, error_detection_prob_XX, error_detection_prob_YY, log_p_fail_x_detect_this_layer, log_p_fail_y_detect_this_layer, log_p_fail_z_detect_this_layer




def log_failure_errors_with_fail(eps, eps_f, eps_p_fail_x, eps_p_fail_y, eps_p_fail_z, log_succ, log_p_fail_x, log_p_fail_x_detect, log_p_fail_y, log_p_fail_y_detect, log_p_fail_z, log_p_fail_z_detect,
                                 log_lost, sing_trans, ZZ_par, XX_par, YY_par, ZZ_par_det, XX_par_detect, YY_par_detect):

    # eps = pauli meas error rate, eps_f = pauli meas detect rate, eps_p_fail_i = fusion failure error rate in basis "i",
    # log_succ = fusion success probability, log_lost = fusion lost, , log_p_fail_i = fusion failure rate probability in basis "i",
    # log_p_fail_i_detect = fusion failure detection rate in basis "i", sing_trans = Photon transmission, ZZ/XX/YY_par = error rate in parites from successfull fusions,
    # ZZ/XX/YY_par_det = detection rate in parities from successfull fusions.

    # Logical failure error and detection rates as we go up with the concatenation layers

    p_s = 1 - eps - eps_f
    no_err_detect_XX = 1 - XX_par - XX_par_detect
    no_err_detect_YY = 1 - YY_par - YY_par_detect
    no_err_detect_ZZ = 1 - ZZ_par - ZZ_par_det

    no_error_or_detect_p_z = 1 - eps_p_fail_z - log_p_fail_z_detect
    no_error_or_detect_p_x = 1 - eps_p_fail_x - log_p_fail_x_detect
    no_error_or_detect_p_y = 1 - eps_p_fail_y - log_p_fail_y_detect


    # (1 - eta ** 2) * p_l * p_s * eta ** 2
    term_one = (YY_par * (p_s ** 2) + YY_par * eps * eps + 2 * eps * p_s * no_err_detect_YY) * (log_lost * log_succ * sing_trans * sing_trans) * ((1 - sing_trans ** 2))
    term_one_detect = (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 2)) * (log_lost * log_succ * sing_trans * sing_trans) * ((1 - sing_trans ** 2))

    # (1 - eta ** 2) * p_l * p_s * p_y
    term_two = (YY_par * no_error_or_detect_p_y + eps_p_fail_y * no_err_detect_YY) * (log_lost * log_succ * log_p_fail_y) * ((1 - sing_trans ** 2))
    term_two_detect = (1 - (1 - YY_par_detect) * (1 - log_p_fail_y_detect)) * (log_lost * log_succ * log_p_fail_y) * ((1 - sing_trans ** 2))


    # p_l * p_s * eta ** 2 * (1 - eta) ** 2

    # term_one = (YY_par * (p_s ** 2) + YY_par * eps * eps + 2 * eps * p_s * no_err_detect_YY) * (log_lost * log_succ * sing_trans * sing_trans) * ((1 - sing_trans) ** 2)
    # term_one_detect = (1 - (1 - YY_par_detect) * ((1 - eps_f) ** 2)) * (log_lost * log_succ * sing_trans * sing_trans) * ((1 - sing_trans) ** 2)

    # p_l * p_y * p_s * (1 - eta) ** 2

    # term_two = (YY_par * no_error_or_detect_p_y + eps_p_fail_y * no_err_detect_YY) * (log_lost * log_succ * log_p_fail_y) * ((1 - sing_trans) ** 2)
    # term_two_detect = (1 - (1 - YY_par_detect) * (1 - log_p_fail_y_detect)) * (log_lost * log_succ * log_p_fail_y) * ((1 - sing_trans) ** 2)

    # p_x * p_x * p_z * p_z

    term_three = (2 * eps_p_fail_x * no_error_or_detect_p_z + eps_p_fail_z * (no_error_or_detect_p_x ** 2)) * ((log_p_fail_x ** 2) * (log_p_fail_z ** 2))
    term_three_detect = (1 - ((1 - log_p_fail_x_detect) ** 2) * (1 - log_p_fail_z_detect)) * ((log_p_fail_x ** 2) * (log_p_fail_z ** 2))


    # tot_fail_prob = (((log_p_fail_x ** 2) * (log_p_fail_z ** 2)) + (log_lost * log_succ * log_p_fail_y) * ((1 - sing_trans) ** 2) + (log_lost * log_succ * sing_trans * sing_trans) * ((1 - sing_trans) ** 2))
    tot_fail_prob = (((log_p_fail_x ** 2) * (log_p_fail_z ** 2)) + (log_lost * log_succ * log_p_fail_y) * ((1 - sing_trans ** 2)) + (log_lost * log_succ * sing_trans * sing_trans) * ((1 - sing_trans ** 2)))

    log_p_fail_x_this_layer = (term_one + term_two + term_three) / tot_fail_prob
    log_p_fail_x_detect_this_layer = (term_one_detect + term_two_detect + term_three_detect) / tot_fail_prob

    log_p_fail_y_this_layer = 0
    log_p_fail_y_detect_this_layer = 0

    # p_s * eta ** 2 * (1 - eta) ** 4

    # term_one = (XX_par * (p_s ** 2) + no_err_detect_XX * 2 * eps * p_s + XX_par * (eps ** 2)) * (log_succ * sing_trans * sing_trans) * ((1 - sing_trans) ** 4)
    # term_one_detect = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * (log_succ * sing_trans * sing_trans) * ((1 - sing_trans) ** 4)

    # p_x * p_s * (1 - eta) ** 4

    # term_two = (ZZ_par * no_error_or_detect_p_x + eps_p_fail_x * no_err_detect_ZZ) * (log_p_fail_x * log_succ) * ((1 - sing_trans) ** 4)
    # term_two_detect = (1 - (1 - ZZ_par_det) * (1 - log_p_fail_x_detect)) * (log_p_fail_x * log_succ) * ((1 - sing_trans) ** 4)


    # p_s eta ** 2 * (4 * eta * (1 - eta) ** 3 + 2 * eta ** 2 * (1 - eta) ** 2)

    term_one = (XX_par * (p_s ** 2) + no_err_detect_XX * binom_coeff(2, 1) * eps * p_s + XX_par * (eps ** 2)) * (log_succ * sing_trans * sing_trans) * (4 * sing_trans * ((1 - sing_trans) ** 3) + ((1 - sing_trans) ** 4) + 2 * (sing_trans ** 2) * ((1 - sing_trans) ** 2))
    term_one_detect = (1 - (1 - XX_par_detect) * ((1 - eps_f) ** 2)) * (log_succ * sing_trans * sing_trans) * (4 * sing_trans * ((1 - sing_trans) ** 3) + ((1 - sing_trans) ** 4) + 2 * (sing_trans ** 2) * ((1 - sing_trans) ** 2))

    # p_x * p_s * (1 - (eta ** 2 +2 * eta * (1 -eta)) ** 2)

    term_two = (ZZ_par * no_error_or_detect_p_x + eps_p_fail_x * no_err_detect_ZZ) * (log_p_fail_x * log_succ) * (1 - ((sing_trans ** 2 + 2 * sing_trans * (1- sing_trans)) ** 2))
    term_two_detect = (1 - (1 - ZZ_par_det) * (1 - log_p_fail_x_detect)) * (log_p_fail_x * log_succ) * (1 - ((sing_trans ** 2 + 2 * sing_trans * (1- sing_trans)) ** 2))


    # p_x * p_l * eta ** 4

    term_three = (eps_p_fail_x * (p_s ** 4) + eps_p_fail_x * binom_coeff(4, 2) * eps * eps * p_s * p_s + \
                  no_error_or_detect_p_x * binom_coeff(4, 1) * eps * (p_s ** 3) + no_error_or_detect_p_x * binom_coeff(4, 3) * (eps ** 3) * p_s) * log_p_fail_x * log_lost * (sing_trans ** 4)
    term_three_detect = (1 - (1 - log_p_fail_x_detect) * ((1 - eps_f) ** 4)) * log_p_fail_x * log_lost * (sing_trans ** 4)


    # p_l * p_l * eta ** 4

    term_four = ((binom_coeff(4, 1) * eps * (p_s ** 3) + binom_coeff(4, 3) * (eps ** 3) * p_s) * (log_lost ** 2) * (sing_trans ** 4))
    term_four_detect = (1 - (1 - eps_f) ** 4) * (log_lost ** 2) * (sing_trans ** 4)

    tot_fail_prob = log_p_fail_x * log_lost * (sing_trans ** 4) + (log_p_fail_x * log_succ) * (1 - ((sing_trans ** 2 + 2 * sing_trans * (1- sing_trans)) ** 2)) \
                   + log_succ * (4 * sing_trans * ((1 - sing_trans) ** 3) + ((1 - sing_trans) ** 4) + 2 * (sing_trans ** 2) * ((1 - sing_trans) ** 2)) +  (log_lost ** 2) * (sing_trans ** 4)

    # tot_fail_prob = log_p_fail_x * log_lost * (sing_trans ** 4) + (log_p_fail_x * log_succ) * (
    #             ((1 - sing_trans) ** 4)) \
    #                 + log_succ * log_p_fail_x * ((1 - sing_trans) ** 4)

    log_p_fail_z_this_layer = (term_one + term_two + term_three + term_four) / tot_fail_prob
    log_p_fail_z_detect_this_layer = (term_one_detect + term_two_detect + term_three_detect + term_four_detect) / tot_fail_prob

    return log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_p_fail_x_detect_this_layer, log_p_fail_y_detect_this_layer, log_p_fail_z_detect_this_layer


def final_correction_layer_with_fail(p_s, p_l, sing_trans, p_x, p_z):

    # p_s = fusion succees, p_l = fusion lost, sing_trans = photon transmission, p_i = failure prob rate in basis "i"

    # Logical fusion success probability for the error correction layers.

    term_one = p_s ** 4  ##  All succed
    term_two = p_s * p_l * (sing_trans ** 4)  ## first succeed and second is lost
    term_three = p_s * p_s * p_l * (sing_trans ** 2)  ## two first succed and third is lost
    term_four = (p_s ** 3) * (1 - p_s)  ## first three succed
    term_five = p_l * p_s * (sing_trans ** 4)  ## first is lost
    term_six = p_s * p_z * ((sing_trans ** 2 + 2 * sing_trans * (1 - sing_trans)) ** 2)  ## first succeed and second fails in Z
    term_seven = p_s * p_x * (sing_trans ** 2)  ## first fails in X and second succeeds
    term_eigth = p_s * p_x * (sing_trans ** 4)  ## first succeeds and second fails in X
    term_nine = p_s * p_z * (sing_trans ** 4)  ## first fails in Z and second succeeds
    term_ten = p_s * p_s * p_z * (sing_trans ** 2)  ## first two succeeds and third fails in Z
    term_eleven = p_s * p_s * p_x  ## first two succeeds and third fails in X

    return term_one + term_two + term_three + term_four + term_five + term_six + term_seven + term_eigth + term_nine + term_ten + term_eleven


def final_correction_layer_failure_probs(p_s, p_l, sing_trans, p_x, p_z):

    # p_s = fusion succees, p_l = fusion lost, sing_trans = photon transmission, p_i = failure prob rate in basis "i"

    # The failure fusion probabilities for the logical fusion error correction layers.

    term_one = p_s * p_z * (4 * sing_trans * ((1 - sing_trans) ** 3) + (1 - sing_trans) ** 4 + 2 * (sing_trans ** 2) * ((1 - sing_trans) ** 2)) ## first succeed and second fails in Z, but lose at least two photons in on graph
    term_two = p_s * p_x * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)  ## first fails in X and second succeeds but then a photon is lost in single qbt. meas.
    term_three = p_s * p_s * p_z * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)  ## first two succeeds and third fails in Z, but then a photon is lost in single qbt. meas
    term_four = p_s * p_s * p_l * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)  ## two first succed and third is lost, but then a photon is lost in single qbt. meas
    failure_z = term_one + term_two + term_three + term_four

    term_one = p_z * p_s * (sing_trans ** 2) * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)  # p_s_2 p_z_1 eta_4
    term_two = p_x * p_s * (sing_trans ** 2) * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)  # p_s_1 p_x_2 eta_4
    term_three = p_s * p_l * (sing_trans ** 2) * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)  # p_s_1 p_l_2 eta_4
    term_four = p_s * p_l * (sing_trans ** 2) * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)  # p_l_1 p_s_2 eta_3
    failure_x = term_one + term_two + term_three + term_four

    return failure_z, failure_x


def final_correction_layer_failure_errors(p_s, p_l, sing_trans, p_x, p_z, XX_par, ZZ_par, YY_par, eps_sing,
                                                             eps_f_sing, XX_par_det, ZZ_par_detect, YY_par_det,
                                                             p_z_fail_error, p_z_detect, p_x_fail_error, p_x_detect):

    # p_s = fusion succees, p_l = fusion lost, sing_trans = photon transmission, p_i = failure prob rate in basis "i", XX/ZZ/YY_par = error prob in parities,
    # eps_sing = pauli meas error prob, eps_f_sing = pauli meas detect prob, XX/ZZ/YY_det = detection prob in parities,
    # p_i_fail_error = error rate in fusion failure in basis "i", p_i_detect = detectoin rate in failure basis "i"

    # Fusion failure error and detect probabilities in the last error correction layer

    no_error_or_detect_X = 1 - XX_par - XX_par_det
    no_error_or_detect_Z = 1 - ZZ_par - ZZ_par_detect
    no_error_or_detect_Y = 1 - YY_par - YY_par_det
    no_error_or_detect_sing = 1 - eps_sing - eps_f_sing
    no_error_or_detect_p_z = 1 - p_z_fail_error - p_z_detect
    no_error_or_detect_p_x = 1 - p_x_fail_error - p_x_detect

    term_one = p_s * p_z * (4 * sing_trans * ((1 - sing_trans) ** 3) + (1 - sing_trans) ** 4 + 2 * (sing_trans ** 2) * ((1 - sing_trans) ** 2))
    term_one_error = (XX_par * no_error_or_detect_p_z + p_z_fail_error * no_error_or_detect_X) * term_one
    term_one_detect = (1 - (1 - XX_par_det) * (1 - p_z_detect)) * term_one

    term_two = p_s * p_x * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)
    term_two_error = (ZZ_par * no_error_or_detect_p_x + p_x_fail_error * no_error_or_detect_Z) * term_two
    term_two_detect = (1 - (1 - ZZ_par_detect) * (1 - p_x_detect)) * term_two

    term_three = p_s * p_s * p_z * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)
    term_three_error = (XX_par * no_error_or_detect_Z + ZZ_par * no_error_or_detect_X) * term_three
    term_three_detect = (1 - (1 - XX_par_det) * (1 - ZZ_par_detect)) * term_three

    term_four = p_s * p_s * p_l * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)
    term_four_error = (XX_par * no_error_or_detect_Z + ZZ_par * no_error_or_detect_X) * term_four
    term_four_detect = (1 - (1 - XX_par_det) * (1 - ZZ_par_detect)) * term_four

    failure_z = term_one + term_two + term_three + term_four

    if failure_z > 0:
        fail_z_error = (term_one_error + term_two_error + term_three_error + term_four_error) / failure_z
        fail_z_detect = (term_one_detect + term_two_detect + term_three_detect + term_four_detect) / failure_z
    else:
        fail_z_error = 0
        fail_z_detect = 0


    term_one = p_z * p_s * (sing_trans ** 2) * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)
    term_one_error = (p_z_fail_error * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_p_z) * term_one
    term_one_detect = (1 - (1 - p_z_detect) * ((1 - eps_f_sing) ** 2)) * term_one

    term_two = p_x * p_s * (sing_trans ** 2) * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)
    term_two_error = (ZZ_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Z) * term_two
    term_two_detect = (1 - (1 - ZZ_par_detect) * ((1 - eps_f_sing) ** 2)) * term_two


    term_three = p_s * p_l * (sing_trans ** 2) * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)
    term_three_error = (ZZ_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Z) * term_three
    term_three_detect = (1 - (1 - ZZ_par_detect) * ((1 - eps_f_sing) ** 2)) * term_three

    term_four = p_s * p_l * (sing_trans ** 2) * (2 * sing_trans * (1 - sing_trans) + (1 - sing_trans) ** 2)
    term_four_error = (YY_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Y) * term_four
    term_four_detect = (1 - (1 - YY_par_det) * ((1 - eps_f_sing) ** 2)) * term_four

    failure_x = term_one + term_two + term_three + term_four
    if failure_x > 0:
        fail_x_error = (term_one_error + term_two_error + term_three_error + term_four_error) / failure_x
        fail_x_detect = (term_one_detect + term_two_detect + term_three_detect + term_four_detect) / failure_x
    else:
        fail_x_error = 0
        fail_x_detect = 0

    return failure_z, fail_z_error, fail_z_detect, failure_x, fail_x_error, fail_x_detect


def fault_tolerant_fusion_layers_error_individ_det_with_fail(p_s, p_l, sing_trans, XX_par, ZZ_par, YY_par, eps_sing,
                                                             eps_f_sing, XX_par_det, ZZ_par_detect, YY_par_det,
                                                             p_z_fail_error, p_z_fail_prob, p_z_detect, p_x_fail_error,
                                                             p_x_fail_prob, p_x_detect):
    # p_s = fusion succees, p_l = fusion lost, sing_trans = photon transmission, XX/ZZ/YY_par = error prob in parities,
    # eps_sing = pauli meas error prob, eps_f_sing = pauli meas detect prob, XX/ZZ/YY_det = detection prob in parities,
    # p_i_fail_error = error rate in fusion failure in basis "i", p_i_fail_prob = failure rate prob in basis "i",
    # p_i_detect = detectoin rate in failure basis "i"

    # Fusion succees, error, and detection probabilities in the last error correction layers.
    # All the terms with "new" are fusion success trajetories that include fusion failures.

    logical_fusion_succes = final_correction_layer_with_fail(p_s, p_l, sing_trans, p_x_fail_prob, p_z_fail_prob)
    no_error_or_detect_X = 1 - XX_par - XX_par_det
    no_error_or_detect_Z = 1 - ZZ_par - ZZ_par_detect
    no_error_or_detect_Y = 1 - YY_par - YY_par_det
    no_error_or_detect_sing = 1 - eps_sing - eps_f_sing
    no_error_or_detect_p_z = 1 - p_z_fail_error - p_z_detect
    no_error_or_detect_p_x = 1 - p_x_fail_error - p_x_detect

    term_one_error_XX_1, term_one_detect_XX = error_prop_layer_fusion(ZZ_par, ZZ_par_detect, YY_par, YY_par_det)
    term_one_error_ZZ_1, term_one_detect_ZZ = error_prop_layer_fusion_ZY(ZZ_par, ZZ_par_detect, XX_par, XX_par_det)  # error_prop_layer_fusion(ZZ_par, ZZ_par_detect, XX_par, XX_par_det)
    term_one_error_YY_1, term_one_detect_YY = error_prop_layer_fusion_ZY(XX_par, XX_par_det, YY_par, YY_par_det)  # error_prop_layer_fusion(XX_par, XX_par_det, YY_par, YY_par_det)
    term_one_error_XX = term_one_error_XX_1 * (p_s ** 4)
    term_one_error_ZZ = term_one_error_ZZ_1 * (p_s ** 4)
    term_one_error_YY = term_one_error_YY_1 * (p_s ** 4)
    term_one_detect_YY = term_one_detect_YY * (p_s ** 4)
    term_one_detect_ZZ = term_one_detect_ZZ * (p_s ** 4)
    term_one_detect_XX = term_one_detect_XX * (p_s ** 4)


    # p_l_1 * p_s_2 * eta ** 4

    sing_error = binom_coeff(4, 1) * eps_sing * (no_error_or_detect_sing ** 3) + binom_coeff(4, 3) * (eps_sing ** 3) * no_error_or_detect_sing
    term_two_error_ZZ = (sing_error * (no_error_or_detect_Z) + ZZ_par * (1 - sing_error)) * p_s * p_l * (sing_trans ** 4)
    term_two_error_XX = (YY_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Y) * p_s * p_l * (sing_trans ** 4)
    term_two_error_YY = (XX_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_X) * p_s * p_l * (sing_trans ** 4)


    term_two_detect_XX = (1 - (1 - YY_par_det) * ((1 - eps_f_sing) ** 2)) * p_s * p_l * (sing_trans ** 4)
    term_two_detect_YY = (1 - (1 - XX_par_det) * ((1 - eps_f_sing) ** 2)) * p_s * p_l * (sing_trans ** 4)
    term_two_detect_ZZ = (1 - (1 - ZZ_par_detect) * ((1 - eps_f_sing) ** 4)) * p_s * p_l * (sing_trans ** 4)

    ## p_s_1 * p_z_2 * (eta ** 2 + 2 * eta * (1 - eta))** 2

    prob_traj_new_two = p_s * p_z_fail_prob * ((sing_trans ** 2 + 2 * sing_trans * (1 - sing_trans)) ** 2)
    term_new_two_error_ZZ = (p_z_fail_error * (no_error_or_detect_X) + XX_par * no_error_or_detect_p_z) * prob_traj_new_two
    term_new_two_error_XX = (ZZ_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Z) * p_s * p_z_fail_prob * (
                                        sing_trans ** 4 + sing_trans * sing_trans * ((1 - sing_trans) ** 2) + 2 * (sing_trans ** 3) * (1 - sing_trans)) + \
                            (ZZ_par * (no_error_or_detect_sing ** 2) * no_error_or_detect_p_z + 2 * no_error_or_detect_Z * eps_sing * no_error_or_detect_sing * no_error_or_detect_p_z + \
                             no_error_or_detect_Z * (no_error_or_detect_sing ** 2) * p_z_fail_error) * p_s * p_z_fail_prob * (3 * sing_trans * sing_trans * ((1 - sing_trans) ** 2) + 2 * (sing_trans ** 3) * (1 - sing_trans))

    term_new_two_error_YY = (YY_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_Y) * p_s * p_z_fail_prob * (
                                    sing_trans ** 4 + sing_trans * sing_trans * ((1 - sing_trans) ** 2) + 2 * (sing_trans ** 3) * (1 - sing_trans)) + \
                            (YY_par * (no_error_or_detect_sing ** 2) * no_error_or_detect_p_z + 2 * no_error_or_detect_Y * eps_sing * no_error_or_detect_sing * no_error_or_detect_p_z + \
                             no_error_or_detect_Y * (no_error_or_detect_sing ** 2) * p_z_fail_error) * p_s * p_z_fail_prob * (3 * sing_trans * sing_trans * ((1 - sing_trans) ** 2) + 2 * (sing_trans ** 3) * (1 - sing_trans))

    term_new_two_detect_ZZ = (1 - (1 - XX_par_det) * (1 - p_z_detect)) * prob_traj_new_two
    term_new_two_detect_XX = (1 - (1 - ZZ_par_detect) * ((1 - eps_f_sing) ** 2)) * p_s * p_z_fail_prob * (
                sing_trans ** 4 + sing_trans * sing_trans * ((1 - sing_trans) ** 2) + 2 * (sing_trans ** 3) * (1 - sing_trans)) + \
                             (1 - (1 - ZZ_par_detect) * ((1 - eps_f_sing) ** 2) * (1 - p_z_detect)) * p_s * p_z_fail_prob * (3 * sing_trans * sing_trans * ((1 - sing_trans) ** 2) + 2 * (sing_trans ** 3) * (1 - sing_trans))
    term_new_two_detect_YY = (1 - (1 - YY_par_det) * ((1 - eps_f_sing) ** 2)) * p_s * p_z_fail_prob * (
                sing_trans ** 4 + sing_trans * sing_trans * ((1 - sing_trans) ** 2) + 2 * (sing_trans ** 3) * (1 - sing_trans)) + \
                             (1 - (1 - YY_par_det) * ((1 - eps_f_sing) ** 2) * (1 - p_z_detect)) * p_s * p_z_fail_prob * (3 * sing_trans * sing_trans * ((1 - sing_trans) ** 2) + 2 * (sing_trans ** 3) * (1 - sing_trans))

    ## p_s_1 * p_x_2 * eta ** 4

    prob_traj_new_three = p_s * p_x_fail_prob * (sing_trans ** 4)
    term_three_new_error_XX = (ZZ_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Z) * prob_traj_new_three
    term_three_new_error_YY = (YY_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Y) * prob_traj_new_three
    term_three_new_error_ZZ = (sing_error * (no_error_or_detect_X) + XX_par * (1 - sing_error)) * prob_traj_new_three

    term_three_new_detect_XX = (1 - (1 - ZZ_par_detect) * ((1 - eps_f_sing) ** 2)) * prob_traj_new_three
    term_three_new_detect_YY = (1 - (1 - YY_par_det) * ((1 - eps_f_sing) ** 2)) * prob_traj_new_three
    term_three_new_detect_ZZ = (1 - (1 - XX_par_det) * ((1 - eps_f_sing) ** 4)) * prob_traj_new_three

    ## p_z_1 * p_s_2 * eta ** 4

    prob_traj_new_four = p_s * p_z_fail_prob * (sing_trans ** 4)
    term_four_new_error_XX = (YY_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Y) * prob_traj_new_four
    term_four_new_error_YY = (XX_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_X) * prob_traj_new_four
    term_four_new_error_ZZ = (sing_error * (no_error_or_detect_Z) + ZZ_par * (1 - sing_error)) * prob_traj_new_four

    term_four_new_detect_XX = (1 - (1 - YY_par_det) * ((1 - eps_f_sing) ** 2)) * prob_traj_new_four
    term_four_new_detect_YY = (1 - (1 - XX_par_det) * ((1 - eps_f_sing) ** 2)) * prob_traj_new_four
    term_four_new_detect_ZZ = (1 - (1 - ZZ_par_detect) * ((1 - eps_f_sing) ** 4)) * prob_traj_new_four

    ## p_x_1 * p_s_2 * eta ** 2

    prob_traj_new_five = p_s * p_x_fail_prob * (sing_trans ** 2)
    term_five_new_error_XX = (YY_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Y) * prob_traj_new_five
    term_five_new_error_YY = (p_x_fail_error * no_error_or_detect_X * (no_error_or_detect_sing ** 2) + no_error_or_detect_p_x * XX_par * (no_error_or_detect_sing ** 2) + \
                              2 * no_error_or_detect_p_x * no_error_or_detect_X * eps_sing * no_error_or_detect_sing) * prob_traj_new_five
    term_five_new_error_ZZ = (p_x_fail_error * (no_error_or_detect_Z) + ZZ_par * no_error_or_detect_p_x) * prob_traj_new_five

    term_five_new_detect_XX = (1 - (1 - YY_par_det) * ((1 - eps_f_sing) ** 2)) * prob_traj_new_five
    term_five_new_detect_YY = (1 - (1 - XX_par_det) * (1 - p_x_detect) * ((1 - eps_f_sing) ** 2)) * prob_traj_new_five
    term_five_new_detect_ZZ = (1 - (1 - ZZ_par_detect) * (1 - p_x_detect)) * prob_traj_new_five

    ## p_s_1 * p_s_2 * p_z_3 * eta ** 2

    prob_traj_new_six = p_s * p_s * p_z_fail_prob * (sing_trans ** 2)
    term_six_new_error_XX = (ZZ_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Z) * prob_traj_new_six
    term_six_new_error_YY = (YY_par * no_error_or_detect_Z * (no_error_or_detect_sing ** 2) + ZZ_par * no_error_or_detect_Y * (no_error_or_detect_sing ** 2) + \
                             2 * no_error_or_detect_Y * no_error_or_detect_Z * eps_sing * no_error_or_detect_sing) * prob_traj_new_six
    term_six_new_error_ZZ = (XX_par * no_error_or_detect_Z + ZZ_par * no_error_or_detect_X) * prob_traj_new_six

    term_six_new_detect_XX = (1 - (1 - ZZ_par_detect) * ((1 - eps_f_sing) ** 2)) * prob_traj_new_six
    term_six_new_detect_YY = (1 - (1 - ZZ_par_detect) * (1 - YY_par_det) * ((1 - eps_f_sing) ** 2)) * prob_traj_new_six
    term_six_new_detect_ZZ = (1 - (1 - XX_par_det) * (1 - ZZ_par_detect)) * prob_traj_new_six

    ## p_s_1 * p_s_2 * p_x_3

    prob_traj_new_seven = p_s * p_s * p_x_fail_prob
    term_seven_new_error_XX = (2 * ZZ_par * no_error_or_detect_Z * no_error_or_detect_p_x + (no_error_or_detect_Z ** 2) * p_x_fail_error) * prob_traj_new_seven
    term_seven_new_error_YY = (YY_par * no_error_or_detect_p_x + no_error_or_detect_Y * p_x_fail_error) * prob_traj_new_seven
    term_seven_new_error_ZZ = (XX_par * no_error_or_detect_Z + ZZ_par * no_error_or_detect_X) * prob_traj_new_seven

    term_seven_new_detect_XX = (1 - ((1 - ZZ_par_detect) ** 2) * (1 - p_x_detect)) * prob_traj_new_seven
    term_seven_new_detect_YY = (1 - (1 - YY_par_det) * (1 - p_x_detect)) * prob_traj_new_seven
    term_seven_new_detect_ZZ = (1 - (1 - XX_par_det) * (1 - ZZ_par_detect)) * prob_traj_new_seven

    ## p_s ** 2 p_l * eta ** 2

    sing_error = 2 * no_error_or_detect_sing * eps_sing
    term_three_error_ZZ = (XX_par * no_error_or_detect_Z + ZZ_par * no_error_or_detect_X) * p_s * p_s * p_l * (sing_trans ** 2)
    term_three_error_XX = (ZZ_par * (no_error_or_detect_sing ** 2) + 2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Z) * p_s * p_s * p_l * (sing_trans ** 2)
    term_three_error_YY = (YY_par * no_error_or_detect_Z * (no_error_or_detect_sing ** 2) + ZZ_par * no_error_or_detect_Y * (no_error_or_detect_sing ** 2) + \
                           no_error_or_detect_Y * no_error_or_detect_Z * sing_error) * p_s * p_s * p_l * (sing_trans ** 2)
    term_three_detect_ZZ = (1 - (1 - XX_par_det) * (1 - ZZ_par_detect)) * p_s * p_s * p_l * (sing_trans ** 2)
    term_three_detect_XX = (1 - (((1 - eps_f_sing) ** 2) * (1 - ZZ_par_detect))) * p_s * p_s * p_l * (sing_trans ** 2)
    term_three_detect_YY = (1 - (1 - YY_par_det) * (1 - ZZ_par_detect) * ((1 - eps_f_sing) ** 2)) * p_s * p_s * p_l * (sing_trans ** 2)

    ## p_s ** 3 * (1 - p_s)


    term_four_error_ZZ = (XX_par * no_error_or_detect_Z + ZZ_par * no_error_or_detect_X) * (1 - p_s) * (p_s ** 3)
    term_four_error_YY = (YY_par * no_error_or_detect_X + XX_par * no_error_or_detect_Y) * (1 - p_s) * (p_s ** 3)
    term_four_error_XX = (2 * YY_par * no_error_or_detect_Y) * (1 - p_s) * (p_s ** 3)

    term_four_detect_ZZ = (1 - (1 - XX_par_det) * (1 - ZZ_par_detect)) * (1 - p_s) * (p_s ** 3)
    term_four_detect_YY = (1 - (1 - XX_par_det) * (1 - YY_par_det)) * (1 - p_s) * (p_s ** 3)
    term_four_detect_XX = (1 - ((1 - YY_par_det) ** 2)) * (1 - p_s) * (p_s ** 3)

    # p_s_1 * p_l_2 * eta ** 4

    sing_error = binom_coeff(4, 1) * eps_sing * (no_error_or_detect_sing ** 3) + binom_coeff(4, 3) * (eps_sing ** 3) * no_error_or_detect_sing
    term_five_error_XX = (2 * eps_sing * no_error_or_detect_sing * no_error_or_detect_Z + ZZ_par * (no_error_or_detect_sing ** 2)) * p_l * p_s * (sing_trans ** 4)
    term_five_error_YY = (2 * no_error_or_detect_sing * eps_sing * no_error_or_detect_Y + YY_par * (no_error_or_detect_sing ** 2)) * p_l * p_s * (sing_trans ** 4)
    term_five_error_ZZ = (no_error_or_detect_X * sing_error + (1 - sing_error) * XX_par) * p_l * p_s * (sing_trans ** 4)

    term_five_detect_ZZ = (1 - (1 - XX_par_det) * ((1 - eps_f_sing) ** 4)) * p_l * p_s * (sing_trans ** 4)
    term_five_detect_XX = (1 - (1 - ZZ_par_detect) * ((1 - eps_f_sing) ** 2)) * p_l * p_s * (sing_trans ** 4)
    term_five_detect_YY = (1 - (1 - YY_par_det) * ((1 - eps_f_sing) ** 2)) * p_l * p_s * (sing_trans ** 4)

    error_rate_ZZ = (term_one_error_ZZ + term_two_error_ZZ + term_three_error_ZZ + term_four_error_ZZ + term_five_error_ZZ + term_new_two_error_ZZ + term_three_new_error_ZZ + term_four_new_error_ZZ + term_five_new_error_ZZ \
                                + term_six_new_error_ZZ + term_seven_new_error_ZZ) / logical_fusion_succes
    error_rate_XX = (term_one_error_XX + term_two_error_XX + term_three_error_XX + term_four_error_XX + term_five_error_XX + term_new_two_error_XX + term_three_new_error_XX + term_four_new_error_XX + term_five_new_error_XX \
                                + term_six_new_error_XX + term_seven_new_error_XX) / logical_fusion_succes
    error_rate_YY = (term_one_error_YY + term_two_error_YY + term_three_error_YY + term_four_error_YY + term_five_error_YY + term_new_two_error_YY + term_three_new_error_YY + term_four_new_error_YY + term_five_new_error_YY \
                                + term_six_new_error_YY + term_seven_new_error_YY) / logical_fusion_succes

    detection_rate_ZZ = (term_one_detect_ZZ + term_two_detect_ZZ + term_three_detect_ZZ + term_four_detect_ZZ + term_five_detect_ZZ + term_new_two_detect_ZZ + term_three_new_detect_ZZ + term_four_new_detect_ZZ + \
                                    term_five_new_detect_ZZ + term_six_new_detect_ZZ + term_seven_new_detect_ZZ) / logical_fusion_succes
    detection_rate_XX = (term_one_detect_XX + term_two_detect_XX + term_three_detect_XX + term_four_detect_XX + term_five_detect_XX + term_new_two_detect_XX + term_three_new_detect_XX + term_four_new_detect_XX + \
                                    term_five_new_detect_XX + term_six_new_detect_XX + term_seven_new_detect_XX) / logical_fusion_succes
    detection_rate_YY = (term_one_detect_YY + term_two_detect_YY + term_three_detect_YY + term_four_detect_YY + term_five_detect_YY + term_new_two_detect_YY + term_three_new_detect_YY + term_four_new_detect_YY + \
                                    term_five_new_detect_YY + term_six_new_detect_YY + term_seven_new_detect_YY) / logical_fusion_succes

    eps_sing_up, eps_f_sing_up, eta_up = error_prop_layer_with_loss(eps_sing, eps_f_sing, sing_trans)

    failure_z, fail_z_error, fail_z_detect, failure_x, fail_x_error, fail_x_detect = final_correction_layer_failure_errors(
        p_s, p_l, sing_trans, p_x_fail_prob, p_z_fail_prob, XX_par, ZZ_par, YY_par, eps_sing,
        eps_f_sing, XX_par_det, ZZ_par_detect, YY_par_det,
        p_z_fail_error, p_z_detect, p_x_fail_error, p_x_detect)

    return error_rate_ZZ, error_rate_XX, error_rate_YY, logical_fusion_succes, 1 - logical_fusion_succes - failure_z - failure_x, eps_sing_up, eps_f_sing_up, eta_up, detection_rate_ZZ, detection_rate_XX, detection_rate_YY, \
        failure_z, fail_z_error, fail_z_detect, failure_x, fail_x_error, fail_x_detect


def succ_ring_with_individ_det_with_fail_traj(N, eps, eta_init, N_first_layers=3):

    # Calculates the logical fusion success, error, and detection rates given concatenation layer N with N_first_layers for lost tolerant fusion.

    log_succ_error_ZZ, log_succ_error_XX, log_succ_error_YY, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, \
    log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f_up, eta_up, log_succ_det_ZZ, log_succ_det_XX, log_succ_det_YY, \
    log_fail_x_this_layer_detect, log_fail_y_this_layer_detect, log_fail_z_this_layer_detect = \
        log_fusion_and_fail_error_prob_individ_parities_with_detection(eps, 0, failed_flip(eps), failed_flip(eps), failed_flip(eps),
                              (1 / 2) * (eta_init ** 2), (1 / 2) * (eta_init ** 2), (1 / 2) * (eta_init ** 2),
                              (1 / 2) * (eta_init ** 2), 1 - eta_init ** 2, eta_init, failed_flip(eps), failed_flip(eps), failed_flip(eps), 0, 0, 0)

    init_eps_f = intial_eps_f_with_loss(eps, eta_init)
    sing_trans = log_transmission(eta_init)
    epsilon_up = epsilon_up
    epsilon_f = init_eps_f
    log_lost = 1 - log_succ_this_layer - log_fail_x_this_layer - log_fail_y_this_layer - log_fail_z_this_layer
    # To be done
    for _ in range(N_first_layers-1):
        log_succ_error_ZZ, log_succ_error_XX, log_succ_error_YY, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, \
        log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f, sing_trans, log_succ_det_ZZ, log_succ_det_XX, log_succ_det_YY, \
            log_fail_x_this_layer_detect, log_fail_y_this_layer_detect, log_fail_z_this_layer_detect = \
            log_fusion_error_prob_individ_parities_with_detection_with_fail(epsilon_up, epsilon_f, log_p_fail_x_this_layer,
                                                                  log_p_fail_y_this_layer,
                                                                  log_p_fail_z_this_layer, log_succ_this_layer,
                                                                  log_fail_x_this_layer, log_fail_x_this_layer_detect,
                                                                  log_fail_y_this_layer, log_fail_y_this_layer_detect, log_fail_z_this_layer,
                                                                  log_fail_z_this_layer_detect, log_lost, sing_trans, log_succ_error_ZZ,
                                                                  log_succ_error_XX, log_succ_error_YY, log_succ_det_ZZ,
                                                                  log_succ_det_XX, log_succ_det_YY)
        log_lost = 1 - log_succ_this_layer - log_fail_x_this_layer - log_fail_y_this_layer - log_fail_z_this_layer


    p_x_detect = log_fail_x_this_layer_detect
    p_z_detect = log_fail_z_this_layer_detect
    for _ in range(N - N_first_layers):
        log_succ_error_ZZ, log_succ_error_XX, log_succ_error_YY, log_succ_this_layer, log_lost, epsilon_up, epsilon_f, sing_trans,log_succ_det_ZZ, log_succ_det_XX, log_succ_det_YY, \
        log_fail_z_this_layer, log_p_fail_z_this_layer, p_z_detect, log_fail_x_this_layer, log_p_fail_x_this_layer, p_x_detect  = \
        fault_tolerant_fusion_layers_error_individ_det_with_fail(log_succ_this_layer, log_lost, sing_trans, log_succ_error_XX, log_succ_error_ZZ,
                                                                 log_succ_error_YY, epsilon_up,
                                                       epsilon_f, log_succ_det_XX, log_succ_det_ZZ, log_succ_det_YY, log_p_fail_z_this_layer,  log_fail_z_this_layer, p_z_detect, log_p_fail_x_this_layer,
                                                                 log_fail_x_this_layer, p_x_detect)

    err_detect = log_succ_det_ZZ * (1 - log_succ_det_XX) + log_succ_det_XX * (1 - log_succ_det_ZZ) + log_succ_det_XX * log_succ_det_ZZ
    log_succ_error = log_succ_error_ZZ * (1 - log_succ_error_XX - log_succ_det_XX) + log_succ_error_XX * (1 - log_succ_error_ZZ - log_succ_det_ZZ) + log_succ_error_XX * log_succ_error_ZZ
    return log_succ_this_layer, err_detect, log_succ_error
