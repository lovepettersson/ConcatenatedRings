import numpy as np
import matplotlib.pyplot as plt

def erasure_error_threshold(p_erase):
    # To be change to the new arch.
    z = np.array([-1.81355971e+04, 5.28526533e+02, -1.28163926e+01, 1.32647569e-01])
    f = np.poly1d(z)
    errors = np.linspace(0, 0.02, 1000)
    erasure = [f(x) for x in errors]
    for i,erase in enumerate(erasure):
        if p_erase < erase:
            error = errors[i-1]
    if p_erase > 0.13258135915692:
        error = 0

    return error


def parity_flip_RUS(epsilon):
    p_flip = 4 * (epsilon * (1 - 3 * epsilon) + (epsilon ** 2))
    fusion_error = 2 * p_flip * (1 - p_flip) + p_flip ** 2
    return fusion_error
def failed_flip_RUS(epsilon):
    return 4 * (epsilon * (1 - 3 * epsilon) + (epsilon ** 2))
    # return 2 * epsilon * (1 - epsilon)

def RUS_iterative_reintialize(eta, n_rounds, eps, erasure_flag=False):
    p_fail = 1 / 2
    p_succes = 0
    p_failure = 1
    last_fail_X = 0
    error_rate = 0
    error_failed = 0
    for n in n_rounds:
        if n == 0:
            p_succes += p_fail * eta * eta
            p_failure *= p_fail * eta * eta
            p_lost = (1 - eta**2)
            error_rate += parity_flip_RUS(eps) * p_fail * eta * eta
            error_failed += failed_flip_RUS(eps)#  * p_fail * eta * eta
        else:
            p_succes += p_failure * (p_fail * eta * eta)
            error_rate += (parity_flip_RUS(eps) * (1 - error_failed) + error_failed * (1- parity_flip_RUS(eps))) * p_failure * p_fail * eta * eta
            # error_rate += (parity_flip(eps) * (1 - error_failed) + error_failed * (
            #             1 - parity_flip(eps))) * p_fail * eta * eta
            # p_lost = p_failure * p_fail * eta * eta * (1 - eta ** 2)
            p_lost_new = p_lost * (1- eta ** 2) + p_failure * (1 - eta ** 2)
            last_fail_X = p_failure * p_fail * eta * eta
            # error_failed = (p_failure * p_fail * eta * eta + p_lost * eta * eta) * (failed_flip(eps) * (1 - error_failed) + error_failed * (1 - failed_flip(eps)))
            error_failed = (failed_flip_RUS(eps) * (1 - error_failed) + error_failed * (1 - failed_flip_RUS(eps)))
            p_failure = p_failure * p_fail * eta * eta + p_lost * eta * eta

            p_lost = p_lost_new
    if erasure_flag:
        erase_x = p_succes + last_fail_X
        erase_z = 1
        erasure_rate = (erase_z + erase_x) / 2
        return erasure_rate
    else:
        return p_succes, error_rate / p_succes


if __name__ == '__main__':
    from Analyticformulas import *
    # TODO: Check error detection again, seems super high for logical fusion.... Kinda makes sense
    erasure_flag = False
    depolar = np.linspace(0, 1, 100)

    print(RUS_iterative_reintialize(1, [0, 1, 2, 3, 4, 5], 0.007/3, erasure_flag=False))
    errors = [eps / 3 for eps in depolar]
    # trans = np.linspace(0, 1, 100)
    colors = ["red", "black", "blue", "orange"]
    for idx, loss in enumerate([0.001, 0.1, 0.2, 0.4]):
        log_errors = []
        for eps in errors:
            succ_prob, err = RUS_iterative_reintialize(1-loss, [x for x in range(64)], eps, erasure_flag)
            log_errors.append(err)
        plt.plot(depolar, log_errors, color=colors[idx])
    plt.show()

    fig, ax = plt.subplots(figsize=(40, 27))
    depolar = np.linspace(0, 0.018, 1000)
    errors = [eps / 3 for eps in depolar]
    losses = np.linspace(0, 0.45, 20)
    log_errors = []
    for loss in losses:
        succ_prob, err = RUS_iterative_reintialize(1 - loss, [x for x in range(64)], eps, erasure_flag)
        erasure_rate = 1 - succ_prob
        error_threshold = erasure_error_threshold(erasure_rate)
        print("error threshold: ", error_threshold)
        for idx, eps in enumerate(errors):
            succ_prob, err = RUS_iterative_reintialize(1 - loss, [x for x in range(64)], eps, erasure_flag)
            # print(err)
            if err > error_threshold:
                print("here", loss, 3 * eps, erasure_rate)
                print(RUS_iterative_reintialize(1 - loss, [x for x in range(64)], errors[idx-1], erasure_flag))
                log_errors.append(depolar[idx-1])
                break
    print(log_errors)
    ax.plot(log_errors, losses, color="red")
    ax.fill_between(log_errors, 0, losses, color="red", alpha=0.3)
    ax.set_xlabel("pauli error rate")
    ax.set_ylabel("photon loss")
    plt.show()


    log_succ_plot = []
    fancy_log_succ = []
    Z_failures = []
    tranmissions = np.linspace(0.5, 1, 100)
    for eta in tranmissions:
        # Layer one
        log_succ = log_fusion_prob((1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2),
                                   (1 / 2) * (eta ** 2), 1 - eta ** 2, eta)
        log_p_fail_x, log_p_fail_z = log_failure((1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2), (1 / 2) * (eta ** 2),
                                                 (1 / 2) * (eta ** 2), 1 - eta ** 2, eta)
        log_p_fail_y = 0
        sing_trans = eta
        print("log succ: ", log_fusion_prob((1 / 2) * (0.86 ** 2), (1 / 2) * (0.86 ** 2), (1 / 2) * (0.86 ** 2),
                                            (1 / 2) * (0.86 ** 2), 1 - 0.86 ** 2, 0.86))
        # log_p_fail_x, log_p_fail_z = 0, 0
        for _ in range(2):
            sing_trans = log_transmission(sing_trans)
            log_lost = 1 - log_succ - log_p_fail_z - log_p_fail_x
            log_succ_copy = copy.deepcopy(log_succ)
            log_succ = log_fusion_prob(log_succ, log_p_fail_x,
                                       log_p_fail_y, log_p_fail_z, log_lost, sing_trans)
            log_p_fail_x, log_p_fail_z = log_failure(log_succ_copy, log_p_fail_x,
                                                     log_p_fail_y, log_p_fail_z, log_lost, sing_trans)

            if _ == 1:
                Z_fail = (log_lost ** 2) * (sing_trans ** 2)
                Z_failures.append(Z_fail)
                l_succ = final_correction_layer(log_succ, 1 - log_succ, sing_trans)

        log_succ_plot.append(log_succ)
        sing_trans = log_transmission(sing_trans)
        # log_succ_fancy = final_correction_layer(l_succ, 1 - l_succ, sing_trans)  # All fusions in two layers !!
        log_succ_fancy = final_correction_layer(log_succ, 1 - log_succ, sing_trans)
        print("log succ: ", log_succ, ", fancy log succ:", final_correction_layer(log_succ, 1 - log_succ, sing_trans))
        fancy_log_succ.append(log_succ_fancy)

    erasure_rate = [(2 * log_succ_plot[i] + Z_failures[i]) / 2 for i in range(len(tranmissions))]
    plt.plot(tranmissions, log_succ_plot, color="black", label="Concat. rings")
    plt.plot(tranmissions, fancy_log_succ, color="purple", label="Fancy error correcting layer")
    plt.plot(tranmissions, Z_failures, color="orange", label="Failure in Z")
    plt.plot(tranmissions, erasure_rate, "k:", label="Erasure rate for 3 layered rings")
    plt.xlabel("Transmission $\eta$")
    plt.ylabel("Fusion success prob.")
    plt.legend()
    plt.show()

    eta_init = 0.999
    err = np.linspace(0, 0.05, 100)
    errors = [2 * eps / 3 for eps in err]  # Adjusting the error rate to a depolarizing channel
    log_error_plot = []
    error_detect_plot = []
    fancy_detection_prob = []
    fancy_error_prob = []
    for eps in errors:
        log_succ_error, error_detection_prob, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f_up, eta_up = \
            log_fusion_error_prob(eps, 0, failed_flip(eps), failed_flip(eps), failed_flip(eps), parity_flip(eps),
                                  (1 / 2) * (eta_init ** 2), (1 / 2) * (eta_init ** 2), (1 / 2) * (eta_init ** 2),
                                  (1 / 2) * (eta_init ** 2), 1 - eta_init ** 2, eta_init)
        print("Errors first layer: ", log_succ_error, error_detection_prob)
        init_eps_f = intial_eps_f_with_loss(eps, eta_init)
        sing_trans = log_transmission(eta_init)
        epsilon_up = epsilon_up
        epsilon_f = init_eps_f
        # error_detection_prob = init_eps_f
        err_detect = 0
        log_lost = 1 - log_succ_this_layer - log_fail_x_this_layer - log_fail_y_this_layer - log_fail_z_this_layer
        print(log_succ_this_layer)
        print("eps: ", eps, " , log fusion error: ", log_succ_error)
        for _ in range(1):
            log_succ_error, error_detection_prob, log_p_fail_x_this_layer, log_p_fail_y_this_layer, log_p_fail_z_this_layer, log_succ_this_layer, log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, epsilon_up, epsilon_f, sing_trans = \
                log_fusion_error_prob(epsilon_up, epsilon_f, log_p_fail_x_this_layer, log_p_fail_y_this_layer,
                                      log_p_fail_z_this_layer, log_succ_error, log_succ_this_layer,
                                      log_fail_x_this_layer, log_fail_y_this_layer, log_fail_z_this_layer, log_lost,
                                      sing_trans)
            print("Error detection: ", err_detect, "layer: ", _, "from function: ", error_detection_prob, " error: ",
                  log_succ_error)
            err_detect += error_detection_prob * (1 - err_detect)  # stop if fusion succeds and an error is detected
            log_lost = 1 - log_succ_this_layer - log_fail_x_this_layer - log_fail_y_this_layer - log_fail_z_this_layer

        log_error_plot.append(log_succ_error)
        error_detect_plot.append(err_detect)

        log_lost = 1 - log_succ_this_layer
        print("log succ", log_succ_this_layer)
        for _ in range(10):
            log_succ_error, error_detection_prob, log_succ_this_layer, log_lost, epsilon_up, epsilon_f, sing_trans = fault_tolerant_fusion_layers_error(
                log_succ_this_layer, log_lost, sing_trans, log_succ_error, error_detection_prob, epsilon_up, epsilon_f)
            # log_succ_error, err_detect = error_prop_layer(log_succ_error, err_detect)
            print("log_succ_error", log_succ_error, ", error detection", error_detection_prob, ", sum ",
                  error_detection_prob + log_succ_error)
        print("log succ", log_succ_this_layer)
        fancy_error_prob.append(log_succ_error)
        fancy_detection_prob.append(error_detection_prob)
        # fancy_detection_prob.append(final_correction_layer_error_detection(log_succ_error, err_detect))
        # fancy_error_prob.append(final_correction_layer_error(log_succ_error, err_detect))
    # plt.plot(err, fancy_detection_prob, color="purple", label="Log fusion detection rate")
    # plt.plot(err, detection_abort, color="orange", label="Fancy detection prob. abort")
    plt.plot(err, fancy_detection_prob, color="purple", label="Log fusion error rate")
    # plt.plot(err, fancy_error_prob, color="orange", label="Log fusion error rate")
    plt.plot(err, err, "k:", label="Physical error rate")
    plt.xlabel("Physical error rate")
    plt.legend()
    plt.show()