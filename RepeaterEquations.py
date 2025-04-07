import math


def binary_entropy_function(x):
    return -x * math.log(x) - (1 - x) * math.log(1-x)

def key_siphing(fid):
    if fid > 0.874:
        return fid - binary_entropy_function(1 - fid) - fid * binary_entropy_function((3 * fid - 1) / (2 * fid))
    else:
        return 0

def number_of_meas(N, n, M_flag=False):
    if M_flag:
        numb_meas = N**(n-1) + 1
    else:
        numb_meas = n + 1
        # numb_meas = 2 * n
        for _ in range(N-2):
            numb_meas += 1
            numb_meas *= n
        numb_meas += 1
    return numb_meas


def number_of_CZ(N, n):
    numb_CZ = 2 * n + n + 1
    for _ in range(N - 2):
        numb_CZ *= n
        numb_CZ += n + 1
    numb_CZ += n + 1
    return numb_CZ


def number_of_photons(N, n):
    return n ** N


def generation_time(N, n, t_CZ, t_meas, t_gen):
    # Times 2 because we have two concatenated rings...
    t = 2 * (number_of_photons(N, n) * t_gen + number_of_CZ(N, n) * t_CZ + number_of_meas(N, n) * t_meas)
    return t


def gen_time_tree(b0, b1, b2, t_gen, t_CZ):
    generation_time = b0 * (100 + b1 * (1 + b2)) * t_gen
    control_phase_time = b0 * (3 + b1) * t_CZ
    return generation_time + control_phase_time




def generation_time_RGS(b_0, b_1, t_CZ, t_meas, t_gen, N):
    gen_t = (1 + b_0 * b_1) * t_gen
    CZ_t = (2 + b_0) * t_CZ
    meas_E_t = (b_0 + 2) * t_meas
    t = N * (gen_t + CZ_t + meas_E_t) + t_meas
    return t


