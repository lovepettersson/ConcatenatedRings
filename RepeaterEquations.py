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



