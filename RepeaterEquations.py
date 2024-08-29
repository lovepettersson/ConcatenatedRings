import numpy as np
import matplotlib.pyplot as plt
import math


def binary_entropy_function(x):
    return -x * math.log(x) - (1 - x) * math.log(1-x)

def key_siphing(fid):
    if fid > 0.874:
        return fid - binary_entropy_function(1 - fid) - fid * binary_entropy_function((3 * fid - 1) / (2 * fid))
    else:
        return 0

def fiber_transmission(L_att, L_0):
    return np.exp(-L_0 / L_att)


def number_of_stations(L, eta, L_att=20):
    # Diveded by two as each photon only experience half the distance, in a one way quantum repeater this would be the full distance.
    n_stations = (-1) * L / (math.log(eta) * L_att) / 2
    return n_stations



def number_of_stations_one_way(L, eta, L_att=20):
    # Diveded by two as each photon only experience half the distance, in a one way quantum repeater this would be the full distance.
    n_stations = (-1) * L / (math.log(eta) * L_att)
    return n_stations


def number_of_meas(N, n, M_flag=False):
    if M_flag:
        numb_meas = N**(n-1) + 1
    else:
        numb_meas = 2 * n
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


def generation_time_no_meas(N, n, t_CZ, t_gen):
    # Times 2 because we have two concatenated rings...
    # print("Numb photons: ", 2 * number_of_photons(N, n), t_gen * 2 * number_of_photons(N, n), " , numb CZ gates: ", 2 * number_of_CZ(N, n), 2 * number_of_CZ(N, n) * t_CZ)
    t = (number_of_photons(N, n) * t_gen + number_of_CZ(N, n) * t_CZ)
    return t


def effective_rate(N, n, L, L_att, eta, t_CZ, t_meas, t_gen, fid, p_s, effective=False):
    N_emitters = N + 1
    n_stations = number_of_stations(L, eta)
    t = generation_time(N, n, t_CZ, t_meas, t_gen)
    if effective:
        eff_rate = key_siphing(fid) * p_s * (1 / t) * (1 / (n_stations * N_emitters)) * (L / L_att)
    else:
        eff_rate = key_siphing(fid) * p_s * (1 / t)
    return eff_rate


if __name__ == '__main__':
    L_0 = 20
    L_s = np.linspace(0.001, L_0/2, 100)
    transmission = [fiber_transmission(L_0, L) for L in L_s]
    plt.plot(L_s,transmission)
    plt.show()

    print(number_of_meas(5, 4))
    print(number_of_CZ(5, 4))
    print(number_of_photons(5, 4))

    t_CZ = 1 / (10**9)
    t_meas = 1 / (10 ** 9)
    t_gen = 1 / (10 ** 10)


    eta = 0.86
    L = 1000
    L_att = 20
    N = 5
    n = 4
    N_emitters = N + 1
    n_stations = number_of_stations(L, eta)
    t = generation_time(N, n, t_CZ, t_meas, t_gen)
    effective_rate = 0.999 * (1 / t) * (1 / (n_stations * N_emitters)) * (L / L_att)

    print(generation_time(5, 4, 10 / (10**9), 100 / (10 ** 9), 1 / (10 ** 9)))
    print(effective_rate)

