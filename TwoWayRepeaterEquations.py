import numpy as np
import math
import matplotlib.pyplot as plt

# NOTES: Johannes found the optimal number of spin qubits per station to be 2 for his 3 emitter per station comparison
# l is spacing of repeater stations, m is the number of repeater stations

def binom_coeff(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

def Z_function(l, p_ent):
    Z = 0
    for k in range(1, l + 2):
        Z += binom_coeff(l + 1, k) * (((-1) ** (k + 1)) / (1 - (1 - p_ent) ** k))
    return Z


def entanglement_link_prob(L, l, eta_d, m, n, L_att=20):
    quot = quotient_function(n * (m + 1), 2 * (l + 1)) # Setting it to the number of spin qubits for the concatenated rings for now....
    p = 1 - (1 - ((eta_d ** 2) / 2) * np.exp((-L /((l + 1) * L_att)))) ** quot
    return p



def entanglement_link_prob_no_m(L, l, eta_d, L_att=20):
    quot = 2 * (l + 1) # Setting it to the number of spin qubits for the concatenated rings for now....
    p = 1 - (1 - ((eta_d ** 2) / 2) * np.exp((-L /((l + 1) * L_att))))
    return p

def rate_two_way(L, l, eta_d, m, n):
    c = 2 * (10 ** 8)  # Speed of light in fiber
    L_m = L * (10 ** 3)
    p_ent = entanglement_link_prob(L, l, eta_d, m, n)
    Z = Z_function(l, p_ent)
    rate = (1 / Z) * (((l + 1) * c) / L_m)
    return rate


def quotient_function(a, b):
    return a // b
    # return math.ceil(a / b)





path_tree = r"tree_p.txt"
path_ring = r"ring_p.txt"

parameters_ring = np.loadtxt(path_ring)
print(parameters_ring)
for idx in range(len(parameters_ring)):
    print(parameters_ring[idx])

Ls = np.linspace(100, 1000, 100)
eta_d = 0.95

L_att = 20
ms = [270, 340]
ns = [5, 3]
# ms = [51, 63]
# ns = [5, 3]
for i in range(2):
    rates = []
    m = ms[i] # 47
    n = ns[i]
    L = 740
    ls_init = [x for x in range(int(L/50), L)]
    ls = []
    for l in ls_init:
        if l % 2 != 0:
            continue
            # ls.append(l)
        else:
            ls.append(l)
    for l in ls:

        r = rate_two_way(L, l, eta_d, m, n)
        rates.append(r)
    idx = rates.index(max(rates))
    tot_spins = n * (m + 1)
    p_no_m = entanglement_link_prob_no_m(L, ls_init[idx], eta_d)
    print(rates[idx], ls_init[idx], tot_spins, tot_spins / (2 * (ls_init[idx] + 1)), rates[idx] * quotient_function(n * (m + 1), 2 * (ls_init[idx] + 1)))
    succ_prob = 1 - (1 - p_no_m) ** (tot_spins / (ls_init[idx] + 1))
    print(succ_prob, succ_prob ** (ls_init[idx] + 1))
    # print(p_no_m, p_no_m * tot_spins / (ls_init[idx] + 1))
    # print("quat: ",  quotient_function(n * (m + 1), 2 * (ls_init[idx] + 1)))
    # print(math.ceil(n * (m + 1) / (2 * (ls_init[idx] + 1))), n * (m + 1) / (2 * (ls_init[idx] + 1)))
    plt.plot(ls, rates)
plt.show()
