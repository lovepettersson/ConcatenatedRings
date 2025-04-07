
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



