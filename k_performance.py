""" Code written by Matthew Barnett """
import numpy as np

A_param = 482.01
B_param = 2085.43
alpha_param = 0.3478
beta_param = 0.3658

# Scaling law from Hoffmann et al.
def KL_divergence(N, D, A=A_param, B=B_param, alpha=alpha_param, beta=beta_param):
    return A/(N**alpha) + B/(D**beta)


# K-performance formula
def k_performance(kl_divergence, slowdown=1, confidence=0.9):
    delta_0 = confidence/(1 - confidence)
    delta_1 = (1-confidence)/confidence
    D_1 = ((1 - delta_0)/(delta_1 - delta_0))
    D_0 = delta_1 *D_1
    return slowdown*(D_1*np.log2(D_0/D_1) + (1 - D_1)*np.log2((1-D_0)/(1-D_1)))/(-kl_divergence)


# Finds the optimal settings for parameters (N) and data (D) given a compute budget (C)
def optimal_N_D(C, A=A_param, B=B_param, alpha=alpha_param, beta=beta_param):
    G = (alpha * A)/(beta * B) ** (1 / (alpha + beta))
    a = beta/(alpha + beta)
    b = alpha/(alpha + beta)
    return (G*(C/6)**a, G**(-1)*(C/6)**b)


# Solves a linear equation to determine how much compute would be required to match k-performance = k
def computation_for_k_performance(k, slowdown=1):
    """Returns log(flops)"""
    A = np.random.normal(A_param, 25)
    B = np.random.normal(B_param, 25)
    alpha = np.random.normal(alpha_param, 0.05)
    beta = np.random.normal(beta_param, 0.05)

    compute_points = [10, 30]
    log10_k_perf_points = []

    for compute in compute_points:
        N_opt, D_opt = optimal_N_D(10**compute, A=A, B=B, alpha=alpha, beta=beta)
        kl_divergence = KL_divergence(N_opt, D_opt, A=A, B=B, alpha=alpha, beta=beta)
        log10_k_perf = np.log10(k_performance(kl_divergence, slowdown))
        log10_k_perf_points.append(log10_k_perf)

    m = np.diff(compute_points)/np.diff(log10_k_perf_points)
    b = compute_points[0] - log10_k_perf_points[0]*m

    compute = m*np.log10(k) + b
    return compute
