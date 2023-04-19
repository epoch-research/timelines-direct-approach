""" Code written by Matthew Barnett """
import numpy as np


# Scaling law from Hoffmann et al.
def KL_divergence(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    return A/(N**alpha) + B/(D**beta)


# K-performance formula
def k_performance(kl_divergence, slowdown=1, confidence=0.9):
    delta_0 = confidence/(1 - confidence)
    delta_1 = (1-confidence)/confidence
    D_1 = ((1 - delta_0)/(delta_1 - delta_0))
    D_0 = delta_1 *D_1
    return slowdown*(D_1*np.log2(D_0/D_1) + (1 - D_1)*np.log2((1-D_0)/(1-D_1)))/(-kl_divergence)


# Finds the optimal settings for parameters (N) and data (D) given a compute budget (C)
def optimal_N_D(C, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    G = (alpha * A)/(beta * B)
    a = beta/(alpha + beta)
    b = alpha/(alpha + beta)
    return (G*(C/6)**a, G**(-1)*(C/6)**b)


# Solves a linear equation to determine how much compute would be required to match k-performance = k
def computation_for_k_performance(k, slowdown=1):
    """Returns log(flops)"""
    A = np.random.normal(406.4, 25)
    B = np.random.normal(410.7, 25)
    alpha = np.random.normal(0.34, 0.05)
    beta = np.random.normal(0.28, 0.05)
    compute1 = 10

    N_opt, D_opt = optimal_N_D(10**compute1)
    kl_divergence = KL_divergence(N_opt, D_opt)
    log10_k_perf1 = np.log10(k_performance(kl_divergence, slowdown))
    compute2 = 30
    N_opt, D_opt = optimal_N_D(10**compute2, A=A, B=B, alpha=alpha, beta=beta)
    kl_divergence = KL_divergence(N_opt, D_opt, A=A, B=B, alpha=alpha, beta=beta)
    log10_k_perf2 = np.log10(k_performance(kl_divergence, slowdown))
    m = 20.0/(log10_k_perf2 - log10_k_perf1)
    b = 10 - log10_k_perf1*m

    return m*np.log10(k) + b