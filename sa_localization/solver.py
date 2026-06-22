import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import openjij as oj

from .config import Config


def run_sa(Q, cfg):
    """Run OpenJij Simulated Annealing on a QUBO problem.

    Args:
        Q   : dict { (int, int): float } — QUBO matrix from build_qubo()
        cfg : Config                     — SA hyperparameters

    Returns:
        best_sample : dict { int: int }  — best binary solution {pixel_index: 0_or_1}
        best_energy : float              — energy of the best sample
    """
    sampler = oj.SASampler()

    response = sampler.sample_qubo(
        Q,
        num_reads  = cfg.num_reads,
        num_sweeps = cfg.num_sweeps,
        beta_min   = cfg.beta_min,
        beta_max   = cfg.beta_max,
    )

    best_idx    = int(np.argmin(response.energies))
    best_sample = response.samples[best_idx]
    best_energy = float(response.energies[best_idx])

    return best_sample, best_energy
