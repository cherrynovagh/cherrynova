"""
cvdistributions.py

Cryptographically secure random sampling from probability distributions.
All randomness comes from Python's secrets module.
"""
import math
import secrets


def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample."""
    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53)  # in [0, 1)
    return a + (b - a) * u


def exponentialdist(lam: float) -> float:
    """
    Cryptographically secure sample from an Exponential(lambda) distribution,
    using inverse transform sampling: x = -ln(1 - u) / lambda.
    """
    if lam <= 0:
        raise ValueError("lambda must be positive")
    u = uniform()
    return -math.log(1.0 - u) / lam


def poissiondist(lam: float) -> int:
    """
    Cryptographically secure sample from a Poisson(lambda) distribution,
    using discrete inverse transform sampling.
    Returns the smallest k such that F(k) >= u.
    """
    if lam <= 0:
        raise ValueError("lambda must be positive")
    u = uniform()
    k = 0
    p = math.exp(-lam)   # P(X = 0)
    cdf = p
    while u > cdf:
        k += 1
        p = p * lam / k  # P(X = k) from P(X = k-1)
        cdf += p
        if p == 0.0 and cdf < u:   # safety stop for rounding errors
            break
    return k
