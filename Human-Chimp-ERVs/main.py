from scipy.stats import binom
from math import comb
import numpy as np
import random

# actual numbers: N = 10 million, a = 211, b = 208, k = 205
N = 216  # number of ERV insertion sites available - reduced to 216 for testing purposes
a = 211  # number of ERVs in the human genome
b = 208  # number of ERVs in the chimp genome
k = 205  # number of ERVs shared between humans and chimps


def prob_binomial(N, a, b, k):
    # calculate the probability of observing at least k shared ERVs between humans and chimps
    # based on the binomial distribution
    # X ~ Binomial(n, 1/N), find P(X >= k)
    n = a + b - k  # total number of ERVs combined, n = 214
    #return sum([comb(n, i) * (1/N) ** i * (1 - 1/N) ** (n - i) for i in range(k, n + 1)])
    #return sum([binom.pmf(i, n, 1 / N) for i in range(k, n + 1)])
    return 1 - binom.cdf(k - 1, n, 1 / N)

def prob_combinatorial(N, a, b, k):
    # calculate the probability of observing at least k shared ERVs between humans and chimps
    # based on combinatorial analysis
    return sum([comb(N, z) * comb(N - z, b - z) * comb(N - b, a - z) / (comb(N, a) * comb(N, b)) for z in range(k, min(a, b) + 1)])

def trial(N, a, b, k):
    # simulate a trial of inserting n ERVs into N sites
    x = random.sample(range(N), a)  # human ERVs
    y = random.sample(range(N), b)  # chimp ERVs
    return len(set(x).intersection(y)) >= k  # check if at least k ERVs are shared

NUM_TRIALS = 100_000

num_sucesses = 0

for _ in range(NUM_TRIALS):
    if trial(N, a, b, k):
        num_sucesses += 1
    if _ % 1000 == 0:
        print(f'Finished trial {_ + 1} out of {NUM_TRIALS}')

p_sim = num_sucesses / NUM_TRIALS

p_bin = prob_binomial(N, a, b, k)
p_comb = prob_combinatorial(N, a, b, k)

print(f'Simulation: {p_sim}')
print(f'Calculation using combinatorial method: {p_comb}')
print(f'Calculation using binomial method: {p_bin}')
