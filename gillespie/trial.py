from itertools import combinations

def species_combinations(J, S):
    for c in combinations(range(J+S-1),S-1):
        yield tuple( b - a - 1 for a, b in zip((-1,) + c, c + (J + S -1,)))

#pecies_combinations(200,99)
