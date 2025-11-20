import time
from enum import Enum
from typing import NamedTuple, Union

from pysat.card import CardEnc, EncType
from pysat.solvers import Glucose4, pysolvers

from ontolearn.learners.spell_kit.structures import (
    Signature,
    Structure,
    conceptname_ext,
    conceptnames,
    generate_all_trees,
    ind,
    restrict_to_neighborhood,
    rolenames,
    solution2sparql,
)

mode = Enum("mode", "exact neg_approx full_approx alc")

HC = dict[str, list[int]]
Simul = list[list[int]]
Pi = list[dict[int, int]]
Pr = dict[str, list[int]]


class Variables(NamedTuple):
    simul: Simul
    pi: Pi
    pr: Pr
    hc: HC


def compute_successors(sigma: Signature, A: Structure):
    succs: dict[str, dict[int, set[int]]] = {}
    for rn in rolenames(sigma):
        succs[rn] = {a: set() for a in ind(A)}

    for a in ind(A):
        for b, rn in A.rn_ext[a]:
            if rn in rolenames(sigma):
                succs[rn][a].add(b)
    return succs


var_counter: int = 1


def fresh_var():
    global var_counter
    r = var_counter
    var_counter = var_counter + 1
    return r


def constraint_conceptname(
    size: int,
    A: Structure,
    hc: HC,
    ind_tp_idx,
    anti_types,
    type_var: list[dict[int, int]],
    simul: Simul,
):

    for pInd in range(size):
        for a in ind(A):
            yield (-simul[pInd][a], type_var[pInd][ind_tp_idx[a]])

        for idx, tp in anti_types.items():
            for cn in tp:
                yield (-type_var[pInd][idx], -hc[cn][pInd])

            yield [type_var[pInd][idx]] + [hc[cn][pInd] for cn in tp]


def constraint_succ(
    size: int,
    A: Structure,
    sigma: Signature,
    pi: Pi,
    pr: Pr,
    simul: Simul,
    DR: list[list[list[int]]],
):
    succs = compute_successors(sigma, A)

    D2 = [[fresh_var() for a in ind(A)] for j in range(size)]

    for a in ind(A):
        for pInd2 in range(size):
            for pInd in range(pInd2):
                yield (DR[pInd][pInd2][a], -pi[pInd][pInd2], D2[pInd2][a])

    # for a in ind(A):
    #     for pInd2 in range(size):
    #         rns = []
    #         for (b, rn) in A[2][a]:
    #             if rn in rolenames(sigma):
    #                 rns.append(pr[rn][pInd2])
    #         yield [-D2[pInd2][a] ] + rns

    for a in ind(A):
        for pInd2 in range(size):
            for rn in rolenames(sigma):
                succ_sim = [simul[pInd2][b] for b in succs[rn][a]]
                yield [-D2[pInd2][a], -pr[rn][pInd2]] + succ_sim


def complement_type(tp, sigma: Signature):
    return tuple(cn for cn in conceptnames(sigma) if cn not in tp)


def compute_types(A: Structure, sigma: Signature):
    types: list[list[str]] = [[] for a in ind(A)]
    for cn in conceptnames(sigma):
        for a in conceptname_ext(A, cn):
            types[a].append(cn)

    fixed_types = {tuple(tp) for tp in types}
    fixed_types = list(fixed_types)
    fixed_types.sort(key="{}".format)
    fixed_types = list(map(frozenset, fixed_types))

    tp_map = {tp: idx for idx, tp in enumerate(fixed_types)}
    anti_types = {idx: complement_type(tp, sigma) for tp, idx in tp_map.items()}

    ind_tp_idx = [tp_map[frozenset(types[a])] for a in ind(A)]

    return ind_tp_idx, anti_types


# Returns a list of constraints that enforce the simulation conditions
def simulation_constraints(
    size: int, sigma: Signature, A: Structure, mapping: Variables
):
    simul = mapping[0]
    pi = mapping[1]
    pr = mapping[2]
    hc = mapping[3]

    # Defect vars
    DR = [[[fresh_var() for a in ind(A)] for j in range(size)] for i in range(size)]

    ind_tp_idx, anti_types = compute_types(A, sigma)

    type_var = [{idx: fresh_var() for idx in set(ind_tp_idx)} for i in range(size)]

    yield from constraint_conceptname(
        size, A, hc, ind_tp_idx, anti_types, type_var, simul
    )

    yield from constraint_succ(size, A, sigma, pi, pr, simul, DR)

    # positive Simulationsbedingung
    for pInd in range(size):
        for a in ind(A):
            # TODO: In some cases this can be a bottleneck, we could use the type-variables here
            cn_part = [-type_var[pInd][ind_tp_idx[a]]]
            rn_part = [DR[pInd][pInd2][a] for pInd2 in range(pInd + 1, size)]
            yield [simul[pInd][a]] + cn_part + rn_part

    # Same for roles
    for pInd in range(size):
        for pInd2 in range(pInd + 1, size):
            for a in ind(A):
                yield (-DR[pInd][pInd2][a], pi[pInd][pInd2])
                for b, rn in A.rn_ext[a]:
                    if rn in rolenames(sigma):
                        yield (-DR[pInd][pInd2][a], -pr[rn][pInd2], -simul[pInd2][b])

    for pInd in range(size):
        for a in ind(A):
            for pInd2 in range(pInd + 1, size):
                yield (-simul[pInd][a], -DR[pInd][pInd2][a])


def real_coverage(model, P: list[int], N: list[int], mapping: Variables) -> int:
    simul = mapping.simul
    cov = 0

    for a in P:
        if simul[0][a] in model:
            cov += 1
    for b in N:
        if -simul[0][b] in model:
            cov += 1

    return cov


def is_model(
    size: int, sigma: Signature, model: set[int], mapping: Variables, solver: Glucose4
):
    assums = []

    pi = mapping.pi
    pr = mapping.pr
    hc = mapping.hc

    for pInd in range(size):
        for cn in conceptnames(sigma):
            if hc[cn][pInd] in model:
                assums.append(hc[cn][pInd])
            else:
                assums.append(-hc[cn][pInd])
        for pInd2 in range(pInd + 1, size):
            for rn in rolenames(sigma):
                if pi[pInd][pInd2] in model and pr[rn][pInd2] in model:
                    assums.append(pi[pInd][pInd2])
                    assums.append(pr[rn][pInd2])

    return solver.solve(assumptions=assums)


def minimize_concept_assertions(
    size: int, sigma: Signature, solver: Glucose4, mapping: Variables, model: set[int]
) -> set[int]:
    best_model = model

    # Greedily reduce number of concept assertions and abuse sat solver as a fast query engine
    for i in range(size):
        for cn in conceptnames(sigma):
            if mapping.hc[cn][i] in best_model:
                test_model = set(best_model)
                test_model.remove(mapping.hc[cn][i])
                test_model.add(-mapping.hc[cn][i])
                if is_model(size, sigma, test_model, mapping, solver):
                    best_model = test_model
    return best_model


def model2fitting_query(
    size: int, sigma: Signature, mapping: Variables, model: set[int]
) -> Structure:
    pi = mapping.pi
    pr = mapping.pr
    hc = mapping.hc

    q = Structure(
        max_ind=size,
        cn_ext={cn: set() for cn in conceptnames(sigma)},
        rn_ext={a: set() for a in range(size)},
        indmap={},
        nsmap={},
    )

    for pInd in range(size):
        for cn in conceptnames(sigma):
            if hc[cn][pInd] in model:
                q.cn_ext[cn].add(pInd)
        for pInd2 in range(pInd + 1, size):
            for rn in rolenames(sigma):
                if pi[pInd][pInd2] in model and pr[rn][pInd2] in model:
                    q.rn_ext[pInd].add((pInd2, rn))
    return q


def create_variables(size: int, sigma: Signature, A: Structure) -> Variables:
    global var_counter
    var_counter = 1

    simul = [[fresh_var() for ind in ind(A)] for pInd in range(size)]

    # pi[i][j] is true if there is an edge between i and j
    pi = [
        {pInd2: fresh_var() for pInd2 in range(pInd1 + 1, size)}
        for pInd1 in range(size)
    ]

    # pr[rn][i] is true if product ind i has an incoming rn role
    pr = {rn: [fresh_var() for pInd in range(size)] for rn in rolenames(sigma)}

    # Conceptnames of product individuals
    hc = {cn: [fresh_var() for pInd in range(size)] for cn in conceptnames(sigma)}

    return Variables(simul, pi, pr, hc)


def tree_query_constraints(size: int, sigma: Signature, v: Variables):
    pi = v.pi
    pr = v.pr

    if size > 0 and size < 12:
        ktrees = list(generate_all_trees(size))

        treechoice = [fresh_var() for tree in ktrees]
        # At least one tree
        yield treechoice

        if size < 14:
            # At most one tree. Skip this if size gets too large, since it grows quadratically in the number of trees
            for j in range(0, len(ktrees)):
                for i in range(j):
                    yield (-treechoice[i], -treechoice[j])

        for t in range(len(ktrees)):
            tree = ktrees[t]
            for j in range(1, size):
                for i in range(j):
                    if tree[j - 1] == i:
                        yield (-treechoice[t], pi[i][j])
                    else:
                        yield (-treechoice[t], -pi[i][j])

    # Every pInd has at least a predecessor
    for j in range(1, size):
        yield [pi[i][j] for i in range(j)]

    # Every pInd has at most one predecessor
    for j in range(1, size):
        for i1 in range(j):
            for i2 in range(i1):
                yield (-pi[i1][j], -pi[i2][j])

    # Every pind has at least one incoming role
    for i in range(1, size):
        yield [pr[rn][i] for rn in rolenames(sigma)]

    # Every pInd has at most one incoming role
    rns = list(rolenames(sigma))
    for i in range(1, size):
        for r1 in range(len(rns)):
            for r2 in range(r1):
                yield (-pr[rns[r1]][i], -pr[rns[r2]][i])


def create_coverage_formula(
    P: list[int], N: list[int], coverage: int, mapping: Variables, all_pos: bool
) -> list[list[int]]:
    simul = mapping.simul

    global var_counter
    if coverage == len(P) + len(N):
        return [[simul[0][a]] for a in P] + [[-simul[0][b]] for b in N]
    elif all_pos:
        lits = [-simul[0][b] for b in N]

        bound = max(coverage - len(P), 1)
        enc = CardEnc.atleast(
            lits, bound=bound, top_id=var_counter, encoding=EncType.kmtotalizer
        )

        var_counter = enc.nv + 1

        return [[simul[0][a]] for a in P] + enc.clauses
    else:
        lits = [simul[0][a] for a in P] + [-simul[0][b] for b in N]

        enc = CardEnc.atleast(
            lits, bound=coverage, top_id=var_counter, encoding=EncType.kmtotalizer
        )

        var_counter = enc.nv + 1

        return enc.clauses


def non_empty_symbols(A: Structure) -> Signature:
    cns = [cn for cn in A.cn_ext.keys() if A.cn_ext[cn]]
    rns: set[str] = set()
    for a in ind(A):
        for _, rn in A.rn_ext[a]:
            rns.add(rn)
    rns2 = list(rns)

    cns.sort(key="{}".format)
    rns2.sort(key="{}".format)
    return (cns, rns2)


# Returns the (concept and role) symbols that are relevant given the positive
# examples
def determine_relevant_symbols(
    A: Structure, P: list[int], minP: int, dist: int
) -> Signature:

    (cns, rns) = non_empty_symbols(A)

    count = {cn: 0 for cn in cns}
    countr = {rn: 0 for rn in rns}

    for p in P:
        cns2: set[str] = set()
        rns2: set[str] = set()
        for cn in cns:
            if p in A.cn_ext[cn]:
                cns2.add(cn)

        dinds = {p}
        for r in range(dist):
            step: set[int] = set()
            for i1 in dinds:
                for i2, rn in A.rn_ext[i1]:
                    step.add(i2)
                    rns2.add(rn)
                    for cn in cns:
                        if i2 in A.cn_ext[cn]:
                            cns2.add(cn)
            dinds = step

        for cn in cns2:
            count[cn] += 1
        for rn in rns2:
            countr[rn] += 1

    cns = list(cn for (cn, c) in count.items() if c >= minP)
    rns = list(rn for (rn, c) in countr.items() if c >= minP)
    cns.sort(key="{}".format)
    rns.sort(key="{}".format)

    return (cns, rns)


def restrict_nb(
    k: int, A: Structure, P: list[int], N: list[int]
) -> tuple[Structure, list[int], list[int]]:
    (A2, mapping) = restrict_to_neighborhood(k - 1, A, P + N)
    P2 = [mapping[a] for a in P]
    N2 = [mapping[a] for a in N]
    return A2, P2, N2


# Constructs a formula to find a separating query of size and solves it
# Guaranted that we can reach min_coverage
def solve(
    size: int,
    A: Structure,
    P: list[int],
    N: list[int],
    coverage_lb: int,
    all_pos: bool,
    timeout: float = -1,
) -> Union[tuple[int, Structure], None]:

    time_start = time.process_time()
    A, P, N = restrict_nb(size, A, P, N)

    if all_pos:
        min_pos = len(P)
    else:
        # If we want to cover at least min_coverage examples, we have to cover at
        # least min_pos positive examples
        min_pos = max(coverage_lb - len(N), 1)
    # Use symbols that occur in distance k - 1 of at least min_pos positive example
    sigma = determine_relevant_symbols(A, P, min_pos, size - 1)

    g = Glucose4()
    mapping = create_variables(size, sigma, A)

    for c in tree_query_constraints(size, sigma, mapping):
        pysolvers.glucose41_add_cl(g.glucose, c)

    for c in simulation_constraints(size, sigma, A, mapping):
        pysolvers.glucose41_add_cl(g.glucose, c)

    dt = time.process_time() - time_start
    best_sol = None
    coverage_ub = len(P) + len(N)
    while coverage_lb <= coverage_ub and (dt < timeout or timeout < 0):

        for c in create_coverage_formula(P, N, coverage_lb, mapping, all_pos):
            pysolvers.glucose41_add_cl(g.glucose, c)

        satisfiable = g.solve()
        if not satisfiable:
            g.delete()
            return best_sol

        # print(g.accum_stats())
        model: set[int] = set(g.get_model())  # type: ignore
        coverage_lb = real_coverage(model, P, N, mapping)

        if True:
            # Required for minimization
            for c in create_coverage_formula(P, N, coverage_lb, mapping, all_pos):
                pysolvers.glucose41_add_cl(g.glucose, c)

            model = minimize_concept_assertions(size, sigma, g, mapping, model)
        best_q = model2fitting_query(size, sigma, mapping, model)
        best_sol = (coverage_lb, best_q)
        print(solution2sparql(best_q))

        print(
            "== Coverage: {}/{} == Accuracy: {}".format(
                coverage_lb, coverage_ub, coverage_lb / coverage_ub
            )
        )
        coverage_lb = coverage_lb + 1
        dt = time.process_time() - time_start

    g.delete()
    return best_sol


# Search for a small separating query by incrementally increasing the size
def solve_incr(
    A: Structure,
    P: list[int],
    N: list[int],
    m: mode,
    timeout: float = -1,
    starting_size: int = 1,
    max_size: int = 19,
) -> tuple[int, Structure]:
    time_start = time.process_time()
    i = starting_size
    best_coverage = len(P)
    best_q = Structure(max_ind=1, cn_ext={}, rn_ext={0: set()}, indmap={}, nsmap={})
    dt = time.process_time() - time_start
    while (
        best_coverage < len(P) + len(N)
        and i <= max_size
        and (dt < timeout or timeout == -1)
    ):
        print("== Searching for a fitting query of size {}".format(i))
        if m == mode.exact:
            sol = solve(i, A, P, N, len(P) + len(N), True, timeout - dt)
        elif m == mode.neg_approx:
            sol = solve(i, A, P, N, best_coverage + 1, True, timeout - dt)
        else:
            sol = solve(i, A, P, N, best_coverage + 1, False, timeout - dt)
        if sol is not None:
            best_coverage, best_q = sol
        i += 1
        dt = time.process_time() - time_start

    print(
        "== Best query found with coverage {}/{}".format(best_coverage, len(P) + len(N))
    )
    print(solution2sparql(best_q))
    return (best_coverage, best_q)
