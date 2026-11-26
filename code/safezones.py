import itertools
import random
import tarfile
from functools import lru_cache
from pathlib import Path

import networkx as nx

BRIDGE_PATTERNS = [
    ((0, 3), (4, 6)),
    ((0, 4), (5, 7)),
    ((1, 3), (5, 6)),
    ((0, 3), (4, 7)),
    ((0, 3), (5, 6)),
    ((1, 3), (4, 6)),
    ((1, 5), (4, 7)),
    ((1, 3), (4, 7)),
]

BROOM_PARAMS = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)]
LAYER_PARAMS = [
    (1, 2, 2),
    (1, 2, 2, 1),
    (1, 3, 2),
    (1, 2, 3, 1),
    (1, 3, 3),
    (1, 2, 2, 2),
]
BARBELL_PARAMS = [(3, 1), (3, 2), (4, 1), (4, 2)]
BRIDGEBLOCKS = [
    ("block-path",(3,1,3,1),((0,1),(1,2),(2,3))),
    ("block-star",(3,1,1,1),((0,1),(0,2),(0,3))),
    ("block-branched",(3,1,1,4,1),((0,1),(1,2),(1,3),(3,4))),
    ("block-asymmetric",(1,1,3,1,1,1,3),((0,1),(0,2),(2,3),(0,4),(4,5),(5,6))),
]


def edge(e):
    u, v = e
    return (u, v) if u < v else (v, u)


def edge_order(graph):
    return tuple(sorted(edge(e) for e in graph.edges()))


def broom(r, k):
    graph = nx.Graph()
    graph.add_edge(0, 1)
    start = 2
    for _ in range(r):
        clique = list(range(start, start + k))
        graph.add_edges_from(itertools.combinations(clique, 2))
        graph.add_edge(1, clique[0])
        start += k
    return graph


def layered(sizes):
    graph = nx.Graph()
    layers = []
    start = 0
    for size in sizes:
        layer = list(range(start, start + size))
        graph.add_nodes_from(layer)
        layers.append(layer)
        start += size
    for left, right in zip(layers, layers[1:]):
        graph.add_edges_from((u, v) for u in left for v in right)
    return graph


def bridge_tree(seed):
    graph = nx.Graph()
    for start in (0, 3, 6):
        graph.add_edges_from(itertools.combinations(range(start, start + 3), 2))
    for e in BRIDGE_PATTERNS[seed]:
        graph.add_edge(*e)
    return graph


def bridgeblocks(sizes,links):
    graph = nx.Graph()
    blocks = []
    for size in sizes:
        block = list(range(len(graph),len(graph)+size))
        graph.add_nodes_from(block)
        graph.add_edges_from(itertools.combinations(block,2))
        blocks.append(block)
    graph.add_edges_from((blocks[u][0],blocks[v][0]) for u,v in links)
    return graph

def treefamily(m):
    if not isinstance(m,int) or isinstance(m,bool) or m<2:
        raise ValueError("m must be an integer at least 2")
    s,h = 0,1
    path = list(range(2,m+3))
    c = m+3
    leaves = list(range(m+4,2*m+3))
    graph = nx.Graph([(s,h),(h,c)])
    nx.add_path(graph,[h]+path)
    graph.add_edges_from((c,leaf) for leaf in leaves)
    return graph,s,h,path,c,leaves

def treegraphs():
    graphs = [("path(7)",nx.path_graph(7)),("star(7)",nx.star_graph(6)),
              ("branched(6)",nx.Graph([(0,1),(0,2),(0,3),(3,4),(4,5)])),
              ("balanced(7)",nx.balanced_tree(2,2))]
    for m in (3,4,6):
        graph,s,h,path,c,leaves = treefamily(m)
        graph.graph.update(m=m,source=s)
        graphs.append((f"laura({m})",graph))
    return graphs

def families():
    return [
        ("bridge blocks", [(label,bridgeblocks(sizes,links)) for label,sizes,links in BRIDGEBLOCKS]),
        ("trees",treegraphs()),
        ("brooms", [(f"broom({r},{k})", broom(r, k)) for r, k in BROOM_PARAMS]),
        ("layered", [(f"layered{list(sizes)}", layered(sizes)) for sizes in LAYER_PARAMS]),
        ("barbells", [(f"barbell({a},{b})", nx.barbell_graph(a, b)) for a, b in BARBELL_PARAMS]),
    ]


def spread(nodes, surviving, infected):
    adjacency = {u: [] for u in nodes}
    for u, v in surviving:
        adjacency[u].append(v)
        adjacency[v].append(u)
    result = set(infected)
    for u in infected:
        result.update(adjacency[u])
    return frozenset(result)


def first_spread(graph, source):
    return frozenset({source, *graph.neighbors(source)})


def centralities(graph, strengths):
    bc = {edge(e): value for e, value in nx.edge_betweenness_centrality(
        graph, normalized=False
    ).items()}
    weighted = graph.copy()
    for e in edge_order(graph):
        weighted.edges[e]["distance"] = 1.0 / strengths[e]
    bcw = {edge(e): value for e, value in nx.edge_betweenness_centrality(
        weighted, normalized=False, weight="distance"
    ).items()}
    return bc, bcw


def strengths(graph, topology_index, realization):
    rng = random.Random(20260929 + 8 * topology_index + realization)
    return {e: rng.uniform(1, 10) for e in edge_order(graph)}


def choose_edge(policy, surviving, infected, values, bc, bcw):
    candidates = [
        e for e in surviving
        if not (e[0] in infected and e[1] in infected)
    ]
    if not candidates:
        return None
    if policy == "weak":
        return min(candidates, key=lambda e: (values[e], e))
    if policy == "strong":
        return min(candidates, key=lambda e: (-values[e], e))
    if policy == "bc":
        return min(candidates, key=lambda e: (-bc[e], e))
    if policy == "bcw":
        return min(candidates, key=lambda e: (-bcw[e], e))
    if policy == "source-aware":
        best = max(bc[e] for e in candidates)
        tied = [e for e in candidates if abs(bc[e] - best) <= 1e-12]
        threshold = [e for e in tied if (e[0] in infected) != (e[1] in infected)]
        return min(threshold or tied)
    if policy == "threshold":
        threshold = [e for e in candidates if (e[0] in infected) != (e[1] in infected)]
        if not threshold:
            return None
        return min(threshold, key=lambda e: (-bc[e], e))
    raise ValueError(policy)


def simulate(graph, source, policy, values=None, bc=None, bcw=None, trace=False):
    nodes = tuple(sorted(graph.nodes()))
    surviving = edge_order(graph)
    infected = first_spread(graph, source)
    values = values or {e: 1.0 for e in surviving}
    if bc is None or bcw is None:
        bc, bcw = centralities(graph, values)
    cuts = []
    states = [(frozenset({source}), None), (infected, None)]
    while True:
        if spread(nodes, surviving, infected) == infected:
            result = len(nodes) - len(infected)
            return (result, cuts, states) if trace else result
        cut = choose_edge(policy, surviving, infected, values, bc, bcw)
        if cut is not None:
            surviving = tuple(e for e in surviving if e != cut)
            cuts.append(cut)
        after = spread(nodes, surviving, infected)
        states.append((after, cut))
        if after == infected:
            result = len(nodes) - len(infected)
            return (result, cuts, states) if trace else result
        infected = after


def exact(graph, source, threshold_only=False, trace=False):
    nodes = tuple(sorted(graph.nodes()))
    initial_edges = edge_order(graph)
    initial = first_spread(graph, source)
    choice = {}

    @lru_cache(None)
    def best(infected_tuple, surviving):
        infected = frozenset(infected_tuple)
        if spread(nodes, surviving, infected) == infected:
            return len(nodes) - len(infected)
        actions = [None]
        for e in surviving:
            if e[0] in infected and e[1] in infected:
                continue
            if threshold_only and ((e[0] in infected) == (e[1] in infected)):
                continue
            actions.append(e)
        value = -1
        selected = None
        for action in actions:
            remaining = surviving if action is None else tuple(e for e in surviving if e != action)
            after = spread(nodes, remaining, infected)
            candidate = len(nodes) - len(infected) if after == infected else best(
                tuple(sorted(after)), remaining
            )
            if candidate > value:
                value = candidate
                selected = action
        choice[(tuple(sorted(infected)), surviving)] = selected
        return value

    value = best(tuple(sorted(initial)), initial_edges)
    if not trace:
        return value
    infected = initial
    surviving = initial_edges
    cuts = []
    while spread(nodes, surviving, infected) != infected:
        action = choice[(tuple(sorted(infected)), surviving)]
        if action is not None:
            surviving = tuple(e for e in surviving if e != action)
            cuts.append(action)
        after = spread(nodes, surviving, infected)
        if after == infected:
            break
        infected = after
    return value, cuts


def load_projected(path):
    graph = nx.Graph()
    with tarfile.open(path, "r:gz") as archive:
        member = next(
            item for item in archive.getmembers()
            if item.name.endswith(".txt") and "DATA-DESCRIPTION" not in item.name
        )
        with archive.extractfile(member) as source:
            for raw in source:
                u, v, value = raw.decode().split()
                graph.add_edge(int(u), int(v), strength=float(value))
    return graph


def real_counts(graph):
    values = {edge(e): graph.edges[e]["strength"] for e in graph.edges()}
    bc, bcw = centralities(graph, values)
    policies = ["weak", "strong", "bc", "bcw", "source-aware", "threshold"]
    out = {policy: {} for policy in policies}
    for source in graph.nodes():
        for policy in policies:
            saved = simulate(graph, source, policy, values, bc, bcw)
            out[policy][saved] = out[policy].get(saved, 0) + 1
    return out
