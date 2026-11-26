# Core and application regression suite
from pathlib import Path
from unittest.mock import patch
from fractions import Fraction
from itertools import combinations
from random import Random
from time import perf_counter

import networkx as nx

from experiments import synthetic,metrics,POLICIES,SEEDSTART
from safezones import broom, centralities, exact, families, layered, simulate, strengths
from safezones import edge, edge_order, first_spread
from safezones import BRIDGEBLOCKS,bridge_tree,treefamily
from thresholdsearch import regression as testthresholdsearch
from cyclesearch import regression as testcyclesearch
from unicyclelemmas import regression as testunicyclelemmas


def test_simulation():
    graph = nx.path_graph(3)
    assert exact(graph, 0) == 1
    values = {e: 1.0 for e in [(0, 1), (1, 2)]}
    bc, bcw = centralities(graph, values)
    assert simulate(graph, 0, "threshold", values, bc, bcw) == 1


def test_strategies():
    graph = broom(3, 2)
    values = strengths(graph, 8, 0)
    bc, bcw = centralities(graph, values)
    assert simulate(graph, 5, "source-aware", values, bc, bcw) == 6
    assert simulate(graph, 5, "threshold", values, bc, bcw) == 6


def test_exact_optimization():
    graph = layered((1, 2, 2, 1))
    value, cuts = exact(graph, 0, trace=True)
    threshold = exact(graph, 0, threshold_only=True)
    assert value == 1
    assert cuts == [(3, 5), (4, 5)]
    assert threshold == 0


def test_families():
    groups = families()
    assert [len(graphs) for _, graphs in groups] == [4, 7, 5, 6, 4]
    assert [sum(graph.number_of_nodes() for _, graph in graphs) for _, graphs in groups] == [35, 62, 42, 38, 34]
    assert sum(sum(graph.number_of_nodes() for _, graph in graphs) for _, graphs in groups) == 211


def test_delay_family():
    graph = layered((1, 2, 2, 1))
    values = strengths(graph, 14, 0)
    bc, bcw = centralities(graph, values)
    exact_value, exact_cuts = exact(graph, 0, trace=True)
    threshold_value, threshold_cuts, _ = simulate(
        graph, 0, "threshold", values, bc, bcw, trace=True
    )
    assert exact_value == 1
    assert exact_cuts == [(3, 5), (4, 5)]
    assert threshold_value == 0
    assert threshold_cuts == [(1, 3), (3, 5)]


def test_synthetic_regression():
    result = synthetic()
    assert result["topologies"] == 26
    assert result["topology_source_pairs"] == 211
    assert result["weighted_records"] == 1688
    assert result["zero_pairs"] == 36
    assert result["nontrivial_weighted_records"] == 1400
    assert list(result["zero_counts"].values())==[1,1,1,32,1]
    assert [result["summary"][family]["denominator"] for family in result["pair_counts"]]==[272,488,328,48,264]
    expected = {
        "overall": [23.5, 27.142857142857, 50.285714285714, 50.357142857143, 67.428571428571, 94.285714285714],
        "bridge blocks": [24.264705882353, 24.264705882353, 61.764705882353, 60.661764705882, 61.764705882353, 97.058823529412],
        "trees": [26.229508196721, 30.737704918033, 47.540983606557, 47.540983606557, 54.098360655738, 95.081967213115],
        "brooms": [26.524390243902, 32.012195121951, 46.341463414634, 46.341463414634, 97.560975609756, 100.0],
        "layered": [27.083333333333, 39.583333333333, 16.666666666667, 25.0, 16.666666666667, 0.0],
        "barbells": [13.257575757576, 15.151515151515, 54.545454545455, 54.545454545455, 69.696969696970, 100.0],
    }
    policies = ["weak", "strong", "bc", "bcw", "source-aware", "threshold"]
    for family, target in expected.items():
        for policy, value in zip(policies, target):
            assert abs(result["summary"][family][policy]["optimal"] - value) < 1e-9
    checktheory(result)


def replaytree(graph,s,cuts):
    sequence = iter(cuts)
    with patch("safezones.choose_edge",side_effect=lambda *args:next(sequence,None)):
        return simulate(graph,s,"scripted",trace=True)

def testtree(ms=range(2,9)):
    rows = []
    for m in ms:
        graph,s,h,path,c,leaves = treefamily(m)
        original = graph.copy()
        assert len(graph)==2*m+3 and nx.is_tree(graph)
        assert first_spread(graph,s)=={s,h}
        pathedge,staredge = edge((h,path[0])),edge((h,c))
        assert edge((path[0],h))==pathedge and edge((c,h))==staredge
        values = dict.fromkeys(edge_order(graph),1.0)
        bc,bcw = centralities(graph,values)
        bcpath,bcstar = 2*bc[pathedge],2*bc[staredge]
        assert bcpath==2*(m+1)*(m+2) and bcstar==2*m*(m+3)
        assert bcpath-bcstar==4
        with patch("safezones.centralities",wraps=centralities) as counted:
            tbc,cuts,states = simulate(graph,s,"threshold",trace=True)
        assert counted.call_count==1 and counted.call_args.args[0] is graph
        assert states[1][0]=={s,h} and states[1][1] is None
        assert len(cuts)==2 and cuts[0]==pathedge
        assert cuts[1] in [edge((c,leaf)) for leaf in leaves]
        for i,cut in enumerate(cuts,1):
            assert (cut[0] in states[i][0])!=(cut[1] in states[i][0])
        assert states[2][0]=={s,h,c}
        assert tbc==m+2 and set(graph)-states[-1][0]==set(path)|set(cuts[1])-{c}
        explicit = [staredge,edge((path[0],path[1]))]
        saved,removed,history = replaytree(graph,s,explicit)
        assert saved==2*m and removed==explicit
        assert history[1][0]=={s,h} and history[2][0]=={s,h,path[0]}
        assert set(graph)-history[-1][0]==set(leaves)|{c}|set(path[1:])
        optimum,optimalcuts = exact(graph,s,trace=True)
        assert optimum==2*m
        saved,removed,history = replaytree(graph,s,optimalcuts)
        assert saved==optimum and removed==optimalcuts
        assert nx.utils.graphs_equal(graph,original)
        rows.append((m,bcpath,bcstar,tbc,optimum,tbc/optimum))
    return rows

def checkunknown(graph,cuts,p):
    assert set(p)==set(graph) and sum(p.values())==1
    assert all(isinstance(value,Fraction) and value>=0 for value in p.values())
    assert all(graph.has_edge(*cut) for cut in cuts)
    remaining = graph.copy()
    remaining.remove_edges_from(cuts)
    n = len(remaining)
    groups = list(nx.connected_components(remaining))
    component = n-sum(sum(p[v] for v in group)*len(group) for group in groups)
    direct = Fraction(0)
    for source in remaining:
        infected = {source}
        while True:
            after = infected|{v for u in infected for v in remaining.neighbors(u)}
            if after==infected:
                break
            infected = after
        direct += p[source]*(n-len(infected))
    pairs = [(s,t) for s in remaining for t in remaining
             if s!=t and not nx.has_path(remaining,s,t)]
    cutbc = sum((p[s] for s,t in pairs),Fraction(0))
    assert direct==component==cutbc,(cuts,p,direct,component,cutbc)
    assert len(pairs)==n*n-sum(len(group)**2 for group in groups)
    if all(value==Fraction(1,n) for value in p.values()):
        assert component==Fraction(len(pairs),n)
    return component

def testunknown():
    graphs = [nx.path_graph(4),nx.cycle_graph(5),nx.complete_graph(4),
              broom(2,2),layered((1,2,2,1)),bridge_tree(0),
              nx.barbell_graph(3,1),nx.empty_graph(1),
              nx.disjoint_union(nx.path_graph(3),nx.path_graph(2))]
    generator = Random(20261006)
    for n in range(3,8):
        for probability in (0.25,0.5,0.75):
            for rep in range(2):
                graph = nx.gnp_random_graph(n,probability,seed=generator)
                while not nx.is_connected(graph):
                    graph = nx.gnp_random_graph(n,probability,seed=generator)
                graphs.append(graph)
    checks,bridgechecks = 0,0
    for graph in graphs:
        original = graph.copy()
        nodes = sorted(graph)
        n = len(nodes)
        edges = edge_order(graph)
        uniform = dict.fromkeys(nodes,Fraction(1,n))
        increasing = {v:Fraction(i+1,n*(n+1)//2) for i,v in enumerate(nodes)}
        concentrated = {v:Fraction(int(v==nodes[0])) for v in nodes}
        cutsets = {cuts for size in range(3) for cuts in combinations(edges,size)}|{edges}
        bridges = {edge(e) for e in nx.bridges(graph)} if nx.is_connected(graph) else set()
        bc,bcw = centralities(graph,dict.fromkeys(edges,1.0))
        for cuts in sorted(cutsets):
            for p in (uniform,increasing,concentrated):
                value = checkunknown(graph,cuts,p)
                checks += 1
                if len(cuts)==1 and cuts[0] in bridges:
                    remaining = graph.copy()
                    remaining.remove_edges_from(cuts)
                    left,right = list(nx.connected_components(remaining))
                    bridgevalue = sum(p[v] for v in left)*len(right)+sum(p[v] for v in right)*len(left)
                    assert value==bridgevalue
                    if p==uniform:
                        assert value==Fraction(2*len(left)*len(right),n)
                        assert abs(2*bc[cuts[0]]-2*len(left)*len(right))<1e-10
                    bridgechecks += 1
        assert nx.utils.graphs_equal(graph,original)
    print(f"Static identity: 9 fixed + 30 random graphs; {checks} rational checks; {bridgechecks} bridge checks")

def testbridgeswitch():
    graph = nx.path_graph(4)
    edges = edge_order(graph)
    distributions = [dict.fromkeys(graph,Fraction(1,4)),
                     {0:Fraction(7,10),1:Fraction(1,10),2:Fraction(1,10),3:Fraction(1,10)}]
    expected = [(Fraction(3,2),Fraction(2),Fraction(3,2)),
                (Fraction(12,5),Fraction(2),Fraction(6,5))]
    for p,target,winner in zip(distributions,expected,((1,2),(0,1))):
        values = tuple(checkunknown(graph,[cut],p) for cut in edges)
        assert values==target
        assert [cut for cut,value in zip(edges,values) if value==max(values)]==[winner]
    cycle = nx.cycle_graph(4)
    assert checkunknown(cycle,[(0,1)],distributions[0])==0
    bc,bcw = centralities(cycle,dict.fromkeys(edge_order(cycle),1.0))
    assert bc[(0,1)]>0

def bridgesafe(tree,infected,cuts):
    remaining = tree.copy()
    remaining.remove_edges_from(cuts)
    return set().union(*(group for group in nx.connected_components(remaining) if not group & infected))

def bridgebrute(tree,weights,infected,b):
    bestvalue,bestcuts = -1,[]
    for size in range(min(b,tree.number_of_edges())+1):
        for cuts in combinations(edge_order(tree),size):
            safe = bridgesafe(tree,infected,cuts)
            value = sum(weights[v] for v in safe)
            if value>bestvalue:
                bestvalue,bestcuts = value,list(cuts)
    return bestvalue,bestcuts

def bridgedp(tree,weights,infected,b,root=None):
    b = min(b,tree.number_of_edges())
    root = min(tree) if root is None else root
    parent,order = {root:None},[root]
    for v in order:
        for u in tree[v]:
            if u!=parent[v]:
                parent[u] = v
                order.append(u)
    table = {}
    for v in reversed(order):
        current = [[-1,-1] for k in range(b+1)]
        current[0] = [0,weights[v] if v not in infected else -1]
        for u in tree[v]:
            if parent.get(u)!=v:
                continue
            merged = [[-1,-1] for k in range(b+1)]
            for k in range(b+1):
                for x in (0,1):
                    if current[k][x]<0:
                        continue
                    for j in range(b-k+1):
                        for y in (0,1):
                            cost = k+j+(x!=y)
                            if cost<=b and table[u][j][y]>=0:
                                merged[cost][x] = max(merged[cost][x],current[k][x]+table[u][j][y])
            current = merged
        table[v] = current
    return max(max(row) for row in table[root])

def checklabels(tree,weights,infected):
    edges = edge_order(tree)
    cutchecks,labelchecks = 0,0
    for size in range(len(edges)+1):
        for cuts in combinations(edges,size):
            safe = bridgesafe(tree,infected,cuts)
            boundary = [e for e in edges if (e[0] in safe)!=(e[1] in safe)]
            assert set(boundary)<=set(cuts)
            recovered = bridgesafe(tree,infected,boundary)
            assert recovered==safe
            assert sum(weights[v] for v in recovered)==sum(weights[v] for v in safe)
            cutchecks += 1
    available = sorted(set(tree)-infected)
    for size in range(len(available)+1):
        for vertices in combinations(available,size):
            labeled = set(vertices)
            boundary = [e for e in edges if (e[0] in labeled)!=(e[1] in labeled)]
            actual = bridgesafe(tree,infected,boundary)
            assert labeled<=actual
            assert sum(weights[v] for v in labeled)<=sum(weights[v] for v in actual)
            closed = [e for e in edges if (e[0] in actual)!=(e[1] in actual)]
            assert set(closed)<=set(boundary) and bridgesafe(tree,infected,closed)==actual
            labelchecks += 1
    return cutchecks,labelchecks

def testbridgeexact():
    generator = Random(20261006)
    counts = {}
    instances,comparisons,cutchecks,labelchecks = 0,0,0,0
    for n in range(1,9):
        trees = [nx.empty_graph(1)] if n==1 else nx.nonisomorphic_trees(n)
        counts[n] = 0
        for tree in trees:
            tree = nx.convert_node_labels_to_integers(tree,ordering="sorted")
            nodes = sorted(tree)
            assignments = [dict.fromkeys(nodes,1),{v:v+1 for v in nodes},
                           {v:1+(3*v+1)%5 for v in nodes}]
            if n<=6:
                infections = {frozenset(v for v in nodes if mask>>v & 1) for mask in range(1,1<<n)}
            else:
                leaves = [v for v in nodes if tree.degree(v)==1]
                internal = [v for v in nodes if tree.degree(v)>1]
                infections = {frozenset({v}) for v in nodes}|{frozenset(e) for e in tree.edges}
                infections.update((frozenset(nodes),frozenset(nodes[:3]),frozenset(leaves[:2]),
                                   frozenset((leaves[0],internal[0]))))
                while len(infections)<32:
                    mask = generator.randrange(1,1<<n)
                    infections.add(frozenset(v for v in nodes if mask>>v & 1))
            for infected in sorted(infections,key=lambda group:(len(group),tuple(sorted(group)))):
                cuts,labels = checklabels(tree,assignments[1],infected)
                cutchecks += cuts
                labelchecks += labels
                instances += 1
                for weights in assignments:
                    for b in range(min(4,n-1)+1):
                        value,solution = bridgebrute(tree,weights,infected,b)
                        actual = bridgedp(tree,weights,infected,b)
                        assert actual==value,(list(tree.edges),weights,infected,b,value,actual)
                        assert len(solution)<=b and sum(weights[v] for v in bridgesafe(tree,infected,solution))==value
                        comparisons += 1
            counts[n] += 1
    assert counts=={1:1,2:1,3:1,4:2,5:3,6:6,7:11,8:23}
    print(f"Bridge trees by n: {counts}")
    print(f"Bridge exhaustive: {instances} tree/infection cases; 3 weight assignments; {comparisons} DP/brute comparisons")
    print(f"Label equivalence: {cutchecks} removal sets; {labelchecks} feasible labelings")

def testbridgesingle():
    generator = Random(20261007)
    roots,checks = 0,0
    for index in range(100):
        n = 2+index%25
        tree = nx.from_prufer_sequence([generator.randrange(n) for i in range(n-2)])
        weights = {v:generator.randint(1,9) for v in tree}
        for root in generator.sample(sorted(tree),min(3,n)):
            remaining = tree.copy()
            remaining.remove_node(root)
            branches = sorted((sum(weights[v] for v in group) for group in nx.connected_components(remaining)),reverse=True)
            for b in sorted(set(range(min(4,n-1)+1))|{n-1,n+1}):
                assert bridgedp(tree,weights,{root},b,root=n-1)==sum(branches[:b])
                checks += 1
            roots += 1
    print(f"Single infected block: 100 random weighted trees; {roots} infected roots; {checks} branch-formula checks")

def testbridgecases():
    shapes = [nx.path_graph(5),nx.star_graph(5),nx.balanced_tree(2,2),
              nx.Graph([(0,1),(1,2),(1,3),(3,4),(3,5)])]
    checks = 0
    for tree in shapes:
        nodes = sorted(tree)
        weights = {v:1+v%4 for v in tree}
        leaves = [v for v in nodes if tree.degree(v)==1]
        internal = [v for v in nodes if tree.degree(v)>1]
        infections = [set(leaves[:2]),{leaves[0],internal[0]},set(edge_order(tree)[0]),
                      set(nodes[:3]),set(nodes),set()]
        for infected in infections:
            for b in (0,1,2,len(tree)-1,len(tree)+1):
                value,cuts = bridgebrute(tree,weights,infected,b)
                assert bridgedp(tree,weights,infected,b)==value
                assert bridgedp(tree,weights,infected,b,root=nodes[-1])==value
                if not infected or b>=len(tree)-1:
                    assert value==sum(weights[v] for v in set(tree)-infected)
                elif b==0 or infected==set(tree):
                    assert value==0
                checks += 1
    path = nx.path_graph(3)
    assert bridgesafe(path,{0},[(0,1),(1,2)])=={1,2}
    assert bridgesafe(path,{0},[(0,1)])=={1,2}
    checklabels(path,dict.fromkeys(path,1),set())
    graph = nx.disjoint_union_all([nx.complete_graph(size) for size in (3,4,5)])
    graph.add_edges_from([(2,3),(6,7)])
    bridges = sorted(nx.bridges(graph))
    remaining = graph.copy()
    remaining.remove_edges_from(bridges)
    blocks = sorted(nx.connected_components(remaining),key=min)
    membership = {v:i for i,block in enumerate(blocks) for v in block}
    tree = nx.Graph((membership[u],membership[v]) for u,v in bridges)
    weights = {i:len(block) for i,block in enumerate(blocks)}
    assert nx.is_tree(tree) and list(weights.values())==[3,4,5]
    for sources in ({0},{0,1},{0,1,7}):
        infected = {membership[v] for v in sources}
        assert infected==({0,2} if 7 in sources else {0})
        for size in range(3):
            for cuts in combinations(bridges,size):
                mapped = [(membership[u],membership[v]) for u,v in cuts]
                safe = bridgesafe(tree,infected,mapped)
                assert bridgesafe(graph,sources,cuts)==set().union(*(blocks[v] for v in safe))
        for b in range(3):
            value = max(len(bridgesafe(graph,sources,cuts)) for size in range(b+1) for cuts in combinations(bridges,size))
            assert bridgedp(tree,weights,infected,b)==bridgebrute(tree,weights,infected,b)[0]==value
    print(f"Bridge edge cases: {checks} cases, each with two DP roots; block projection checks passed")

def testbridgescaling():
    generator = Random(20261008)
    print("Bridge DP timing: n b seconds")
    for n in (50,200,800):
        tree = nx.from_prufer_sequence([generator.randrange(n) for i in range(n-2)])
        weights = {v:generator.randint(1,9) for v in tree}
        infected = set(generator.sample(sorted(tree),3))
        previous = 0
        for b in (2,4,8,16):
            start = perf_counter()
            value = bridgedp(tree,weights,infected,b)
            elapsed = perf_counter()-start
            assert previous<=value<=sum(weights[v] for v in set(tree)-infected)
            previous = value
            print(f"{n:4} {b:2} {elapsed:.6f}")

def testtopologies():
    groups = families()
    graphs = [graph for family,items in groups for label,graph in items]
    assert all(nx.is_connected(graph) for graph in graphs)
    assert all(not nx.is_isomorphic(left,right) for left,right in combinations(graphs,2))
    quotients = []
    for (label,graph),(name,sizes,links) in zip(groups[0][1],BRIDGEBLOCKS):
        assert label==name
        bridges = list(nx.bridges(graph))
        remaining = graph.copy()
        remaining.remove_edges_from(bridges)
        blocks = sorted(nx.connected_components(remaining),key=min)
        assert [len(block) for block in blocks]==list(sizes)
        for block in blocks:
            assert len(block)==1 or len(block)>=3
            assert remaining.subgraph(block).number_of_edges()==len(block)*(len(block)-1)//2
        membership = {v:i for i,block in enumerate(blocks) for v in block}
        quotient = nx.Graph()
        quotient.add_nodes_from((i,{"size":size}) for i,size in enumerate(sizes))
        quotient.add_edges_from((membership[u],membership[v]) for u,v in bridges)
        assert nx.is_tree(quotient) and edge_order(quotient)==tuple(sorted(links))
        quotients.append(quotient)
    for left,right in combinations(quotients,2):
        assert not nx.is_isomorphic(left,right)
        assert not nx.is_isomorphic(left,right,node_match=lambda a,b:a["size"]==b["size"])
    asymmetric = quotients[-1]
    assert max(nx.single_source_shortest_path_length(asymmetric,0).values())==3
    assert sum(1 for mapping in nx.algorithms.isomorphism.GraphMatcher(asymmetric,asymmetric).isomorphisms_iter())==1
    assert all(nx.is_tree(graph) for label,graph in groups[1][1])
    print("Topology audit: all 26 graphs pairwise nonisomorphic; four distinct quotient shapes and weighted quotients")

def testmetrics():
    records = [dict(exact=1,**dict.fromkeys(POLICIES,1)),
               dict(exact=4,**dict.fromkeys(POLICIES,1)),
               dict(exact=0,**dict.fromkeys(POLICIES,0))]
    result = metrics(records)
    assert result["denominator"]==2 and result["zero_records"]==1
    for policy in POLICIES:
        assert result[policy]=={"optimal":50.0,"mean":0.625,"minimum":0.25}
        assert metrics(records[2:])[policy]=={"optimal":None,"mean":None,"minimum":None}

def checktheory(result):
    records = result["records"]
    keys = {(row["label"],row["source"],row["realization"]) for row in records}
    assert len(keys)==len(records)
    for topology in result["topology_manifest"]:
        rows = [row for row in records if row["label"]==topology["label"]]
        assert {row["source"] for row in rows}==set(range(topology["nodes"]))
        assert len(rows)==8*topology["nodes"]
    indices = [row["strength_index"] for row in result["topology_manifest"]]
    assert indices==list(range(23,34))+list(range(8,23))
    assert SEEDSTART=={"bridge blocks":23,"trees":27,"brooms":8,"layered":13,"barbells":19}
    assert all(row["weight_seed"]==20260929+8*row["strength_index"]+row["realization"] for row in records)
    trees = [row for row in records if row["tree"] and row["exact"]>0]
    assert len(trees)==680
    assert all(Fraction(row["threshold"],row["exact"])>=Fraction(1,2) for row in trees)
    assert all(pair["thresholdopt"]==pair["exact"] for pair in result["pairs"] if pair["tree"])
    assert [row["m"] for row in result["laura"]]==[3,4,6]
    for m in (3,4,6):
        rows = [row for row in records if row["label"]==f"laura({m})" and row["source"]==0]
        assert len(rows)==8
        for row in rows:
            assert row["threshold"]==m+2 and row["exact"]==2*m
            assert Fraction(row["threshold"],row["exact"])==Fraction(m+2,2*m)
    failures = {("layered[1, 2, 2, 1]",0),("layered[1, 2, 2, 1]",5),
                ("layered[1, 2, 3, 1]",6),("layered[1, 2, 2, 2]",0),
                ("layered[1, 2, 2, 2]",5),("layered[1, 2, 2, 2]",6)}
    assert {(row["label"],row["source"]) for row in result["layered_failures"]}==failures
    for row in result["layered_failures"]:
        assert row["exact"]==1 and row["thresholdopt"]==row["threshold"]==0
    rows = [row for row in records if row["family"]=="layered" and row["exact"]>0]
    assert len(rows)==48 and all(row["threshold"]==0 for row in rows)
    print("Theory checks: 680 tree records, 24 designated Laura records, and 48 layered-failure records passed")

TESTS = [
    ("simulation", test_simulation),
    ("strategies", test_strategies),
    ("exact optimization", test_exact_optimization),
    ("families", test_families),
    ("topology distinctness",testtopologies),
    ("protection-ratio aggregation",testmetrics),
    ("delay family", test_delay_family),
    ("synthetic regression", test_synthetic_regression),
    ("tree theorem", testtree),
    ("independent threshold search",testthresholdsearch),
    ("sparse-cycle exact oracle",testcyclesearch),
    ("unicyclic structural lemmas",testunicyclelemmas),
    ("unknown-source identity", testunknown),
    ("bridge switch", testbridgeswitch),
    ("multiple-infected bridge DP", testbridgeexact),
    ("single-infected bridge formula", testbridgesingle),
    ("bridge edge cases", testbridgecases),
    ("bridge DP scaling", testbridgescaling),
]


def main():
    passed = 0
    for name, test in TESTS:
        test()
        passed += 1
        print("PASS", name)
    print(f"Passed {passed}, Failed 0")

def testair():
    import csv

    import hashlib

    import json

    import random

    from fractions import Fraction

    from itertools import combinations

    import networkx as nx

    from air import root,sourcehash,buildgraph,readgraph,edges,objectives,contraction,partitions,numerator,optimize,efficiency,rankings,cohort,washington

    from hospital import score,components

    def decorate(graph,seed=0):
        graph = graph.copy()
        rng = random.Random(seed)
        for v in sorted(graph): graph.nodes[v]['inbound'] = rng.randrange(0,31)
        graph.nodes[min(graph)]['inbound'] += 1
        for e in edges(graph): graph.edges[e].update(passengers=rng.randrange(1,41),departures=1,seats=50)
        return graph

    def mustfail(action,kind=ValueError):
        try: action()
        except kind: return
        raise AssertionError('Expected '+kind.__name__)

    def parsing():
        def row(u,v,p=10,dep=2,seats=20,year='2025',service='F',us='WA',vs='WA'):
            return dict(ORIGIN=u,DEST=v,ORIGIN_CITY_NAME=u+' City',DEST_CITY_NAME=v+' City',
                        ORIGIN_STATE_ABR=us,DEST_STATE_ABR=vs,YEAR=year,CLASS=service,MONTH='1',
                        PASSENGERS=str(p),DEPARTURES_PERFORMED=str(dep),SEATS=str(seats))
        rows = [row('AAA','BBB'),row('BBB','AAA',p=7),row('BBB','CCC',p=3),row('AAA','AAA'),
                row('AAA','DDD',p=0),row('AAA','EEE',service='L'),row('AAA','FFF',vs='AK'),
                row('AAA','GGG',vs='HI'),row('AAA','HHH',vs='PR'),row('AAA','III',year='2024'),row('XXX','YYY')]
        graph,audit = buildgraph(rows)
        assert set(graph)=={'AAA','BBB','CCC'} and edges(graph)==(('AAA','BBB'),('BBB','CCC'))
        assert graph.edges['AAA','BBB']==dict(passengers=17,departures=4,seats=40)
        assert {v:(graph.nodes[v]['inbound'],graph.nodes[v]['outbound']) for v in graph}=={'AAA':(7,10),'BBB':(10,10),'CCC':(3,0)}
        assert audit['retained']==4 and audit['componentcounts']==[3,2]
        assert buildgraph(reversed(rows))[0].nodes==graph.nodes
        mustfail(lambda:buildgraph([row('AAA','BBB',p='0.5')]))
        mustfail(lambda:buildgraph([row('AAA','BBB',p=0)]))
        assert buildgraph([row('AAA','BBB',us='DC')])[0].nodes['AAA']['state']=='DC'
        print('PASS BTS parser: service/year/traffic/geography filters, directions, totals, self-loops, giant component and airport metadata')

    def brute(graph,b):
        found = []
        for size in range(min(b,graph.number_of_edges())+1):
            for cuts in combinations(edges(graph),size):
                surviving = graph.copy()
                surviving.remove_edges_from(cuts)
                uu,mixed,pp = 0,0,0
                for s in graph:
                    outside = set(graph)-nx.node_connected_component(surviving,s)
                    mass = sum(graph.nodes[v]['inbound'] for v in outside)
                    uu += len(outside)
                    mixed += mass
                    pp += graph.nodes[s]['inbound']*mass
                found.append(dict(cuts=cuts,uu=uu,mixed=mixed,pp=pp,cost=sum(graph.edges[e]['passengers'] for e in cuts)))
        return found

    def exhaustive():
        graphs = [decorate(g,i) for i,g in enumerate(nx.graph_atlas_g()) if 2<=len(g)<=5 and nx.is_connected(g)]
        rng = random.Random(20261029)
        for n in range(6,9):
            for j in range(6):
                graph = nx.gnp_random_graph(n,.15+.04*j,seed=rng.randrange(10000000))
                graph.add_edges_from((i,i+1) for i in range(n-1))
                while graph.number_of_edges()>11:
                    extra = [e for e in edges(graph) if e[1]!=e[0]+1]
                    graph.remove_edge(*extra[-1])
                graphs.append(decorate(graph,n*10+j))
        graphs += [decorate(nx.complete_bipartite_graph(2,7)),decorate(nx.star_graph(6)),decorate(nx.path_graph(7))]
        comparisons,ratios,penalties = 0,0,0
        for graph in graphs:
            before = nx.node_link_data(graph)
            b = 4 if graph.number_of_edges()>11 else min(6,graph.number_of_edges())
            catalog = partitions(graph,b)
            rows = brute(graph,b)
            for mode in ('uu','mixed','pp'):
                actual,stats = optimize(catalog,mode)
                for budget,state in actual.items():
                    expected = min((r for r in rows if len(r['cuts'])<=budget),key=lambda r:(-r[mode],len(r['cuts']),r['cuts']))
                    assert (numerator(catalog,mode,state),state[4])==(expected[mode],expected['cuts']),(edges(graph),mode,budget,state,expected)
                    comparisons += 1
                best = max((Fraction(r[mode],r['cost']) for r in rows if r['cost']),default=Fraction(0))
                if best:
                    state,iterations = efficiency(catalog,mode)
                    assert Fraction(numerator(catalog,mode,state),state[3])==best
                    ratios += 1
                for penalty in (Fraction(1,3),Fraction(7,2)):
                    actual,stats = optimize(catalog,mode,penalty)
                    expected = min(rows,key=lambda r:(-(r[mode]-penalty*r['cost']),len(r['cuts']),r['cuts']))
                    state = actual[b]
                    assert (numerator(catalog,mode,state)-penalty*state[3],state[4])==(expected[mode]-penalty*expected['cost'],expected['cuts'])
                    penalties += 1
            assert nx.node_link_data(graph)==before
        graph = decorate(nx.complete_bipartite_graph(2,7))
        atoms,quotient,core = contraction(graph,6)
        assert any(set(atom)=={0,1} for atom in atoms) and not graph.has_edge(0,1)
        singleton = decorate(nx.empty_graph(1))
        catalog = partitions(singleton,6)
        assert numerator(catalog,'uu',optimize(catalog,'uu')[0][6])==0
        for b in (-1,True,1.2): mustfail(lambda:partitions(graph,b))
        for graph in (nx.Graph(),nx.path_graph(3).to_directed(),nx.MultiGraph(nx.path_graph(3)),nx.empty_graph(2)):
            mustfail(lambda:partitions(graph,2))
        graph = decorate(nx.path_graph(4))
        mustfail(lambda:partitions(graph,2,limit=0),RuntimeError)
        catalog = partitions(graph,2)
        mustfail(lambda:optimize(catalog,'uu',limit=0),RuntimeError)
        print(f'PASS exhaustive solver: {len(graphs)} graphs; {comparisons} original-edge brute-force optima, {ratios} exact efficiency optima, {penalties} penalized optima; non-induced atoms and explicit search limits')

    def conventions():
        graph = nx.Graph()
        graph.add_edges_from([(0,1,dict(passengers=3)),(1,2,dict(passengers=3)),(0,2,dict(passengers=1))])
        before = nx.node_link_data(graph)
        orders = rankings(graph)
        assert orders['bc']==((0,1),(0,2),(1,2))
        assert orders['bcw']==((0,1),(1,2),(0,2))
        assert orders['weak']==((0,2),(0,1),(1,2))
        assert orders['strong']==((0,1),(1,2),(0,2))
        assert orders['bcperpassenger']==((0,2),(0,1),(1,2))
        assert nx.node_link_data(graph)==before
        reversedgraph = nx.Graph()
        reversedgraph.add_nodes_from(reversed(list(graph)))
        reversedgraph.add_edges_from((v,u,dict(d)) for u,v,d in reversed(list(graph.edges(data=True))))
        assert rankings(reversedgraph)==orders
        print('PASS fixed unweighted BC, inverse-passenger BCw, weak/strong/cost rankings, canonical ties and immutability')

    def application():
        path = root/'data/raw/t100_domestic_segment_2025.zip'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sourcehash
        graph,audit = readgraph(path)
        before = nx.node_link_data(graph)
        assert (len(graph),graph.number_of_edges())==(443,7160)
        assert audit['months']==list(range(1,13)) and audit['years']=={'2025':458793}
        assert (audit['positiveF'],audit['retained'])==(342850,309398)
        assert audit['componentcounts']==[443,3,2,2]
        assert sum(graph.nodes[v]['inbound'] for v in graph)==sum(graph.nodes[v]['outbound'] for v in graph)==sum(graph.edges[e]['passengers'] for e in edges(graph))==797071268
        catalog = partitions(graph,6)
        stats = catalog['stats']
        assert (stats['atoms'],stats['atomsizes'],stats['branches'],stats['localsubsets'],stats['localpartitions'])==(187,{1:185,8:1,250:1},146,7096,432)
        nine = cohort(graph,washington)
        assert nine['connected'] and (nine['internalroutes'],nine['internalpassengers'],nine['boundarysize'],nine['boundarypassengers'],nine['boundarydepartures'])==(30,27447,3,183,37)
        thirteen = tuple(sorted(set(washington)|{'BFI','ESD','FRD','RNT'}))
        other = cohort(graph,thirteen)
        assert other['connected'] and (other['internalroutes'],other['internalpassengers'],other['boundarysize'],other['boundarypassengers'])==(37,61608,6,3713)
        uniform,stats = optimize(catalog,'uu')
        assert [numerator(catalog,'uu',uniform[b]) for b in range(1,7)]==[884,1766,7812,8678,10344,11204]
        assert uniform[3][4]==(('BFI','LPS'),('ESD','LPS'),('FRD','LPS'))
        assert sorted(map(len,components(graph,uniform[6][4])))==[1,12,430]
        assert numerator(catalog,'uu',uniform[6])>2*13*430
        rows = list(csv.DictReader((root/'data/results/airresults.csv').open()))
        assert len(rows)==24 and len({(r['prior'],r['target'],r['budget']) for r in rows})==24
        orders = rankings(graph)
        optimized = {mode:optimize(catalog,mode)[0] for mode in ('uu','mixed','pp')}
        for row in rows:
            p,a = objectives(graph,row['prior'],row['target'])
            cuts = tuple(tuple(e) for e in json.loads(row['optimalcut']))
            groups = components(graph,cuts)
            sourcegroups = {v:set(c) for c in groups for v in c}
            direct = sum(p[s]*sum(a[v] for v in graph if v not in sourcegroups[s]) for s in graph)
            assert direct==score(graph,cuts,p,a)==Fraction(row['exact'])
            mode = 'uu' if (row['prior'],row['target'])==('uniform','airports') else 'pp' if (row['prior'],row['target'])==('inbound','inbound') else 'mixed'
            assert optimized[mode][int(row['budget'])][4]==cuts
            assert int(row['disruptedpassengers'])==sum(graph.edges[e]['passengers'] for e in cuts)
            assert int(row['disrupteddepartures'])==sum(graph.edges[e]['departures'] for e in cuts)
            detail = json.loads(row['components'])
            assert {tuple(c['airports']) for c in detail}==set(groups)
            assert all(c['size']==len(c['airports']) and c['inbound']==sum(graph.nodes[v]['inbound'] for v in c['airports']) for c in detail)
            for method,order in orders.items():
                value = score(graph,order[:int(row['budget'])],p,a)
                assert value==Fraction(row[method+'score'])<=direct
                assert Fraction(row[method+'ratio'])==value/direct
                if method!='bcperpassenger': assert value==0
        for mode in ('uu','mixed','pp'):
            state,iterations = efficiency(catalog,mode)
            expected = uniform[3][4] if mode=='pp' else (('IL4','ORD'),)
            assert state[4]==expected
        assert nx.node_link_data(graph)==before
        print('PASS BTS regression: exact network/contraction, both Washington groups, all 24 saved optima/direct expectations/baselines, secondary optima and input immutability')

    parsing()
    exhaustive()
    conventions()
    application()
    print('Air application tests: Passed 4, Failed 0')

def testapplicationsweeps():
    import csv

    import json

    from fractions import Fraction

    from itertools import combinations

    from pathlib import Path

    import networkx as nx

    import air

    import commuting

    import applicationsweeps as sweep

    from commuting import colorado

    from hospital import edges,components,score

    def mustfail(action):
        try: action()
        except ValueError: return
        raise AssertionError('Expected ValueError')

    def filtering():
        grid = sweep.kappas()
        assert len(grid)==len(set(grid))==37 and (grid[0],grid[-1])==(Fraction(3,4),Fraction(3,2))
        assert all(Fraction(i,100) in grid for i in range(98,111))
        assert all(Fraction(i,10000) in grid for i in range(13000,13201,25))
        assert all(k in grid for k in (Fraction(95,96),Fraction(13,12),Fraction(46,35)))
        graph = nx.path_graph(['A','B','C','D'])
        for v in graph: graph.nodes[v]['inbound'] = 100
        for e,p in zip(edges(graph),(7,8,9)): graph.edges[e]['passengers'] = p
        before = nx.node_link_data(graph)
        filtered = sweep.filterair(graph,8)
        assert edges(filtered)==(('B','C'),('C','D'))
        assert set(sweep.giant(filtered))=={'B','C','D'}
        assert sum(sweep.giant(filtered).nodes[v]['inbound'] for v in sweep.giant(filtered))==300
        assert nx.node_link_data(graph)==before
        for bad in (0,-1,True,1.5): mustfail(lambda:sweep.filterair(graph,bad))
        nodes = {v:dict(resident=10) for v in ('A','B','C')}
        pairs = {('A','B'):(('A','B',26,24),('B','A',5,100)),('B','C'):(('B','C',46,35),)}
        lower = commuting.buildgraph(nodes,pairs,Fraction(13,12)-Fraction(1,100000),False)
        equal = commuting.buildgraph(nodes,pairs,Fraction(13,12),False)
        assert lower.edges['A','B']['flow']==31 and not equal.has_edge('A','B')
        assert equal.has_edge('B','C') and not commuting.buildgraph(nodes,pairs,Fraction(46,35),False).has_edge('B','C')
        assert dict(lower.nodes)==dict(equal.nodes)
        assert sweep.signature('air',graph,'one')==sweep.signature('air',graph.copy(),'one')
        assert sweep.signature('air',graph,'one')!=sweep.signature('air',graph,'two')
        changed = graph.copy()
        changed.nodes['A']['inbound'] += 1
        assert sweep.signature('air',graph,'one')!=sweep.signature('air',changed,'one')
        print('PASS parameter grids, strict rational MOE thresholds, inclusive passenger thresholds, fixed masses and cache keys')

    def adapters():
        graphs = [nx.path_graph(5),nx.cycle_graph(4),nx.complete_graph(4)]
        attached = nx.complete_graph(4)
        attached.add_edges_from([(3,4),(4,5)])
        graphs.append(attached)
        checks = 0
        for graph in graphs:
            for v in graph: graph.nodes[v].update(resident=v+1,inbound=v+1)
            for i,e in enumerate(edges(graph)): graph.edges[e].update(flow=i+1,passengers=i+1)
            before = nx.node_link_data(graph)
            for kind in ('commuting','air'):
                weight = commuting.weights if kind=='commuting' else air.objectives
                payload = sweep.analyze(kind,graph,3,100000)
                for row in payload['rows']:
                    p,a = weight(graph,row['prior'],row['target'])
                    found = []
                    for size in range(row['budget']+1):
                        for cuts in combinations(edges(graph),size):
                            surviving = graph.copy()
                            surviving.remove_edges_from(cuts)
                            direct = sum(p[s]*sum(a[v] for v in graph if not nx.has_path(surviving,s,v)) for s in graph)
                            found.append((direct,cuts))
                    value,cuts = min(found,key=lambda pair:(-pair[0],len(pair[1]),pair[1]))
                    assert row['status']=='exact' and Fraction(row['exact'])==value and row['optimalcut']==cuts
                    assert row['componentsizes']==list(map(len,row['optimalpartition']))
                    for method in sweep.methods:
                        actual = score(graph,row[method+'cuts'],p,a)
                        assert Fraction(row[method+'score'])==actual<=value
                        assert row[method+'ratio']==(str(actual/value) if value else '')
                    checks += 1
                incomplete = sweep.analyze(kind,graph,3,0)
                assert all(r['status']=='incomplete' and r['exact']=='' and r['searcherror'] for r in incomplete['rows'])
            assert nx.node_link_data(graph)==before
        print(f'PASS reused solver adapters: {checks} exact comparisons to independent source-averaged brute force, canonical cuts and honest search-limit failures')

    def candidates():
        core = tuple('core'+str(i) for i in range(8))
        graph = nx.complete_graph(core)
        graph.add_edges_from(zip(colorado,colorado[1:]))
        graph.add_edge(core[0],colorado[0])
        for v in graph: graph.nodes[v]['resident'] = 1
        for e in edges(graph): graph.edges[e]['flow'] = 1
        payload = sweep.decorate(sweep.analyze('commuting',graph,2,100000),'commuting',Fraction(1),graph,graph)
        for row in payload['rows']:
            assert row['candidateconnected'] and row['candidatepresent'] and row['candidateboundarysize']==1
            assert row['candidateoptimal']==(row['budget']==1)
            assert row['candidateinrepresentative']==(row['budget']==1)
        disconnected = graph.copy()
        disconnected.remove_edge(colorado[-2],colorado[-1])
        info = sweep.candidate(disconnected,sweep.giant(disconnected),colorado)
        assert not info['candidateconnected'] and not info['candidatepresent'] and not info['candidaterepresentable']
        assert info['candidatemissing']==[colorado[-1]]
        print('PASS whole-cohort optimum versus larger-budget partitions, disconnected cohorts and vertices outside the giant')

    def outputs():
        results,checks = {},0
        for kind,filename,parameters,B in [('commuting','commutingsweep.csv',sweep.kappas(),4),('air','airsweep.csv',range(1,61),6)]:
            rows = list(csv.DictReader((sweep.root/'data/results'/filename).open()))
            assert len(rows)==len(parameters)*4*B
            assert {Fraction(r['threshold']) for r in rows}==set(parameters)
            assert len({(r['threshold'],r['prior'],r['target'],r['budget']) for r in rows})==len(rows)
            previous = {}
            for row in rows:
                assert row['status']=='exact' and not row['searcherror']
                b,value = int(row['budget']),Fraction(row['exact'])
                cuts = json.loads(row['optimalcut'])
                groups = json.loads(row['optimalpartition'])
                sizes,masses = json.loads(row['componentsizes']),json.loads(row['componentmasses'])
                assert len(cuts)<=b and len(cuts)==len(set(map(tuple,cuts)))
                assert sizes==list(map(len,groups)) and sum(sizes)==int(row['n'])
                assert sum(masses)==int(row['totalmass'])
                assert len({v for c in groups for v in c})==int(row['n'])
                assert json.loads(row['protectedifsourceingiant'])==sorted(v for c in groups[1:] for v in c)
                N,W = int(row['n']),int(row['totalmass'])
                a = sizes if row['target'] in ('counties','airports') else masses
                p = sizes if row['prior']=='uniform' else masses
                expected = sum(a)-sum(Fraction(w,sum(p))*t for w,t in zip(p,a))
                assert value==expected
                key = row['threshold'],row['prior'],row['target']
                assert value>=previous.get(key,Fraction(0))
                previous[key] = value
                feasible = row['candidatepresent']=='True' and row['candidateconnected']=='True' and int(row['candidateboundarysize'])<=b
                assert (row['candidatefeasible']=='True')==feasible
                assert (row['candidateoptimal']=='True')==(feasible and Fraction(row['candidatescore'])==value)
                for method in sweep.methods:
                    result = Fraction(row[method+'score'])
                    assert 0<=result<=value and len(json.loads(row[method+'cuts']))<=b
                    assert row[method+'ratio']==(str(result/value) if value else '')
                checks += 1
                if row['threshold']=='1':
                    restored = dict(row,budget=b,optimalcut=cuts)
                    for method in sweep.methods: restored[method+'cuts'] = json.loads(row[method+'cuts'])
                    results.setdefault((kind,'1'),dict(rows=[]))['rows'].append(restored)
        sweep.anchors(results)
        print(f'PASS {checks} saved objective/budget rows: all parameters, exact partition identity, budgets, ratios, monotonicity and all 40 original-result anchors')

    def replay():
        nodes,pairs,audit = commuting.readflows(commuting.workbookrows(sweep.root/'data/raw/table1.xlsx'))
        base,audit = air.readgraph(sweep.root/'data/raw/t100_domestic_segment_2025.zip')
        checks,graphs = 0,0
        for kind,filename,selected in [('commuting','commutingsweep.csv',{Fraction(3,4),Fraction(95,96),Fraction(1),Fraction(13,12),Fraction(46,35),Fraction(3,2)}),
                                      ('air','airsweep.csv',{Fraction(t) for t in (1,8,40,41,60)})]:
            rows = list(csv.DictReader((sweep.root/'data/results'/filename).open()))
            for parameter in sorted(selected):
                graphs += 1
                full = commuting.buildgraph(nodes,pairs,parameter,False) if kind=='commuting' else sweep.filterair(base,int(parameter))
                graph = sweep.giant(full)
                vertices = colorado if kind=='commuting' else air.washington
                info = sweep.candidate(full,graph,vertices)
                for row in (r for r in rows if Fraction(r['threshold'])==parameter):
                    assert (int(row['n']),int(row['m']))==(len(graph),graph.number_of_edges())
                    assert row['candidateconnected']==str(info['candidateconnected'])
                    assert row['candidatepresent']==str(info['candidatepresent'])
                    assert json.loads(row['candidateboundary'])==list(map(list,info['candidateboundary']))
                    weight = commuting.weights if kind=='commuting' else air.objectives
                    p,a = weight(graph,row['prior'],row['target'])
                    for prefix in ('optimal',)+sweep.methods:
                        cuts = tuple(map(tuple,json.loads(row['optimalcut' if prefix=='optimal' else prefix+'cuts'])))
                        assert all(graph.has_edge(*e) for e in cuts)
                        value = score(graph,cuts,p,a)
                        assert value==Fraction(row['exact' if prefix=='optimal' else prefix+'score'])
                        checks += 1
                    if info['candidatepresent']:
                        assert score(graph,info['candidateboundary'],p,a)==Fraction(row['candidatescore'])
                    else: assert row['candidatescore']==''
        print(f'PASS {checks} raw-graph exact/baseline cut replays across {graphs} anchor/transition/end-point graphs')

    filtering()
    adapters()
    candidates()
    outputs()
    replay()
    print('Application sweep tests: Passed 5, Failed 0')

def testcommuting():
    import csv

    import hashlib

    import json

    import random

    from fractions import Fraction

    from itertools import combinations

    from pathlib import Path

    import networkx as nx

    from commuting import root,sourcehash,workbookrows,readflows,buildgraph,weights,catalog,exact,rankings,boundary,bigbend,bridge,objectives,sensitivity

    from hospital import edges,components,score

    def mustfail(action):
        try: action()
        except (ValueError,RuntimeError): return
        raise AssertionError('Expected validation failure')

    def parsing():
        def row(u,v,flow,moe):
            return dict(A=u[:2],B=u[2:],C=u[:2],D=u,E=v[:2],F=v[2:],G=v[:2],H=v,I=str(flow),J=str(moe))
        rows = [row('48001','48001',100,10),row('48001','48003',12,10),row('48003','48001',5,100),
                row('48003','48003',200,15),row('48003','48005',10,10),row('48005','48003',8,9),
                row('48005','48005',50,10),row('48001','02001',7,2),row('15001','48003',4,1)]
        foreign = row('48001','99999',3,2)
        foreign.update(E='',F='',G='Other workplace outside of the U.S.',H='')
        rows.append(foreign)
        nodes,pairs,audit = readflows(rows)
        assert set(nodes)=={'48001','48003','48005'}
        assert nodes['48001']['resident']==122 and nodes['48003']['resident']==215 and nodes['48005']['resident']==58
        assert nodes['48003']['inbound']==224
        graph = buildgraph(nodes,pairs,1,False)
        assert edges(graph)==(('48001','48003'),) and graph.edges['48001','48003']['flow']==17
        assert len(buildgraph(nodes,pairs))==2
        relaxed = buildgraph(nodes,pairs,Fraction(3,4),False)
        assert relaxed.edges['48003','48005']['flow']==18
        assert len(edges(buildgraph(nodes,pairs,Fraction(6,5),False)))==0
        assert {v:graph.nodes[v]['resident'] for v in graph}=={v:relaxed.nodes[v]['resident'] for v in relaxed}
        p,a = weights(graph,'commuters','commuters')
        assert p['48001']==Fraction(122,395) and a['48003']==215
        assert score(graph,(),p,a)>0
        assert readflows(reversed(rows))==readflows(rows)
        mustfail(lambda:readflows(rows+rows[:1]))
        mustfail(lambda:buildgraph(nodes,pairs,-1))
        mustfail(lambda:weights(graph,'invalid','counties'))
        print('PASS parser: own-direction strict MOE, both flows, FIPS/geography, self/foreign worker masses, isolates and fixed masses')

    def exhaustive():
        graphs = [g for g in nx.graph_atlas_g() if 1<=len(g)<=5 and nx.is_connected(g)]
        rng = random.Random(20261101)
        for n in range(6,9):
            for j in range(4):
                graph = nx.path_graph(n)
                extra = list(nx.non_edges(graph))
                rng.shuffle(extra)
                graph.add_edges_from(extra[:j])
                graphs.append(graph)
        graphs += [nx.star_graph(6),nx.complete_bipartite_graph(2,5)]
        comparisons = 0
        for graph in graphs:
            for v in graph: graph.nodes[v]['resident'] = rng.randrange(1,20)
            for e in edges(graph): graph.edges[e]['flow'] = rng.randrange(1,50)
            before = nx.node_link_data(graph)
            B = min(4,graph.number_of_edges())
            data = catalog(graph,B)
            actual,stats = exact(data)
            found = {}
            for size in range(B+1):
                for cuts in combinations(edges(graph),size):
                    groups = components(graph,cuts)
                    labels = {v:i for i,c in enumerate(groups) for v in c}
                    minimal = tuple(e for e in edges(graph) if labels[e[0]]!=labels[e[1]])
                    N,W = len(graph),sum(graph.nodes[v]['resident'] for v in graph)
                    masses = [sum(graph.nodes[v]['resident'] for v in c) for c in groups]
                    found[minimal] = dict(uu=N*N-sum(len(c)**2 for c in groups),
                                         mixed=N*W-sum(len(c)*w for c,w in zip(groups,masses)),pp=W*W-sum(w*w for w in masses))
            for mode in actual:
                for b,solution in actual[mode].items():
                    value = max(v[mode] for cut,v in found.items() if len(cut)<=b)
                    cuts = {cut for cut,v in found.items() if len(cut)<=b and v[mode]==value}
                    assert (solution['numerator'],set(solution['cuts']))==(value,cuts),(edges(graph),mode,b)
                    comparisons += 1
            for prior,target,mode in objectives:
                p,a = weights(graph,prior,target)
                cut = actual[mode][B]['cuts'][0]
                H = graph.copy()
                H.remove_edges_from(cut)
                direct = sum(p[s]*sum(a[v] for v in graph if not nx.has_path(H,s,v)) for s in graph)
                assert direct==score(graph,cut,p,a)
            assert nx.node_link_data(graph)==before
        mustfail(lambda:exact(data,limit=0))
        print(f'PASS exact partition solver: {len(graphs)} connected graphs, {comparisons} mode/budget comparisons, all co-optima and direct source averages')

    def ordering():
        graph = nx.Graph()
        graph.add_edge('0','1',flow=3)
        graph.add_edge('1','2',flow=3)
        graph.add_edge('0','2',flow=1)
        before = nx.node_link_data(graph)
        orders,bc,bcw = rankings(graph)
        assert orders['bc']==(('0','1'),('0','2'),('1','2'))
        assert bcw=={('0','1'):4,('1','2'):4,('0','2'):0}
        assert orders['bcw']==(('0','1'),('1','2'),('0','2'))
        assert orders['weak'][0]==('0','2') and orders['strong']==orders['bcw']
        assert nx.node_link_data(graph)==before
        graph.edges['0','1']['flow'] = 0
        mustfail(lambda:rankings(graph))
        print('PASS fixed rankings: ordered-pair BC, inverse-flow BCw, canonical ties and unchanged edge attributes')

    def application():
        path = root/'data/raw/table1.xlsx'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sourcehash
        nodes,pairs,audit = readflows(workbookrows(path))
        graph = buildgraph(nodes,pairs)
        assert (len(nodes),len(graph),graph.number_of_edges())==(3109,3105,24338)
        assert sum(graph.nodes[v]['resident'] for v in graph)==152611850
        assert [graph.nodes[v]['resident'] for v in bigbend]==[4531,714,2794]
        assert boundary(graph,bigbend)==(bridge,)
        assert [graph.edges[e]['flow'] for e in edges(graph.subgraph(bigbend))]==[281,221,223]
        assert graph.edges[bridge]['flow']==217
        assert len(list(nx.bridges(graph)))==29
        data = catalog(graph)
        solutions,stats = exact(data)
        assert data['stats']['atomsizes']=={1:288,2817:1}
        assert (data['stats']['localsubsets'],stats['globalpartitions'])==(10628,56296)
        with (root/'data/results/commutingresults.csv').open() as stream: records = list(csv.DictReader(stream))
        assert len(records)==16
        for row in records:
            prior,target,b = row['prior'],row['target'],int(row['budget'])
            mode = next(mode for p,a,mode in objectives if (p,a)==(prior,target))
            p,a = weights(graph,prior,target)
            cuts = tuple(map(tuple,json.loads(row['optimalcut'])))
            assert cuts==solutions[mode][b]['cuts'][0]
            assert Fraction(row['exact'])==score(graph,cuts,p,a)
            assert int(row['optimalpartitions'])==len(solutions[mode][b]['cuts'])==1
            assert json.loads(row['alloptimalboundaries'])==json.loads(json.dumps(solutions[mode][b]['cuts']))
            assert (bridge,) in solutions[mode][1]['cuts']
            for name in ('bc','bcw','weak','strong'):
                chosen = tuple(map(tuple,json.loads(row[name+'cuts'])))
                value = score(graph,chosen,p,a)
                assert value==Fraction(row[name+'score'])<=Fraction(row['exact'])
                assert value==0
                assert Fraction(row[name+'ratio'])==value/Fraction(row['exact'])
                if name=='bc' and b==1: assert len(components(graph,chosen))==1
        assert solutions['uu'][1]['numerator']==18612
        assert [solutions['uu'][b]['numerator'] for b in range(1,5)]==[18612,37188,55764,68140]
        first = json.loads(records[0]['removed'])[0]
        assert first['ranks']==dict(bc=179,bcw=1069,weak=13917,strong=10440)
        assert first['bc']==first['bcw']==18612
        print('PASS raw application: 3105/24338, Big Bend regression, 56296 partitions, all 16 exact optima and saved baseline scores')
        return nodes,pairs

    def thresholds(nodes,pairs):
        expected = [(41595,3,3107,0,True,False,7),(24338,5,3105,29,True,True,1),
                    (17712,18,3091,58,True,True,1),(13898,50,3052,124,True,True,1),(11643,93,2982,196,False,False,1)]
        actual = sensitivity(nodes,pairs)
        for row,want in zip(actual,expected):
            assert row['n']==3109
            assert (row['m'],row['components'],row['giant'],row['bridges'],row['bigbendconnected'],row['bridgeseparated'],len(row['boundary']))==want
            for values in row['scores'].values():
                if values is None: continue
                for item in values.values(): assert Fraction(item['exact'])-Fraction(item['baseline'])==Fraction(item['increment'])
        assert actual[-1]['scores']['giant'] is None
        print('PASS five reliability thresholds: all counties/components, bridges, exact Big Bend connectivity and baseline-adjusted scores')

    parsing()
    exhaustive()
    ordering()
    nodes,pairs = application()
    thresholds(nodes,pairs)
    print('Commuting tests: Passed 5, Failed 0')

def testcommutinginterpret():
    import records as savedreports

    import csv

    import json

    from fractions import Fraction

    from pathlib import Path

    import networkx as nx

    from commuting import workbookrows,readflows,buildgraph,weights,objectives,adjacency

    from commuting import root,colorado,bigbend,edwardsreal,coloradocuts,adjacencypath,readresults,structure,evaluate,replayoriginal,sweep,records

    from hospital import edges,components

    def fixtures():
        graph = nx.cycle_graph(4)
        for v in graph: graph.nodes[v].update(name=str(v),state='X',resident=v+1)
        for u,v in graph.edges: graph.edges[u,v].update(flow=10,directions=((u,v,10,5),))
        before = graph.copy()
        data = structure(graph,(0,1),graph)
        assert data['count']==2 and data['commuters']==3 and data['internalflow']==10
        assert data['boundarysize']==2 and data['disruption']==20
        assert all(not d['bridge'] for d in data['boundaryedges'])
        values = evaluate(graph,data)
        cuts = tuple(d['edge'] for d in data['boundaryedges'])
        H = graph.copy()
        H.remove_edges_from(cuts)
        for prior,target,mode in objectives:
            p,a = weights(graph,prior,target)
            direct = sum(p[s]*sum(a[v] for v in graph if not nx.has_path(H,s,v)) for s in graph)
            assert Fraction(values[prior+'/'+target]['exact'])==direct
            assert Fraction(values[prior+'/'+target]['disruptionperexpected'])==20/direct
        assert replayoriginal(graph,(0,1),cuts)['persistent']
        assert nx.utils.graphs_equal(graph,before)
        graph.remove_edge(*cuts[0])
        assert replayoriginal(graph,(0,1),cuts)['cuts']==cuts[1:]
        assert replayoriginal(graph,(0,1),cuts)['persistent']
        print('PASS interpretation fixtures: coordinated cut, exact source averaging, cost ratios, immutability and absent-edge replay')

    def application():
        nodes,pairs,audit = readflows(workbookrows(root/'data/raw/table1.xlsx'))
        graph,geo = buildgraph(nodes,pairs),adjacency(adjacencypath)
        cases,occurrences,saved = readresults(root/'data/results/commutingresults.csv',graph)
        assert set(occurrences)=={colorado,bigbend,edwardsreal}
        assert [len(occurrences[c]) for c in (colorado,bigbend,edwardsreal)]==[10,11,3]
        regions = {c:structure(graph,c,geo,saved) for c in occurrences}
        C,B = regions[colorado],regions[bigbend]
        assert C['counties']==['Alamosa County','Conejos County','Costilla County','Mineral County','Rio Grande County','Saguache County']
        assert (C['commuters'],C['internalflow'],C['boundarysize'],C['disruption'])==(19095,4592,2,100)
        expected = {('08003','08021'):1149,('08003','08023'):427,('08003','08105'):1620,('08003','08109'):454,
                    ('08021','08023'):57,('08021','08105'):182,('08079','08105'):89,('08105','08109'):614}
        assert {tuple(d['edge']):d['flow'] for d in C['internaledges']}==expected
        assert tuple(d['edge'] for d in C['boundaryedges'])==coloradocuts
        assert [d['flow'] for d in C['boundaryedges']]==[74,26]
        assert [d['ranks'] for d in C['boundaryedges']]==[dict(bc=110,bcw=708,weak=6903,strong=17357),dict(bc=280,bcw=6978,weak=1735,strong=22505)]
        for edge in coloradocuts: assert len(components(graph,(edge,)))==1
        assert sorted(map(len,components(graph,coloradocuts)))==[6,3099]
        assert all(not d['bridge'] for d in C['boundaryedges'])
        for name,sign in [('weak',1),('strong',-1)]:
            order = sorted(edges(graph),key=lambda e:(sign*graph.edges[e]['flow'],e))
            for d in C['boundaryedges']: assert d['ranks'][name]==order.index(tuple(d['edge']))+1
        assert (len(C['geographicadjacencies']),C['geographicdiameter'],C['geographicdensity'])==(9,3,.6)
        assert (len(B['geographicadjacencies']),B['geographicdiameter'],B['geographicdensity'])==(3,1,1)
        assert all(d['geographicconnected'] for d in regions.values())
        cs,bs = evaluate(graph,C),evaluate(graph,B)
        expected = ['4132/345','194946387/30522370','64982129/207','116550346269/3052237']
        for (p,a,mode),want in zip(objectives,expected):
            key = p+'/'+a
            assert cs[key]['exact']==want
            assert Fraction(cs[key]['exact'])>Fraction(bs[key]['exact'])
            assert Fraction(cs[key]['disruptionperexpected'])<Fraction(bs[key]['disruptionperexpected'])
        for case in cases:
            assert len(case['partitions'])==1
            assert sum(map(len,case['partitions'][0]))==3105
            if case['target']==case['prior']=='commuters' and case['budget']>=3:
                assert ('41033',) in case['partitions'][0]
        print('PASS all 16 saved optima, 3 recurring regions, Colorado internal/boundary flows, all ranks, geographic adjacency and four exact scores')
        return nodes,pairs,graph,geo,occurrences,regions

    def robustness(nodes,pairs,graph,geo,regions,occurrences):
        actual = sweep(nodes,pairs,geo)
        expected = [(9,4622,21,681,[6],False,3107),(8,4592,2,100,[6],True,6),(8,4592,1,74,[6],True,6),
                    (7,4503,1,74,[5,1],False,5),(7,4503,1,74,[5,1],False,5)]
        for r,want in zip(actual,expected):
            d,replay = r['data'],r['replay']
            assert (len(d['internaledges']),d['internalflow'],d['boundarysize'],d['disruption'],sorted(map(len,d['internalcomponents']),reverse=True),replay['persistent'],replay['anchorsize'])==want
            assert d['count']==6 and d['commuters']==19095
            for value in r['scores'].values(): assert Fraction(value['exact'])-Fraction(value['baseline'])==Fraction(value['increment'])
        assert len(actual[0]['replay']['added'])==3101
        assert not actual[1]['replay']['added'] and not actual[2]['replay']['lost']
        for row in actual[3:]:
            assert row['replay']['lost']==('08079',) and row['replay']['alreadyisolated']==['08079']
            assert row['replay']['othercandidatecomponents']==[('08079',)]
            assert row['replay']['cuts']==coloradocuts[:1]
        for edge,k in [(coloradocuts[1],Fraction(13,12)),(('08079','08105'),Fraction(46,35)),(coloradocuts[0],Fraction(37,19))]:
            assert not buildgraph(nodes,pairs,k,False).has_edge(*edge)
            assert buildgraph(nodes,pairs,k-Fraction(1,100000),False).has_edge(*edge)
        expectedrows = records(graph,regions,occurrences,actual)
        with (root/'data/results/commutinginterpret.csv').open() as stream: saved = list(csv.DictReader(stream))
        assert len(saved)==len(expectedrows)==32
        for row,want in zip(saved,expectedrows):
            expectedrow = {k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else '' if v is None else str(v) for k,v in want.items()}
            assert row==expectedrow
        report = savedreports.text('commutinginterpret')
        for text in ['4592','19095','681','3107','46/35','all 16 completed main-graph cases','not newly computed exact optima']:
            assert text in report,text
        print('PASS all five robustness thresholds, exact strict-MOE transition points, 32 saved records and report coverage')

    fixtures()
    nodes,pairs,graph,geo,occurrences,regions = application()
    robustness(nodes,pairs,graph,geo,regions,occurrences)
    print('Commuting interpretation tests: Passed 3, Failed 0')

def testcompetition():
    import records as savedreports

    import csv

    import json

    from fractions import Fraction

    import networkx as nx

    from air import edges,rankings

    from competition import bestthreshold,build,cases,evaluate,families,mistakes,root

    def construction():
        configs = cases()
        assert len(configs)==len(set(configs))==70
        assert sum(name=='A' for name,m,h in configs)==35
        assert sum(name=='B' for name,m,h in configs)==35
        assert {m for name,m,h in configs}=={12,16,20,24,30}
        assert {h for name,m,h in configs if name=='A'}=={3,4,5,6,8,10,12}
        assert {h for name,m,h in configs if name=='B'}=={4,5,6,7,8,10,12}
        signatures = set()
        for name,m,h in configs:
            graph,R,A,B = build(name,m,h)
            repeated,P,Q,U = build(name,m,h)
            r,k,widths = families[name]
            assert nx.utils.graphs_equal(graph,repeated) and (R,A,B)==(P,Q,U)
            assert (len(R),len(A),len(B),len(graph))==(r,60-m,m,r+60)
            assert set(graph)==set(range(r+60)) and set(R)|set(A)|set(B)==set(graph)
            assert set(R).isdisjoint(A) and set(R).isdisjoint(B) and set(A).isdisjoint(B)
            for block in (R,A,B):
                assert nx.is_connected(graph.subgraph(block))
                assert graph.subgraph(block).number_of_edges()==len(block)*(len(block)-1)//2
                assert len(block)-1>k
            boundary = tuple((R[i],A[i]) for i in range(k))
            gateway = tuple((A[i],B[i]) for i in range(h))
            assert tuple(e for e in edges(graph) if (e[0] in R)!=(e[1] in R))==boundary
            assert tuple(e for e in edges(graph) if (e[0] in B)!=(e[1] in B))==gateway
            assert len(boundary)==k and len(gateway)==h>k
            assert nx.edge_connectivity(graph)==k
            outside = graph.subgraph(A+B).copy()
            assert nx.edge_connectivity(outside)==h>k
            remaining = graph.copy()
            remaining.remove_edges_from(boundary)
            assert {frozenset(c) for c in nx.connected_components(remaining)}=={frozenset(R),frozenset(A+B)}
            assert graph.number_of_edges()==sum(len(block)*(len(block)-1)//2 for block in (R,A,B))+k+h
            assert all(data==dict(strength=1) for u,v,data in graph.edges(data=True))
            signatures.add(edges(graph))
        assert len(signatures)==70
        print('PASS construction: 35 A + 35 B; deterministic cliques and exact boundaries; global connectivity k, outside connectivity h>k')

    def read():
        with (root/'data/results/competition.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        integers = ('r','k','budget','a_size','b_size','gateway_width','n','m_edges','exact_subsets','exact_states')
        decimals = ('exact','candidate_score','bc_score','bc_ratio','candidate_load','gateway_load','load_ratio')
        for row in rows:
            for field in integers: row[field] = int(row[field])
            for field in decimals: row[field] = float(row[field])
            for field in ('candidate_optimal','top_bc_is_candidate'):
                assert row[field] in ('True','False')
                row[field] = row[field]=='True'
            for field in ('candidate_boundary_ranks','gateway_ranks'): row[field] = json.loads(row[field])
            for field in ('bc_cuts','candidate_boundary','optimal_cut','gateway_edges'):
                row[field] = tuple(tuple(e) for e in json.loads(row[field]))
            row['optimal_partition'] = [tuple(c) for c in json.loads(row['optimal_partition'])]
        assert [(row['family'],row['b_size'],row['gateway_width']) for row in rows]==cases()
        return rows

    def replay(graph,cuts):
        remaining = graph.copy()
        assert len(cuts)==len(set(cuts)) and all(graph.has_edge(*e) for e in cuts)
        remaining.remove_edges_from(cuts)
        value = sum(Fraction(len(graph)-len(nx.node_connected_component(remaining,s)),len(graph)) for s in graph)
        groups = sorted((tuple(sorted(c)) for c in nx.connected_components(remaining)),key=lambda c:(-len(c),c))
        return value,groups

    def results(rows):
        for row in rows:
            graph,R,A,B = build(row['family'],row['b_size'],row['gateway_width'])
            original = graph.copy()
            r,k,widths = families[row['family']]
            assert (row['r'],row['k'],row['budget'],row['n'],row['a_size'],row['m_edges'])==(r,k,k,len(graph),len(A),graph.number_of_edges())
            assert len(row['optimal_cut'])<=k
            exact,groups = replay(graph,row['optimal_cut'])
            assert exact==Fraction(row['exact_fraction']) and float(exact)==row['exact']
            assert groups==row['optimal_partition']
            candidate,groups = replay(graph,row['candidate_boundary'])
            assert candidate==Fraction(2*r*60,r+60)==Fraction(row['candidate_fraction'])<=exact
            assert float(candidate)==row['candidate_score']
            assert row['candidate_optimal']==(candidate==exact)
            assert row['candidate_optimal'] and row['optimal_cut']==row['candidate_boundary']
            assert [len(c) for c in groups]==[60,r]
            bc,groups = replay(graph,row['bc_cuts'])
            assert bc==Fraction(row['bc_fraction'])<=exact and float(bc)==row['bc_score']
            assert bc/exact==Fraction(row['bc_ratio_fraction']) and float(bc/exact)==row['bc_ratio']
            assert bc/exact in (0,1)
            if not bc: assert set(row['bc_cuts'])<=set(row['gateway_edges'])
            assert row['top_bc_is_candidate']==(set(row['bc_cuts'])==set(row['candidate_boundary']))==(bc==exact)
            adapted = graph.copy()
            nx.set_edge_attributes(adapted,1,'passengers')
            order = rankings(adapted)['bc']
            assert row['bc_cuts']==order[:k]
            assert row['candidate_boundary_ranks']==[order.index(e)+1 for e in row['candidate_boundary']]
            assert row['gateway_edges']==tuple((A[i],B[i]) for i in range(row['gateway_width']))
            assert row['gateway_ranks']==[order.index(e)+1 for e in row['gateway_edges']]
            candidate = Fraction(r*60,k)
            gateway = Fraction((r+len(A))*len(B),row['gateway_width'])
            assert candidate==Fraction(row['candidate_load_fraction']) and float(candidate)==row['candidate_load']
            assert gateway==Fraction(row['gateway_load_fraction']) and float(gateway)==row['gateway_load']
            assert gateway/candidate==Fraction(row['load_ratio_fraction']) and float(gateway/candidate)==row['load_ratio']
            assert nx.utils.graphs_equal(graph,original)
        print('PASS results: all 70 exact/candidate/BC cuts independently replayed; every rank, score bound, optimal partition and load proxy verified')

    def repeatability(rows):
        for config in (('A',12,12),('A',30,3),('B',12,12),('B',30,4)):
            first,timing = evaluate(*config)
            second,timing = evaluate(*config)
            saved = next(row for row in rows if (row['family'],row['b_size'],row['gateway_width'])==config)
            assert first==second==saved
        print('PASS repeatability: four extreme tuples reproduce identical exact values, rankings, cuts and partitions')

    def classifiers(rows):
        fixture = [dict(load_ratio_fraction=str(Fraction(i,2)),bc_ratio_fraction=str(int(i<=2))) for i in range(1,5)]
        assert not mistakes(fixture,Fraction(1))
        assert bestthreshold(fixture)==[(0,Fraction(5,4),Fraction(1),Fraction(3,2))]
        assert len(mistakes(fixture,Fraction(3,2)))==1
        tied = [dict(load_ratio_fraction='1',bc_ratio_fraction=str(i)) for i in range(2)]
        assert bestthreshold(tied)==[(1,Fraction(0),None,Fraction(1)),(1,Fraction(1),Fraction(1),None)]
        for result in (0,1):
            fixture = [dict(load_ratio_fraction=str(i),bc_ratio_fraction=str(result)) for i in (1,2)]
            assert bestthreshold(fixture)==([(0,Fraction(0),None,Fraction(1))] if result==0 else [(0,Fraction(2),Fraction(2),None)])
        for selected in (rows,[row for row in rows if row['family']=='A'],[row for row in rows if row['family']=='B']):
            levels = sorted({Fraction(row['load_ratio_fraction']) for row in selected})
            expected = min(len(mistakes(selected,t)) for t in [levels[0]-1]+levels)
            for errors,t,low,high in bestthreshold(selected):
                assert errors==expected==len(mistakes(selected,t))
                assert low is None or low<=t
                assert high is None or t<high
        text = savedreports.text('competition')
        assert 'Exact searches completed: 70/70' in text
        assert 'structural proxies, not exact BC formulas' in text
        print('PASS threshold fitting: strict ties, perfect separation, contradictory labels, both tails and minimum errors checked')

    construction()
    rows = read()
    results(rows)
    repeatability(rows)
    classifiers(rows)
    print('Competition tests: Passed 4, Failed 0')

def testcompetitiontheory():
    from collections import Counter

    from fractions import Fraction

    from time import perf_counter

    import networkx as nx

    from competition import build

    from competition import formulas,margin,theoryreport as report,root,terms,verify

    def structural(rows):
        assert len(rows)==70 and Counter(row['family'] for row in rows)==dict(A=35,B=35)
        assert len({(row['family'],row['b_size'],row['gateway_width']) for row in rows})==70
        for row in rows:
            graph,R,A,B = build(row['family'],int(row['b_size']),int(row['gateway_width']))
            k,h = int(row['k']),int(row['gateway_width'])
            assert all(len(block)-1>k for block in (R,A,B)) and h>k
            for block in (R,A,B):
                assert graph.subgraph(block).number_of_edges()==len(block)*(len(block)-1)//2
            assert nx.edge_connectivity(graph)==k
            assert nx.edge_connectivity(graph.subgraph(A+B))>k
            remaining = graph.copy()
            remaining.remove_edges_from((R[i],A[i]) for i in range(k))
            assert {frozenset(c) for c in nx.connected_components(remaining)}=={frozenset(R),frozenset(A+B)}
        print('PASS structure: all 70 existing graphs have minimum cut k; clique and h>k conditions certify uniqueness')

    def counting(rows,checks):
        assert checks['graphs']==70 and len(checks['classes'])==18
        assert not checks['formulaerrors'],checks['formulaerrors']
        assert not checks['floaterrors'],checks['floaterrors']
        assert len(checks['relevantroundoff'])==checks['relevantunequal']
        for row in rows:
            r,a,m,k,h = (int(row[key]) for key in ('r','a_size','b_size','k','gateway_width'))
            values = formulas(r,a,m,k,h)
            contributions = terms(r,a,m,k,h)
            assert all(isinstance(value,Fraction) for parts in contributions.values() for value in parts)
            assert values['candidate']==Fraction(2*r*(a+m),k)
            assert values['shared']==Fraction(2*a*m,h)+Fraction(r*(2*m+k-h),k)
            assert values['other']==Fraction(2*a*m,h)+r
            assert k*contributions['candidate'][1]==r*a
            assert k*contributions['candidate'][3]==r*m
            assert k*contributions['shared'][3]+(h-k)*contributions['other'][3]==r*m
            assert k*contributions['shared'][2]+(h-k)*contributions['other'][2]==a*m
        print(f'PASS formulas: exact pair-type counts for {checks["edges"]} edges, including {checks["relevant"]} cross-block edges; all 18 classes covered')
        print(f'NetworkX maximum absolute float error: {checks["floatmaximum"]:.16g}; cross-block maximum: {checks["relevantmaximum"]:.16g}')
        print(f'Cross-block bitwise differences from rounded exact values: {checks["relevantunequal"]}/{checks["relevant"]} (roundoff, not exact-count mismatches)')

    def inequalities(rows,checks):
        assert not checks['outcomeerrors'],checks['outcomeerrors']
        assert not checks['cuterrors'],checks['cuterrors']
        assert not checks['ties']
        assert checks['exactfailures']==dict(A=17,B=14)
        for row in rows:
            r,a,m,k,h = (int(row[key]) for key in ('r','a_size','b_size','k','gateway_width'))
            values = formulas(r,a,m,k,h)
            assert values['shared']>values['other']
            assert max(value for label,value in values.items() if label not in ('candidate','shared','other'))<values['candidate']
            delta = margin(r,a,m,k,h)
            assert values['shared']-values['candidate']==Fraction(delta,k*h)
            assert delta==k*(2*m*(60-m)-3*h*(120-2*m+h-k))
            assert (delta>0)==(Fraction(row['bc_ratio_fraction'])==0)
        print('PASS inequalities: 70/70 outcomes and selected cut sets matched; A has 17 failures, B has 14; no cross-class threshold ties')
        print('Family A: failure iff 2m(60-m) > 3h(118-2m+h)')
        print('Family B: failure iff 2m(60-m) > 3h(117-2m+h)')
        for name in ('A','B'):
            for m in sorted({int(row['b_size']) for row in rows if row['family']==name}):
                selected = [row for row in rows if row['family']==name and int(row['b_size'])==m]
                predicted = [int(row['gateway_width']) for row in selected if margin(int(row['r']),int(row['a_size']),m,int(row['k']),int(row['gateway_width']))<=0]
                observed = [int(row['gateway_width']) for row in selected if Fraction(row['bc_ratio_fraction'])==1]
                assert predicted==observed
                print(f'{name}, m={m}: successful tested widths {predicted}')

    def main():
        start = perf_counter()
        rows,checks = verify()
        assert report(rows,checks)
        structural(rows)
        counting(rows,checks)
        inequalities(rows,checks)
        result = f'Competition theory tests: Passed 3, Failed 0\nTest runtime: {perf_counter()-start:.3f} seconds\n'
        print(result,end='')

    main()

def testfigures():
    import csv,hashlib,json,re,sys,tempfile
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'figures'))
    import fig7,fig8
    root = Path(__file__).resolve().parent.parent
    protected = [p for p in (root/'data').rglob('*') if p.is_file()]
    before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    hashes = {7:'8dd1003df65a4e109b80150884958710bd2593535f78cb69e6274cf1c47e5638',
              8:'52af0a9fc1629e785b4688693e297ae22e50b4e53fdcf8ac4e747a69b02a018d'}
    for number,module in ((7,fig7),(8,fig8)):
        tex = module.figure()
        assert tex.encode()==module.output.read_bytes()
        assert hashlib.sha256(tex.encode()).hexdigest()==hashes[number]
        with patch.object(sys,'argv',[f'fig{number}.py','--check']), \
             patch.object(Path,'write_bytes',side_effect=AssertionError('Verification wrote a file')), \
             patch.object(Path,'write_text',side_effect=AssertionError('Verification wrote a file')):
            module.main()
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory)/'invalid.tex'
            invalid.write_text(tex+'changed')
            with patch.object(module,'output',invalid),patch.object(sys,'argv',[f'fig{number}.py','--check']), \
                 patch.object(Path,'write_bytes',side_effect=AssertionError('Verification overwrote a mismatch')):
                try: module.main()
                except SystemExit as error: assert str(error).startswith('FAIL Figure')
                else: raise AssertionError('Mismatching saved figure was accepted')
            assert invalid.read_text()==tex+'changed'
    plots = re.findall(r'coordinates \{\n(.*?)\n\};',fig7.figure(),re.S)
    assert len(plots)==6
    total = 0
    for panel,(path,target,count) in enumerate(((fig7.commuting,'counties',37),(fig7.air,'airports',60))):
        with path.open(newline='') as stream:
            records = [r for r in csv.DictReader(stream) if r['prior']=='uniform' and r['target']==target and int(r['budget'])<=3]
        loaded = fig7.load(path,'uniform',target)
        for b in (1,2,3):
            rows = sorted((r for r in records if int(r['budget'])==b),key=lambda r:Fraction(r['threshold']))
            points = [(Fraction(r['threshold']),Fraction(r['exact'])) for r in rows]
            assert loaded[b]==points and len(points)==count
            assert all(a[0]<b[0] for a,b in zip(points,points[1:]))
            plotted = re.findall(r'\(([^,]+),([^\)]+)\)',plots[3*panel+b-1])
            assert len(plotted)==count
            for (x,y),(px,py),row in zip(points,plotted,rows):
                assert py==f'{float(y):.8g}'
                assert abs(float(px)-float(x))<5e-8
                n = int(row['n'])
                sizes = json.loads(row['componentsizes'])
                assert sum(sizes)==n and y==Fraction(n*n-sum(c*c for c in sizes),n)
                if x==Fraction(46,35):
                    assert px=='1.3142857142857143'
                    assert float(px)==float(Fraction(46,35))
            if panel==0:
                index = [x for x,y in points].index(Fraction(46,35))
                assert points[index-1][0]==Fraction(21,16) and points[index+1][0]==Fraction(263,200)
                assert int(rows[index-1]['n'])==3084 and int(rows[index]['n'])==3083
            total += count
    assert total==291
    print('PASS Figure 7: all 291 CSV values and component scores; rational ordering and full 46/35 breakpoint precision')
    text = fig8.figure()
    scopes = re.findall(r'\\begin\{scope\}\[shift=\{\(0,([^)]*)\)\}\](.*?)\\end\{scope\}',text,re.S)
    classified = {}
    with (root/'data/results/competition.csv').open(newline='') as stream:
        expected = {(r['family'],int(r['b_size']),int(r['gateway_width'])):int(Fraction(r['bc_ratio_fraction'])) for r in csv.DictReader(stream)}
    for name,(r,k,widths) in fig8.families.items():
        body = next(body for offset,body in scopes if '{Family '+name+'}' in body)
        cells = {}
        for line in body.splitlines():
            one = re.fullmatch(r'\\tile\{(fail|succ)\}\{(\d+)\}\{(\d+)\}',line)
            many = re.fullmatch(r'\\foreach \\r in \{([^}]+)\} \{\\tile\{(fail|succ)\}\{(\d+)\}\{\\r\}\}',line)
            if one:
                color,column,row = one.groups()
                indices = [int(row)]
            elif many:
                labels,color,column = many.groups()
                numbers = labels.split(',')
                indices = list(range(int(numbers[0]),int(numbers[-1])+1)) if '...' in numbers else list(map(int,numbers))
            else: continue
            for row in indices:
                key = int(column),row
                assert key not in cells
                cells[key] = int(color=='succ')
        assert set(cells)=={(x,y) for x in range(1,6) for y in range(1,8)}
        curve = re.search(r'\\draw\[black,line width=0.8pt\]\n(.*?);',body,re.S).group(1)
        points = [tuple(map(int,p)) for p in re.findall(r'\((\d+),(\d+)\)',curve)]
        assert points[0][0]==0 and points[-1][0]==5
        for i,m in enumerate(fig8.sizes,1):
            height = sum(2*k*(60-m)*m-r*h*(2*(60-m)+h-k)>0 for h in widths)
            horizontal = [(x,y,u,v) for (x,y),(u,v) in zip(points,points[1:]) if y==v and x<i-0.5<u]
            assert len(horizontal)==1 and horizontal[0][1]==height
            for j,h in enumerate(widths,1):
                key = name,m,h
                classified[key] = cells[i,j]
                assert classified[key]==expected[key]==int(j>height)
    assert len(classified)==70 and classified==expected
    assert '\\begin{axis}' not in text
    print('PASS Figure 8: 70 distinct saved classifications; both discrete theoretical boundaries and exact reference styling')
    changed = fig7.load(fig7.commuting,'uniform','counties')
    changed[1][0] = (changed[1][0][0],changed[1][0][1]+1)
    original = fig7.load
    with patch.object(fig7,'load',side_effect=lambda path,prior,target:changed if target=='counties' else original(path,prior,target)):
        assert fig7.figure()!=fig7.output.read_text()
    changed = fig8.load()
    first = next(iter(changed))
    changed[first] = 1-changed[first]
    with patch.object(fig8,'load',return_value=changed):
        assert fig8.figure()!=fig8.output.read_text()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in before.items())
    print('PASS data-driven generation, non-writing checks, mismatch rejection, and unchanged raw/results/provenance bytes')


def testhighschool():
    from fractions import Fraction

    from itertools import combinations

    from random import Random

    import networkx as nx

    from highschool import componentvalue,identity,weightedexact,weightedpolicy,replay,makegraph,valuesandbc,policies

    from safezones import exact,first_spread,layered,simulate

    def brute(graph,infected,weights):
        after = infected|{v for u in infected for v in graph.neighbors(u)}
        if after==infected: return sum(weights[v] for v in graph if v not in infected)
        values = []
        for action in [None]+list(graph.edges):
            current = graph.copy()
            if action is not None: current.remove_edge(*action)
            after = infected|{v for u in infected for v in current.neighbors(u)}
            values.append(brute(current,after,weights))
        return max(values)

    def testidentity():
        rng = Random(20261008)
        graphs = [nx.path_graph(5),nx.cycle_graph(5),nx.complete_graph(4),nx.barbell_graph(3,1)]
        for n in (4,5,6):
            graph = nx.path_graph(n)
            graph.add_edges_from(e for e in nx.non_edges(graph) if rng.random()<.35)
            graphs.append(graph)
        checks = 0
        for graph in graphs:
            weights = {v:v+1 for v in graph}
            total = sum(weights.values())
            masses = {v:rng.randrange(5) for v in graph}
            masses[0] += 1
            priors = [{v:Fraction(weights[v],total) for v in graph},
                      {v:Fraction(1,len(graph)) for v in graph},
                      {v:Fraction(masses[v],sum(masses.values())) for v in graph}]
            edges = list(graph.edges)
            cuts = [(),tuple(edges),tuple(edges[::2])]+[(e,) for e in edges]+list(combinations(edges,2))
            original = graph.copy()
            for prior in priors:
                for removed in cuts:
                    value = identity(graph,removed,weights,prior)
                    assert isinstance(value,Fraction) and 0<=value<=total
                    checks += 1
            assert nx.utils.graphs_equal(graph,original)
        print('PASS weighted identity:',checks,'exact rational comparisons')

    def testsolver():
        checks = 0
        for graph in nx.graph_atlas_g():
            if not 2<=len(graph)<=5 or not nx.is_connected(graph): continue
            weights = {v:v+1 for v in graph}
            weighted = weightedexact(graph,weights)
            unit = weightedexact(graph,dict.fromkeys(graph,1))
            original = graph.copy()
            for source in graph:
                value,actions = weighted(source)
                assert value==brute(graph,set(first_spread(graph,source)),weights)
                assert replay(graph,source,actions,weights)[0]==value
                assert unit(source)[0]==exact(graph,source)
                checks += 1
            assert nx.utils.graphs_equal(graph,original)
        graph = nx.path_graph(3)
        assert weightedexact(graph,dict.fromkeys(graph,10))(1)==(0,[])
        graph = nx.cycle_graph(4)
        assert weightedexact(graph,dict.fromkeys(graph,1))(0)==(0,[None])
        graph = nx.Graph([(0,1),(1,2),(1,3)])
        weights = {0:1,1:1,2:5,3:20}
        assert weightedexact(graph,weights)(0)==(20,[(1,3)])
        graph = layered((1,2,2,1))
        weights = dict.fromkeys(graph,1)
        weights[5] = 100
        value,actions = weightedexact(graph,weights)(0)
        assert value==100 and not any(v in first_spread(graph,0) for v in actions[0])
        assert replay(graph,0,actions,weights)[1]==[5]
        print('PASS weighted exact:',checks,'independent brute-force and unit-weight production comparisons')
        print('PASS mandatory spread, waiting, weighted choice and proactive protection')

    def testpolicies():
        weights = {'A':3,'B':5,'C':11,'D':17,'E':23}
        strengths = {('A','B'):10,('B','C'):30,('C','D'):5,('C','E'):20,('D','E'):15}
        graph = makegraph(weights,strengths)
        values,bc,bcw = valuesandbc(graph)
        original = graph.copy()
        for source in graph:
            for name in policies:
                value,actions = weightedpolicy(graph,source,name,weights,values,bc,bcw)
                count,cuts,states = simulate(graph,source,name,values,bc,bcw,trace=True)
                assert actions==[action for infected,action in states[2:]]
                assert value==sum(weights[v] for v in graph if v not in states[-1][0])
        assert nx.utils.graphs_equal(graph,original)
        print('PASS six policy action rules and fixed original centralities preserved')

    testidentity()
    testsolver()
    testpolicies()
    print('Application regression: Passed 3, Failed 0')

def testhospital():
    import gzip

    import hashlib

    import random

    import tempfile

    from collections import Counter

    from fractions import Fraction

    from itertools import combinations

    from pathlib import Path

    import networkx as nx

    from hospital import readcontacts,makegraph,selectedgraphs,prior,targets,score,components,partitions,solve,rankings,checkselected,checkvalues,sourcehash,root

    def decorate(graph):
        graph = graph.copy()
        for i,v in enumerate(sorted(graph)): graph.nodes[v]['role'] = ('PAT','NUR','MED','ADM')[i%4]
        for i,e in enumerate(sorted(graph.edges)): graph.edges[e]['duration'] = 20*(i%7+1)
        return graph

    def brute(graph,p,a,b):
        order = sorted(tuple(sorted(e)) for e in graph.edges)
        bestvalue,bestcuts = Fraction(-1),None
        for size in range(min(b,len(order))+1):
            for cuts in combinations(order,size):
                remaining = graph.copy()
                remaining.remove_edges_from(cuts)
                value = sum(p[s]*sum(a[v] for v in graph if v not in nx.node_connected_component(remaining,s)) for s in graph)
                if value>bestvalue:
                    bestvalue,bestcuts = value,cuts
        return bestvalue,bestcuts

    def mustfail(action):
        try: action()
        except ValueError: return
        raise AssertionError('Expected ValueError')

    def parsing():
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'contacts.dat.gz'
            text = '20 1 2 PAT NUR\n40 2 1 NUR PAT\n60 2 3 NUR MED\n25200 3 4 MED ADM\n64800 4 5 ADM PAT\n'
            with gzip.open(path,'wt') as stream: stream.write(text)
            rows,roles = readcontacts(path)
            graph = makegraph(rows,roles,threshold=40)
            assert set(graph)=={1,2,3,4,5}
            assert list(graph.edges)==[(1,2)] and graph.edges[1,2]['duration']==40
            night = makegraph(rows,roles,'night')
            assert set(night.edges)=={(3,4)}
            assert sum(prior(graph,'HCW').values())==1
            assert targets(graph,'PAT')=={1:1,2:0,3:0,4:0,5:1}
            for bad in ('20 1 2 PAT NUR\n20 2 1 NUR PAT\n','20 1 1 PAT PAT\n',
                        '20 1 2 PAT NUR\n40 1 3 NUR MED\n','21 1 2 PAT NUR\n','20 1 2 PAT XYZ\n'):
                with gzip.open(path,'wt') as stream: stream.write(bad)
                mustfail(lambda:readcontacts(path))
        print('PASS parser: duration, roles, duplicate rejection, endpoint boundaries and active isolates')

    def exhaustive():
        graphs = [decorate(g) for g in nx.graph_atlas_g() if 2<=len(g)<=5 and nx.is_connected(g)]
        rng = random.Random(20261029)
        for n in range(4,8):
            for j in range(4):
                graph = nx.gnp_random_graph(n,.2+.15*j,seed=rng.randrange(1000000))
                while graph.number_of_edges()>12: graph.remove_edge(*sorted(graph.edges)[-1])
                graphs.append(decorate(graph))
        graphs += [decorate(nx.complete_bipartite_graph(2,5)),decorate(nx.empty_graph(4)),decorate(nx.path_graph(1))]
        checks,identities,replays = 0,0,0
        for graph in graphs:
            before = nx.node_link_data(graph)
            catalog,stats = partitions(graph,4)
            for cuts,groups in catalog['states']:
                assert components(graph,cuts)==groups
                replays += 1
            n = len(graph)
            nodes = sorted(graph)
            distributions = [(prior(graph,'uniform'),targets(graph,'all')),
                             ({v:Fraction(i+1,n*(n+1)//2) for i,v in enumerate(nodes)},
                              {v:Fraction(i%3,2) for i,v in enumerate(nodes)})]
            if n>1: distributions.append((prior(graph,'HCW'),targets(graph,'PAT')))
            for p,a in distributions:
                for b in range(5):
                    expected = brute(graph,p,a,b)
                    actual = solve(graph,p,a,b,catalog)
                    assert actual==expected,(sorted(graph.edges),b,actual,expected)
                    cuts = actual[1]
                    remaining = graph.copy()
                    remaining.remove_edges_from(cuts)
                    pairvalue = sum(p[s]*a[t] for s in graph for t in graph if not nx.has_path(remaining,s,t))
                    assert pairvalue==score(graph,cuts,p,a)==actual[0]
                    checks += 1
                    identities += 1
            assert nx.node_link_data(graph)==before
        graph = decorate(nx.complete_bipartite_graph(2,5))
        catalog,stats = partitions(graph,4)
        assert stats['atomsizes']==[1,1,1,1,1,2]
        assert not graph.has_edge(0,1)
        print(f'PASS exact contraction: {len(graphs)} graphs, {checks} direct brute-force comparisons, {identities} weighted identities, {replays} partition replays')

    def conventions():
        graph = nx.Graph()
        graph.add_nodes_from((v,dict(role='PAT' if v%2 else 'NUR')) for v in range(3))
        graph.add_weighted_edges_from([(0,1,60),(1,2,60),(0,2,20)],weight='duration')
        before = nx.node_link_data(graph)
        orders = rankings(graph)
        assert orders['bc']==((0,1),(0,2),(1,2))
        assert orders['bcw']==((0,1),(1,2),(0,2))
        assert orders['weak']==((0,2),(0,1),(1,2))
        assert orders['strong']==((0,1),(1,2),(0,2))
        assert nx.node_link_data(graph)==before
        reverse = nx.Graph()
        reverse.add_nodes_from(reversed(list(graph.nodes(data=True))))
        reverse.add_edges_from((v,u,dict(d)) for u,v,d in reversed(list(graph.edges(data=True))))
        assert rankings(reverse)==orders
        p,a = prior(graph,'HCW'),targets(graph,'PAT')
        assert solve(graph,p,a,2)==solve(reverse,p,a,2)
        for b in (-1,1.5,True):
            mustfail(lambda:partitions(graph,b))
            mustfail(lambda:solve(graph,p,a,b))
        catalog,stats = partitions(graph,1)
        mustfail(lambda:solve(graph,p,a,2,catalog))
        for bad in (nx.DiGraph(graph),nx.MultiGraph(graph),nx.Graph()): mustfail(lambda:partitions(bad,2))
        loop = graph.copy();loop.add_edge(0,0)
        mustfail(lambda:partitions(loop,2))
        mustfail(lambda:prior(graph,'unknown'))
        mustfail(lambda:targets(graph,'unknown'))
        mustfail(lambda:score(graph,[],{v:Fraction(1,2) for v in graph},a))
        mustfail(lambda:score(graph,[],p,{v:-1 for v in graph}))
        mustfail(lambda:score(graph,[],p,{v:float('nan') for v in graph}))
        mustfail(lambda:score(graph,[(0,99)],p,a))
        mustfail(lambda:score(graph,[(0,1),(1,0)],p,a))
        for value in (0,-1,float('inf'),float('nan'),'20',True):
            invalid = graph.copy();invalid.edges[0,1]['duration'] = value
            mustfail(lambda:rankings(invalid))
        assert solve(graph,p,{v:0 for v in graph},4)==(0,())
        print('PASS fixed BC/inverse-duration BCw rankings, canonical ties, zero targets, validation and graph immutability')

    def hospital():
        path = root/'data/raw/hospital_lyon_contacts.dat.gz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sourcehash
        rows,roles = readcontacts(path)
        assert len(rows)==32424 and Counter(roles.values())=={'PAT':29,'NUR':27,'MED':11,'ADM':8}
        graphs = selectedgraphs(rows,roles)
        checkselected(graphs)
        catalog = {name:partitions(graph,4)[0] for name,graph in graphs.items()}
        checkvalues(graphs,catalog)
        cases = 0
        for name,graph in graphs.items():
            before = nx.node_link_data(graph)
            orders = rankings(graph)
            for pname in ('uniform','HCW','PAT'):
                for aname in ('all','PAT'):
                    p,a = prior(graph,pname),targets(graph,aname)
                    previous = Fraction(0)
                    for b in range(1,5):
                        value,cuts = solve(graph,p,a,b,catalog[name])
                        assert len(cuts)<=b and score(graph,cuts,p,a)==value>=previous
                        previous = value
                        remaining = graph.copy();remaining.remove_edges_from(cuts)
                        direct = sum(p[s]*sum(a[v] for v in graph if v not in nx.node_connected_component(remaining,s)) for s in graph)
                        assert direct==value
                        for order in orders.values(): assert score(graph,order[:b],p,a)<=value
                        cases += 1
            assert nx.node_link_data(graph)==before
        print(f'PASS hospital: four exact graph fixtures, all 16 supplied scores, {cases} optimum replays and heuristic bounds through b=4')

    parsing()
    exhaustive()
    conventions()
    hospital()
    print('Hospital application tests: Passed 4, Failed 0')

def testhospitalcohorts():
    import records as savedreports

    import csv

    import hashlib

    import json

    from fractions import Fraction

    import networkx as nx

    from hospital import root,sourcehash,readcontacts,partitions

    from hospital import graphsfrom,cohortmetrics,stability,discover,rawpair,estimate,evaluate,resolutions,seeds

    def fixtures():
        graph = nx.Graph()
        graph.add_nodes_from((v,dict(role=role)) for v,role in enumerate(('PAT','PAT','NUR','MED')))
        graph.add_weighted_edges_from([(0,1,180),(0,2,140),(1,2,20),(1,3,20),(2,3,100)],weight='duration')
        before = nx.node_link_data(graph)
        record = cohortmetrics(graph,{0,1})
        assert record['internalseconds']==180 and record['internaledges']==1
        assert record['boundaryedges']==3 and record['boundaryseconds']==180
        assert record['conductance']=='9/19'
        assert record['roles']==dict(PAT=2,NUR=0,MED=0,ADM=0)
        data = evaluate(graph,record,{})
        assert data['componentsizes']==[2,2] and data['exactlytwo']
        assert all(Fraction(d['value'])==2 for d in data['objectives'].values())
        graph.remove_edge(2,3)
        record = cohortmetrics(graph,{0,1})
        data = evaluate(graph,record,{})
        assert data['componentsizes']==[1,1,2] and not data['exactlytwo']
        assert data['objectives']['uniform/all']['value']=='5/2'
        assert data['objectives']['HCW/all']['value']=='3'
        assert data['objectives']['PAT/all']['value']=='2'
        assert data['objectives']['HCW/PAT']['value']=='2'
        graph.add_edge(2,3,duration=100)
        assert nx.node_link_data(graph)==before
        for budget in (1,2,3,4):
            catalog,stats = partitions(graph,budget)
            assert estimate(graph,budget)==stats['bundlesubsets']
        first,origins = discover(graph)
        second,repeated = discover(graph)
        assert first==second and origins==repeated and len(first)==80
        assert {(r['resolution'],r['seed']) for r in first}=={(r,s) for r in resolutions for s in seeds}
        assert nx.node_link_data(graph)==before
        graph.add_node(4,role='PAT')
        runs = [dict(resolution=r,seed=0,communities=c) for r,c in zip(resolutions,
                (((0,1),(2,3,4)),((0,1,2),(3,4)),((0,1,2,4),(3,)),((0,2),(1,3,4))))]
        data = stability(graph,{0,1},runs)
        assert (data['exact'],data['largersubset'],data['exactpatients'],data['patientstogether'])==(1,2,2,3)
        assert data['staffvariants']==[dict(ids=(),runs=1),dict(ids=(2,),runs=1)]
        print('PASS cohort metrics, multi-component cuts, exact identities, reproducible Louvain, stability definitions and preflight counts')

    def application():
        path = root/'data/raw/hospital_lyon_contacts.dat.gz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sourcehash
        rows,roles = readcontacts(path)
        graphs = graphsfrom(rows,roles)
        expected = dict(full=(75,1139,6),tuesday=(49,474,5),wednesday=(49,452,3),night=(53,190,1))
        for name,graph in graphs.items():
            assert (len(graph),graph.number_of_edges(),nx.edge_connectivity(graph))==expected[name]
        audit = rawpair(rows,roles,graphs['night'])
        assert audit['internalseconds']==180 and audit['boundaryedges']==4 and audit['boundaryseconds']==200
        runs,origins = discover(graphs['night'])
        data = stability(graphs['night'],{1460,1784},runs)
        assert (data['exact'],data['largersubset'],data['exactpatients'],data['patientstogether'])==(60,20,60,80)
        assert [r['exact'] for r in data['byresolution']]==[0,20,20,20]
        records = list(csv.DictReader((root/'data/results/hospitalcohorts.csv').open()))
        assert len(records)==174 and sum(r['candidate']=='True' for r in records)==34
        for name in graphs:
            selected = [r for r in records if r['graph']==name]
            covered = {(res,seed):set() for res in resolutions for seed in seeds}
            for r in selected:
                ids = json.loads(r['ids'])
                m = cohortmetrics(graphs[name],ids)
                assert int(r['boundaryedges'])==m['boundaryedges'] and int(r['boundaryseconds'])==m['boundaryseconds']
                assert int(r['internalseconds'])==m['internalseconds'] and (r['candidate']=='True')==m['candidate']
                for res,seed in json.loads(r['stability'])['exactappearances']:
                    assert not covered[res,seed]&set(ids)
                    covered[res,seed].update(ids)
                if r['candidate']=='True':
                    ev = json.loads(r['evaluation'])
                    again = evaluate(graphs[name],m,{})
                    for key in again['objectives']:
                        assert ev['objectives'][key]['value']==again['objectives'][key]['value']
            assert all(vertices==set(graphs[name]) for vertices in covered.values())
        pair = next(r for r in records if r['graph']=='night' and json.loads(r['ids'])==[1460,1784])
        ev = json.loads(pair['evaluation'])
        assert ev['componentsizes']==[2,51]
        expected = {'uniform/all':('204/53','404/53','51/101'), 'HCW/all':('2','88/13','13/44'),
                    'PAT/all':('152/27','100/9','38/75'),'HCW/PAT':('2','4','1/2')}
        for key,(value,optimum,ratio) in expected.items():
            d = ev['objectives'][key]
            assert (d['value'],d['exact']['value'],d['ratio'])==(value,optimum,ratio)
            assert d['classification']=='substantially worse'
        report = savedreports.text('hospitalcohorts')
        assert report.rsplit('\n\n',3)[-3].startswith('Strongest candidate cohort\n')
        assert report.rsplit('\n\n',2)[-2].startswith('Comparison to edge-by-edge optimum\n')
        assert report.rsplit('\n\n',1)[-1].startswith('Candidate paper finding\n')
        print('PASS all four raw graph fixtures, 320 partition coverage, 34 candidate evaluations, exact night pair and report endings')

    fixtures()
    application()
    print('Hospital cohort tests: Passed 2, Failed 0')

def testhospitalinterpret():
    from collections import Counter,defaultdict

    from itertools import combinations

    import networkx as nx

    from hospital import edges,components,partitions

    from hospital import countsets,family,analyze,makereport

    def counting():
        graphs = [nx.path_graph(5),nx.cycle_graph(5),nx.complete_graph(4),nx.star_graph(5),
                  nx.complete_bipartite_graph(2,3),nx.empty_graph(4),nx.disjoint_union(nx.path_graph(3),nx.path_graph(2))]
        checked = 0
        for graph in graphs:
            catalog,stats = partitions(graph,4)
            counts = countsets(graph,catalog,4)
            direct = defaultdict(Counter)
            cutsbygroup = defaultdict(list)
            for size in range(min(4,graph.number_of_edges())+1):
                for cut in combinations(edges(graph),size):
                    groups = components(graph,cut)
                    direct[groups][size] += 1
                    cutsbygroup[groups].append(cut)
                    checked += 1
            assert set(counts)==set(direct)
            for cut,groups in catalog['states']:
                assert counts[groups]==tuple(direct[groups][k] for k in range(5))
                for b in range(len(cut),5):
                    total = sum(counts[groups][:b+1])
                    fam = family(graph,cut,groups,catalog,b,total)
                    expected = {removed for removed in cutsbygroup[groups] if len(removed)<=b}
                    reconstructed = set()
                    internal = tuple(e for e in edges(graph) if e not in cut)
                    for size in range(min(b-len(cut),len(internal))+1):
                        for extra in combinations(internal,size):
                            if any(set(bad)<=set(extra) for bad in fam['forbiddenextras']): continue
                            reconstructed.add(tuple(sorted(cut+extra)))
                    assert reconstructed==expected
                    if fam['explicitsets'] is not None: assert set(fam['explicitsets'])==expected
        print(f'PASS exact edge-set counting and compact families: {len(graphs)} small graphs, {checked} exhaustive removal sets')

    def saved():
        records,graphs,catalogs,ranks,digest = analyze()
        bykey = {(r['graph'],r['prior'],r['target'],r['budget']):r for r in records}
        assert len(records)==96
        assert sum(r['multipersonpartitions']>0 for r in records)==3
        assert all(r['multipatientpartitions']==0 for r in records)
        assert bykey['full60','PAT','PAT',3]['optimaledgesets']==111654376
        assert bykey['full60','uniform','all',3]['optimaledgesets']==874
        assert bykey['tue80','uniform','all',4]['optimalpartitions']==7
        assert bykey['tue80','uniform','all',4]['optimaledgesets']==44558
        assert bykey['wed80','uniform','all',4]['optimalpartitions']==6
        assert [bykey['night','uniform','all',b]['optimalpartitions'] for b in range(1,5)]==[6,15,20,15]
        for row in records:
            assert row['optimaledgesets']==sum(row['edgesetsbycardinality'])
            assert row['optimalboundarycuts']==row['optimalpartitions']
            assert sum(opt['edgesetcount'] for opt in row['cooptimal'])==row['optimaledgesets']
            for opt in row['cooptimal']:
                assert sum(c['size'] for c in opt['components'])==len(graphs[row['graph']])
                assert all(sum(c['roles'].values())==c['size'] for c in opt['components'])
                assert all(c['singleton']==(c['size']==1) for c in opt['components'])
            for method,data in row['heuristics'].items():
                assert all(ranks[row['graph']][e][method+'rank']<=row['budget'] for e in data['cut'])
        report = makereport(records,graphs,catalogs,ranks,digest)
        assert report.rsplit('\n\n',1)[1].startswith('Candidate paper findings\n')
        print('PASS all 96 saved cases, co-optimal multiplicities, component roles, rankings and report findings')

    counting()
    saved()
    print('Hospital interpretation tests: Passed 2, Failed 0')

def testwashington():
    import csv

    import json

    from fractions import Fraction

    import networkx as nx

    import washington as pilot

    from hospital import components,score

    def main():
        base = pilot.original()
        before = nx.node_link_data(base)
        cases = pilot.variants(base)
        assert 1<len(cases)<=35 and len({c['graphhash'] for c in cases})==len(cases)
        for case in cases:
            graph = case['graph']
            again = pilot.perturb(base,case['feature'],case['level'],case['seeds'][0])
            assert pilot.signature(again)==case['graphhash']
            pilot.validate(base,graph,case['feature'],case['level'])
            p,a = pilot.air.objectives(graph,'uniform','airports')
            assert score(graph,pilot.boundary(graph,pilot.R),p,a)==Fraction(7812,443)
            assert sum(d.get('synthetic',False) for u,v,d in graph.edges(data=True))==len(case['added'])
        assert nx.node_link_data(base)==before
        print(f'PASS construction: {len(cases)} unique deterministic graphs, controlled edge changes, fixed vertices and connected regions/complements')
        tiny = nx.path_graph(5)
        exact = pilot.solve(tiny,1)
        assert exact['status']=='exact' and exact['values'][1][0]==Fraction(12,5)
        capped = pilot.solve(tiny,1,work=0)
        assert capped['status']=='unverified' and not capped['values'] and capped['error']
        order = pilot.air.rankings(pilot.adapted(tiny))['bc']
        assert order==((1,2),(2,3),(0,1),(3,4))
        print('PASS reused exact solver, explicit unverified limit handling and canonical BC ties')
        path = pilot.folder/'washington.csv'
        if not path.exists():
            print('Saved-result checks deferred until the pilot has run')
            return
        with path.open(newline='') as stream: rows = list(csv.DictReader(stream))
        lookup = {case['variant']:case for case in cases}
        assert {row['variant'] for row in rows}==set(lookup)
        assert len(rows)==sum(2 if case['feature']=='boundary' else 1 for case in cases)
        for row in rows:
            case = lookup[row['variant']]
            graph = case['graph']
            assert row['graphhash']==case['graphhash']
            p,a = pilot.air.objectives(graph,'uniform','airports')
            value = score(graph,json.loads(row['bccuts']),p,a)
            assert value==Fraction(row['bcscore']) and len(json.loads(row['bccuts']))==int(row['budget'])
            assert json.loads(row['added'])==list(map(list,case['added'])) and json.loads(row['removed'])==list(map(list,case['removed']))
            if row['exactstatus']=='exact':
                cuts = json.loads(row['optimalcut'])
                exact = Fraction(row['exact'])
                assert len(cuts)<=int(row['budget']) and score(graph,cuts,p,a)==exact>=value
                groups = sorted(components(graph,cuts),key=lambda c:(-len(c),c))
                assert json.loads(row['optimalpartition'])==list(map(list,groups))
                assert Fraction(row['bcratio'])==value/exact
                if row['candidatefeasible']=='True': assert Fraction(row['candidatescore'])<=exact
                assert (row['candidateoptimal']=='True')==(row['candidatefeasible']=='True' and Fraction(row['candidatescore'])==exact)
            else:
                assert row['exactstatus']=='unverified' and not row['exact'] and not row['bcratio'] and row['exacterror']
        anchor = next(row for row in rows if row['feature']=='original')
        with (pilot.root/'data/results/airresults.csv').open(newline='') as stream:
            saved = next(row for row in csv.DictReader(stream) if row['prior']=='uniform' and row['target']=='airports' and row['budget']=='3')
        assert anchor['exactstatus']=='exact' and anchor['exact']==saved['exact'] and anchor['bcscore']==saved['bcscore']
        assert json.loads(anchor['optimalcut'])==json.loads(saved['optimalcut'])
        assert json.loads(anchor['bccuts'])==json.loads(saved['bccuts'])
        assert json.loads(anchor['rboundaryranks'])==[1,223,210]
        print(f'PASS saved pilot: {len(rows)} cut/partition replays, feasible-candidate bounds, baseline scores and original-result anchor')

    main()

def testfigure9():
    import csv

    import json

    import re

    from fractions import Fraction

    from pathlib import Path

    import io

    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'figures'))
    import fig9 as figure9

    import records as savedreports

    import hashlib

    from unittest.mock import patch

    folder = Path(__file__).resolve().parent.parent/'figures'

    regenerated = figure9.extract()

    stream = io.StringIO(newline='')

    writer = csv.DictWriter(stream,fieldnames=list(regenerated[0]))

    writer.writeheader()

    writer.writerows(regenerated)

    assert hashlib.sha256(stream.getvalue().encode()).hexdigest()==savedreports.load()['derived']['fig9data.csv']

    rows = list(csv.DictReader(io.StringIO(stream.getvalue())))

    assert len(rows)==44 and len({r['graphhash'] for r in rows})==34

    assert len({(r['variant'],r['budget']) for r in rows})==44

    assert sum(r['status']=='measured' for r in rows)==1

    assert sum(Fraction(r['bc'])>0 for r in rows)==4

    assert sum(Fraction(r['ratio'])==1 for r in rows)==1

    assert sum(r['roptimal']=='True' for r in rows)==28

    assert sum(r['uoptimal']=='True' for r in rows)==4

    assert sum(r['representative']=='other' for r in rows)==12

    assert sum(r['rfeasible']=='False' for r in rows)==6

    assert sum(r['ufeasible']=='False' for r in rows)==40

    for row in rows:
        assert Fraction(row['bc'])/Fraction(row['exact'])==Fraction(row['ratio'])
        for key in ('r','u'):
            if row[key+'feasible']=='False':
                assert row[key+'score']==row[key+'optimal']=='infeasible' and row[key+'decimal']==''
            else:
                assert Fraction(row[key+'score'])<=Fraction(row['exact'])
                assert (row[key+'optimal']=='True')==(Fraction(row[key+'score'])==Fraction(row['exact']))
        assert row['anotheroptimalcut']==('yes: saved representative' if row['representative']=='other' else 'unverified: no co-optimal enumeration')

    tex = (folder/'fig9.tex').read_text()
    assert figure9.figure(regenerated)==tex
    assert hashlib.sha256(tex.encode()).hexdigest()=='a7e44dd5d4fb800c37e7583b202b7290fd1a0603165dcb2ff6a91961095e625d'
    assert '%' not in tex and '\\begin{axis}' not in tex
    assert 'dashed' not in tex and '40 of 44' not in tex and 'BC/OPT' not in tex
    for label in ('Fixed Budget','Matching Budget','Baseline','Expected Protected Airports'):
        assert '{'+label+'}' in tex
    blocks = re.split(r'(?=\\node\[anchor=west,font=\\fontsize\{9\}\{10\}\\selectfont\\bfseries\])',tex)[1:]
    assert len(blocks)==3
    represented = set()
    for panel,(feature,anchor,low,high,bottom) in enumerate((('internal',30,9,36,8.05),('boundary',3,1,6,5),('competing',6,3,9,1.95))):
        body = blocks[panel]
        assert body.count('\\draw[black!75,line width=0.55pt]')==1
        curves = re.findall(r'\\draw\[(RoyalBlue|ForestGreen),line width=0.75pt\] (.*?);',body)
        assert len(curves)==(4 if panel==1 else 2)
        for index,(color,curve) in enumerate(curves):
            seed = index%2
            matching = color=='ForestGreen'
            plotted = re.findall(r'\(([^,]+),([^\)]+)\)',curve)
            subset = [r for r in rows if r['family']==feature and seed in json.loads(r['seeds'])
                      and int(r['budget'])==(int(r['rboundarysize']) if matching else 3)]
            subset += [next(r for r in rows if r['family']=='original')]
            subset.sort(key=lambda r:anchor if r['family']=='original' else int(r['level']))
            assert len(plotted)==len(subset)
            for (px,py),row in zip(plotted,subset):
                level = anchor if row['family']=='original' else int(row['level'])
                x = 1.36+6.04*(level-low)/(high-low)
                y = bottom+1.32*float(Fraction(row['exact']))/27
                assert px==f'{x:.4f}' and py==f'{y:.4f}',(row['variant'],row['budget'],seed)
                represented.add((row['variant'],row['budget']))
                if row['family']=='original': continue
                if seed==0:
                    mark = f'\\fill[{color}] ({x:.4f},{y:.4f}) circle (1.25pt);'
                else:
                    mark = f'\\draw[{color},fill=white,line width=0.75pt] ({x-0.042:.4f},{y-0.042:.4f}) rectangle ({x+0.042:.4f},{y+0.042:.4f});'
                assert mark in body,(row['variant'],row['budget'],seed)
        baseline = next(r for r in rows if r['family']=='original')
        x = 1.36+6.04*(anchor-low)/(high-low)
        y = bottom+1.32*float(Fraction(baseline['exact']))/27
        mark = f'\\fill[black] ({x-0.095:.4f},{y:.4f})--({x:.4f},{y+0.095:.4f})--({x+0.095:.4f},{y:.4f})--({x:.4f},{y-0.095:.4f})--cycle;'
        assert body.count('\\fill[black]')==1 and mark in body
    assert represented=={(r['variant'],r['budget']) for r in rows}
    assert len(represented)==44
    print('PASS all 44 cases in three exact-protection plots, two seeds, fixed/matching budgets and three measured markers')
    print('PASS exact fractions, BC scores, feasible-only candidates, optimality classifications and co-optimality caveat')
    print('PASS final reference bytes, solid lines, Title Case legend and no obsolete BC annotation')
    import copy,tempfile
    changed = copy.deepcopy(regenerated)
    changed[1]['exact'] = str(Fraction(changed[1]['exact'])+1)
    assert figure9.figure(changed)!=tex
    before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder.parent/'data').rglob('*') if p.is_file()}
    with patch.object(figure9,'extract',return_value=regenerated),patch.object(sys,'argv',['fig9.py','--check']), \
         patch.object(Path,'write_bytes',side_effect=AssertionError('Verification wrote a file')), \
         patch.object(Path,'write_text',side_effect=AssertionError('Verification wrote a file')), \
         patch.object(savedreports,'save',side_effect=AssertionError('Verification changed provenance')):
        figure9.main()
    with tempfile.TemporaryDirectory() as directory:
        invalid = Path(directory)/'fig9.tex'
        invalid.write_text(tex+'changed')
        with patch.object(figure9,'folder',Path(directory)),patch.object(figure9,'extract',return_value=regenerated), \
             patch.object(sys,'argv',['fig9.py','--check']),patch.object(Path,'write_bytes',side_effect=AssertionError('Verification overwrote a mismatch')):
            try: figure9.main()
            except SystemExit as error: assert str(error).startswith('FAIL Figure 9')
            else: raise AssertionError('Mismatching Figure 9 was accepted')
        assert invalid.read_text()==tex+'changed'
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in before.items())
    print('PASS data-driven figure, non-writing verification, mismatch rejection, and immutable saved results/provenance')


def testvalley():
    import csv

    import json

    from fractions import Fraction

    from itertools import combinations

    import networkx as nx

    import valley as pilot

    def construction():
        base = pilot.original()
        before = nx.node_link_data(base)
        cases = pilot.variants(base)
        assert 20<=len(cases)<=30 and len({c['graphhash'] for c in cases})==len(cases)
        assert sum(len(c['aliases']) for c in cases)==29
        assert {a['level'] for c in cases for a in c['aliases'] if a['feature']=='internal'}==set(pilot.levels)
        assert {a['level'] for c in cases for a in c['aliases'] if a['feature']=='boundary'}=={1,3,4}
        for case in cases:
            graph = pilot.perturb(base,case['feature'],case['level'],case['seed'])
            pilot.validate(base,graph,case['feature'],case['level'])
            assert pilot.signature(graph)==case['graphhash']
            for alias in case['aliases']:
                again = pilot.perturb(base,alias['feature'],alias['level'],alias['seed'])
                assert pilot.signature(again)==case['graphhash']
            assert set(pilot.edges(graph))-set(pilot.edges(base))==set(case['added'])
            assert set(pilot.edges(base))-set(pilot.edges(graph))==set(case['removed'])
            assert sum(d.get('synthetic',False) for u,v,d in graph.edges(data=True))==len(case['added'])
            assert all(graph.nodes[v]==base.nodes[v] for v in graph)
            value,groups = pilot.protection(graph,pilot.commuting.boundary(graph,pilot.R))
            assert value==Fraction(37188,3105) and list(map(len,groups))==[3099,6]
            if case['feature']=='placement':
                cuts = pilot.commuting.boundary(graph,pilot.R)
                nr = len({v for e in cuts for v in e if v in pilot.R})
                no = len({v for e in cuts for v in e if v not in pilot.R})
                assert (nr,no)=={'distinct':(2,2),'region':(1,2),'outside':(2,1),'alternate':(2,2)}[case['level']]
        assert nx.node_link_data(base)==before
        print(f'PASS construction: {len(cases)} distinct graphs, 29 parameter/seed specifications, exact controlled changes and deterministic aliases')
        print('PASS baseline: 3105 vertices, 24338 edges, six counties, eight internal edges, specified two-edge boundary; fixed outside graph and connected regions')
        return base,cases

    def small():
        graphs = [nx.path_graph(5),nx.star_graph(5),nx.cycle_graph(6),nx.complete_graph(4),nx.barbell_graph(3,1),nx.complete_bipartite_graph(2,3)]
        checks = 0
        for graph in graphs:
            result = pilot.solve(graph,3)
            assert result['status']=='exact'
            for b in range(4):
                brute = max(pilot.protection(graph,cut)[0] for j in range(b+1) for cut in combinations(pilot.edges(graph),j))
                value,cuts = result['values'][b]
                assert value==brute==pilot.protection(graph,cuts)[0] and len(cuts)<=b
                checks += 1
            nx.set_edge_attributes(graph,1,'flow')
            order,values = pilot.bcorder(graph)
            reference,bc,bcw = pilot.commuting.rankings(graph)
            assert order==reference['bc'] and values==bc
        path = nx.path_graph(5)
        order,values = pilot.bcorder(path)
        assert order[:2]==((1,2),(2,3)) and values[(1,2)]==values[(2,3)]==12
        stopped = pilot.solve(path,2,work=0)
        assert stopped['status']=='unverified' and not stopped['values'] and stopped['error']
        print(f'PASS exact solver: {checks} brute-force small-graph comparisons; explicit search-limit failure; ordered-pair BC and canonical ties match commuting.rankings')

    def saved(base,cases):
        path = pilot.folder/'valley.csv'
        if not path.exists():
            print('Saved-result checks deferred until pilot completion')
            return
        with path.open(newline='') as stream: rows = list(csv.DictReader(stream))
        expected = sum(2 if c['feature']=='boundary' else 1 for c in cases)
        assert len(rows)==expected and len({(r['variant'],r['budget']) for r in rows})==expected
        assert {r['variant'] for r in rows}=={c['variant'] for c in cases}
        byname = {c['variant']:c for c in cases}
        ranksbygraph = {}
        for row in rows:
            case = byname[row['variant']]
            graph = pilot.perturb(base,case['feature'],case['level'],case['seed'])
            b = int(row['budget'])
            assert (int(row['n']),int(row['m']))==(len(graph),graph.number_of_edges())
            assert row['graphhash']==pilot.signature(graph)
            for key in ('added','removed','aliases'):
                assert json.loads(row[key])==json.loads(json.dumps(case[key]))
            boundary = pilot.commuting.boundary(graph,pilot.R)
            assert json.loads(row['rboundary'])==list(map(list,boundary)) and len(boundary)==int(row['rboundarysize'])
            assert int(row['rinternaledges'])==graph.subgraph(pilot.R).number_of_edges()
            assert Fraction(row['rdensity'])==Fraction(int(row['rinternaledges']),15)
            assert all(row[k]=='True' for k in ('graphconnected','rconnected','rcomplementconnected'))
            chosen = tuple(map(tuple,json.loads(row['bccuts'])))
            value,groups = pilot.protection(graph,chosen)
            assert len(chosen)==b and value==Fraction(row['bcscore'])
            assert list(map(len,groups))==json.loads(row['bccomponents'])
            assert json.loads(row['bccutranks'])==list(range(1,b+1))
            assert int(row['bcboundaryselected'])==sum(e in boundary for e in chosen)
            ranks = json.loads(row['rboundaryranks'])
            assert len(ranks)==len(boundary) and len(set(ranks))==len(ranks)
            for edge,rank in zip(boundary,ranks):
                assert (edge in chosen)==(rank<=b)
                if rank<=b: assert chosen[rank-1]==edge
            if row['variant'] in ranksbygraph:
                oldchosen,oldranks = ranksbygraph[row['variant']]
                assert chosen[:len(oldchosen)]==oldchosen and ranks==oldranks
            ranksbygraph[row['variant']] = chosen,ranks
            feasible = len(boundary)<=b
            assert (row['candidatefeasible']=='True')==feasible
            if feasible: assert Fraction(row['candidatescore'])==Fraction(37188,3105)
            else: assert row['candidatescore']==row['candidateoptimal']=='infeasible' and not row['candidatedecimal']
            if row['exactstatus']=='exact':
                cuts = tuple(map(tuple,json.loads(row['optimalcut'])))
                exact,groups = pilot.protection(graph,cuts)
                assert exact==Fraction(row['exact'])>=value and len(cuts)<=b
                assert json.loads(row['optimalpartition'])==list(map(list,groups))
                assert json.loads(row['optimalcomponents'])==list(map(len,groups))
                assert Fraction(row['bcratio'])==value/exact
                assert (row['bcoptimal']=='True')==(value==exact)
                assert (row['rinoptimal']=='True')==any(set(c)==pilot.R for c in groups)
                if feasible:
                    assert exact>=Fraction(row['candidatescore'])
                    assert (row['candidateoptimal']=='True')==(exact==Fraction(row['candidatescore']))
                    assert (row['anotherbetter']=='True')==(exact>Fraction(row['candidatescore']))
                for item in json.loads(row['otherregionbetter']):
                    assert pilot.protection(graph,item['boundary'])[0]==Fraction(item['score'])>Fraction(37188,3105)
                    assert len(item['boundary'])<=b
            else:
                assert row['exactstatus']=='unverified' and row['exacterror'] and not row['exact'] and not row['bcratio'] and row['optimalpartition']=='[]'
        baseline = next(r for r in rows if r['feature']=='original')
        with (pilot.root/'data/results/commutingresults.csv').open(newline='') as stream:
            anchor = next(r for r in csv.DictReader(stream) if r['prior']=='uniform' and r['target']=='counties' and r['budget']=='2')
        assert baseline['bcscore']==anchor['bcscore'] and json.loads(baseline['bccuts'])==json.loads(anchor['bccuts'])
        details = {tuple(d['edge']):d for d in json.loads(anchor['removed'])}
        assert json.loads(baseline['rboundaryranks'])==[details[e]['ranks']['bc'] for e in pilot.originalcuts]
        if baseline['exactstatus']=='exact':
            assert baseline['exact']==anchor['exact'] and json.loads(baseline['optimalcut'])==json.loads(anchor['optimalcut'])
        print(f'PASS saved results: {len(rows)} cut/partition replays, exact component arithmetic, feasible-candidate handling, fixed-rank budgets and unchanged baseline anchor')

    base,cases = construction()
    small()
    saved(base,cases)

def testrecords():
    import hashlib,importlib,json,sys,tempfile
    import records as savedreports
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root/'records.json'
        content = 'Saved report\nUnicode: é\n'
        digest = hashlib.sha256(content.encode()).hexdigest()
        source = '{"file":"input.csv","url":"https://example.org/input.csv"}\n'
        sourcehash = hashlib.sha256(source.encode()).hexdigest()
        path.write_text(json.dumps(dict(reports={'fixture':content},reporthashes={'fixture':digest},datasets={'fixture':source},
                                        datasethashes={'fixture':sourcehash},artifacts={},sources=[],paths={'old.txt':'report:fixture','old.json':'source:fixture'})))
        with patch.object(savedreports,'root',root),patch.object(savedreports,'path',path):
            assert savedreports.text('fixture')==content
            assert savedreports.data('old.txt')==content.encode()
            savedreports.verify('old.txt',digest)
            assert savedreports.data('old.json')==source.encode()
            savedreports.verify('source:fixture',sourcehash)
            assert savedreports.check()==2
            try: savedreports.save('fixture','replacement',exclusive=True)
            except FileExistsError: pass
            else: raise AssertionError('Existing report was overwritten')
            assert savedreports.text('fixture')==content
            savedreports.save('new','New report',exclusive=True)
            assert savedreports.exists('new') and savedreports.text('new')=='New report'
            assert savedreports.text('fixture')==content
    print('PASS consolidated report text, hashes, historical paths and overwrite protection')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root/'records.json'
        content = json.dumps(dict(reports={},reporthashes={},artifacts={'figures/fig7.tex':'old'},
                                  sources=[dict(current='figures/fig7.py',currentsha256='old')]))
        path.write_text(content)
        names = {'code/records.py','code/tests.py'}|{f'figures/fig{n}.{suffix}' for n in (7,8,9) for suffix in ('py','tex')}
        hashes = {}
        for name in names:
            item = root/name
            item.parent.mkdir(parents=True,exist_ok=True)
            item.write_text('current '+name)
            hashes[name] = hashlib.sha256(item.read_bytes()).hexdigest()
        manifest = dict(records=hashlib.sha256(path.read_bytes()).hexdigest(),files=hashes,
                        reference={name:digest for name,digest in hashes.items() if name.endswith('.tex')})
        revisions = root/'figures/checks.json'
        revisions.write_text(json.dumps(manifest))
        with patch.object(savedreports,'root',root),patch.object(savedreports,'path',path):
            assert savedreports.check()==1
            for name in ('figures/fig7.tex','figures/fig7.py','code/tests.py','records.json'):
                item = root/name
                old = item.read_bytes()
                item.write_bytes(old+b'\n')
                try: savedreports.check()
                except AssertionError: pass
                else: raise AssertionError('Modified figure/source/provenance was accepted: '+name)
                item.write_bytes(old)
            manifest['files']['data/raw/extra.csv'] = 'not authorized'
            revisions.write_text(json.dumps(manifest))
            try: savedreports.check()
            except AssertionError: pass
            else: raise AssertionError('Figure revision tried to override an experimental input')
            del manifest['files']['data/raw/extra.csv']
            revisions.write_text(json.dumps(manifest))
            item = root/'code/tests.py'
            item.write_text('safety revision')
            update = dict(records=hashlib.sha256(path.read_bytes()).hexdigest(),
                          figures=hashlib.sha256(revisions.read_bytes()).hexdigest(),
                          files={'code/tests.py':dict(previous=hashes['code/tests.py'],current=hashlib.sha256(item.read_bytes()).hexdigest())})
            safety = root/'code/checks.json'
            safety.write_text(json.dumps(update))
            assert savedreports.check()==1
            item.write_text('tampered revision')
            try: savedreports.check()
            except AssertionError: pass
            else: raise AssertionError('Modified safety source was accepted')
            item.write_text('safety revision')
            for name in ('data/results/extra.csv','figures/fig7.tex','code/../data/extra.py'):
                update['files'][name] = dict(previous='old',current='new')
                safety.write_text(json.dumps(update))
                try: savedreports.check()
                except AssertionError: pass
                else: raise AssertionError('Safety revision tried to override non-code artifacts')
                del update['files'][name]
            update['files']['code/tests.py']['previous'] = 'wrong predecessor'
            safety.write_text(json.dumps(update))
            try: savedreports.check()
            except AssertionError: pass
            else: raise AssertionError('Unverified safety revision predecessor was accepted')
    print('PASS separate figure revision checks reject changed figures, generators, tests, historical records and input overrides')
    print('PASS source safety revisions preserve historical checks and reject tampering or non-code overrides')
    for name,entry in [('baselines','main'),('randomgraphs','main'),('transmission','main'),('population','main'),('livestock','main'),('washington','main'),('valley','main'),('livestock','directed')]:
        module = importlib.import_module(name)
        with patch.object(sys,'argv',[name+'.py']):
            try: getattr(module,entry)()
            except FileExistsError: pass
            else: raise AssertionError('Saved results were not protected: '+name)
    print('PASS all eight study entry points refuse to overwrite existing results')
    import subprocess
    root = Path(__file__).resolve().parent.parent
    for name,mode in [('hospital','--interpret'),('hospital','--cohorts'),('commuting','--interpret'),('competition','--theory'),('livestock','--directed')]:
        result = subprocess.run([sys.executable,'-B',str(root/'code'/(name+'.py')),mode,'--help'],capture_output=True,text=True)
        assert result.returncode==0 and mode in result.stdout,(name,mode,result.stdout,result.stderr)
    print('PASS every consolidated analysis mode exposes help without running an experiment')

def testsafety():
    import contextlib,hashlib,importlib,io,json,sys,tempfile
    import records as savedreports
    from unittest.mock import patch
    root = Path(__file__).resolve().parent.parent
    def snapshot():
        return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in ('data','figures') for p in (root/folder).rglob('*') if p.is_file()}
    def blocked(action):
        try: action()
        except FileExistsError as error: assert str(error)
        else: raise AssertionError('Existing output was not protected')
    before = snapshot()
    entries = [('air','main',('airresults.csv',),'air'),
               ('commuting','main',('commutingresults.csv',),'commuting'),
               ('commuting','interpret',('commutinginterpret.csv',),'commutinginterpret'),
               ('hospital','main',('hospitalresults.csv',),'hospital'),
               ('hospital','interpret',('hospitalinterpret.csv',),'hospitalinterpret'),
               ('hospital','cohorts',('hospitalcohorts.csv',),'hospitalcohorts'),
               ('highschool','main',tuple('highschool'+s+'.csv' for s in ('edges','thresholds','static','sequential')),'highschool'),
               ('competition','main',('competition.csv',),'competition'),
               ('competition','theory',(),'competitiontheory'),
               ('applicationsweeps','main',('commutingsweep.csv','airsweep.csv'),'sweeps')]
    protected = [(name,entry) for name,entry,files,report in entries]
    protected += [(name,'main') for name in ('baselines','randomgraphs','population','transmission','livestock','washington','valley')]
    protected += [('livestock','directed')]
    for name,entry in protected:
        module = importlib.import_module(name)
        with patch.object(sys,'argv',[name+'.py']), \
             patch.object(module,'perf_counter',side_effect=AssertionError('Computation started before output checks')), \
             patch.object(Path,'read_bytes',side_effect=AssertionError('Input loaded before output checks')):
            blocked(getattr(module,entry))
    print(f'PASS {len(protected)} default experiment/report entry points reject overwrites before input loading or computation')
    for name,entry,files,report in entries:
        module = importlib.import_module(name)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            for collision in (*files,report+'.txt'):
                item = destination/collision
                item.write_bytes(b'unchanged output')
                with patch.object(sys,'argv',[name+'.py','--output',directory]), \
                     patch.object(module,'perf_counter',side_effect=AssertionError('Computation started before output checks')), \
                     patch.object(Path,'read_bytes',side_effect=AssertionError('Input loaded before output checks')):
                    blocked(getattr(module,entry))
                assert sorted(p.name for p in destination.iterdir())==[collision]
                assert item.read_bytes()==b'unchanged output'
                item.unlink()
        with patch.object(sys,'argv',[name+'.py','--output',str(root/'data/results')]):
            if files: blocked(getattr(module,entry))
    print('PASS all CSV/report destination collisions are checked before computation, with no partial output files')
    import applicationsweeps,competition,cyclesearch,highschool,unicyclelemmas
    published = root/'data/results/competition.csv'
    for action in (lambda: competition.write([],published),lambda: applicationsweeps.writecsv(published,[]),
                   lambda: highschool.writecsv(published,[])):
        blocked(action)
    for module in (cyclesearch,unicyclelemmas):
        with patch.object(module,'perf_counter',side_effect=AssertionError('Search started before output checks')):
            blocked(lambda: module.search(output=published))
    with patch.object(sys,'argv',['highschool.py','--plot']): blocked(highschool.main)
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        (destination/'ratios.png').write_bytes(b'existing plot')
        with patch.object(sys,'argv',['highschool.py','--plot','--output',directory]): blocked(highschool.main)
        assert [p.name for p in destination.iterdir()]==['ratios.png']
        assert (destination/'ratios.png').read_bytes()==b'existing plot'
    print('PASS low-level CSV writers, exploratory searches and optional plots protect existing outputs')
    for exclusive in (False,True):
        blocked(lambda: savedreports.save('air','replacement',exclusive=exclusive))
    blocked(lambda: savedreports.save('new-safety-report','new report'))
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        link = destination/'link.csv'
        link.symlink_to(destination/'missing.csv')
        blocked(lambda: savedreports.guard(link))
        alias = destination/'published'
        alias.symlink_to(root/'data/results',target_is_directory=True)
        blocked(lambda: savedreports.prepare(alias,('competition.csv',)))
        fresh = savedreports.prepare(destination/'fresh',('competition.csv',),('competition',))
        competition.write([dict(case='fixture',value='3/2')],fresh/'competition.csv')
        savedreports.save('competition','Independent report',directory=fresh)
        assert (fresh/'competition.txt').read_text()=='Independent report'
        blocked(lambda: competition.write([],fresh/'competition.csv'))
        blocked(lambda: savedreports.save('competition','replacement',directory=fresh))
        race = savedreports.prepare(destination/'race',('rows.csv',))/'rows.csv'
        race.write_bytes(b'created after preflight')
        blocked(lambda: applicationsweeps.writecsv(race,[dict(value=0)]))
        assert race.read_bytes()==b'created after preflight'
        with patch.object(sys,'argv',['competition.py','--output',str(destination/'regenerated')]), \
             patch.object(competition,'experiment',return_value=([dict(case='fixture',value='3/2')],[])) as compute, \
             patch.object(competition,'report',return_value='Separate report'),contextlib.redirect_stdout(io.StringIO()):
            competition.main()
            assert compute.call_count==1
        assert (destination/'regenerated/competition.txt').read_text()=='Separate report'
        assert '3/2' in (destination/'regenerated/competition.csv').read_text()
        for folder in (root/'data',root/'code',root/'figures',alias):
            try: savedreports.cachefolder(folder)
            except ValueError: pass
            else: raise AssertionError('Cache was allowed inside the repository')
        assert savedreports.cachefolder(destination/'cache')==(destination/'cache').resolve()
    print('PASS separate regeneration, report exclusivity, late collisions, symlinks and cache isolation')
    assert snapshot()==before
    print(f'PASS all {len(before)} data and figure files remain byte-for-byte unchanged after failed writes and isolated regeneration')

CASES = {
    'core':main,
    'air':testair,
    'applicationsweeps':testapplicationsweeps,
    'commuting':testcommuting,
    'commutinginterpret':testcommutinginterpret,
    'competition':testcompetition,
    'competitiontheory':testcompetitiontheory,
    'figures':testfigures,
    'highschool':testhighschool,
    'hospital':testhospital,
    'hospitalcohorts':testhospitalcohorts,
    'hospitalinterpret':testhospitalinterpret,
    'washington':testwashington,
    'figure9':testfigure9,
    'valley':testvalley,
    'records':testrecords,
    'safety':testsafety,
}

def suite():
    import os,subprocess,sys
    import records as savedreports
    root = Path(__file__).resolve().parent.parent
    start = perf_counter()
    print(f'PASS {savedreports.check()} current artifacts and preserved reports',flush=True)
    commands = [('code/tests.py',name) for name in CASES]
    commands += [(f'code/{name}.py','--check') for name in ('baselines','randomgraphs','population','transmission','livestock')]
    commands += [('code/livestock.py','--directed','--check')]
    passed = failed = 0
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for command in commands:
        tick = perf_counter()
        result = subprocess.run([sys.executable,*command],cwd=root,env=env,capture_output=True,text=True)
        if result.returncode:
            failed += 1
            print('FAIL '+' '.join(command),flush=True)
            print(result.stdout,end='')
            print(result.stderr,end='',file=sys.stderr)
        else:
            passed += 1
            print(f'PASS {" ".join(command)} ({perf_counter()-tick:.2f}s)',flush=True)
    print(f'PASS {savedreports.check()} artifact/report checksums still match',flush=True)
    print(f'Groups: {passed+failed}; Passed: {passed}; Failed: {failed}; Seconds: {perf_counter()-start:.2f}',flush=True)
    return int(failed>0)

if __name__=='__main__':
    import sys
    if len(sys.argv)==1: raise SystemExit(suite())
    if len(sys.argv)==2 and sys.argv[1] in CASES: CASES[sys.argv[1]]()
    else:
        print('Run all regressions: python -B code/tests.py')
        print('Run one group: python -B code/tests.py NAME')
        print('Groups: '+', '.join(CASES))
        print('Environment: Python 3.13.12, networkx==3.7')
        raise SystemExit(0 if sys.argv[1] in ('-h','--help') else 1)
