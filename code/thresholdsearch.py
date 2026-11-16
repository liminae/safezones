# Exact threshold comparisons on small graphs
import argparse
import json
from collections import deque
from fractions import Fraction
from time import perf_counter
from unittest.mock import patch
import networkx as nx
from safezones import edge_order,exact,families,first_spread,simulate,spread

def reference(graph,source,threshold=True):
    nodes = tuple(sorted(graph))
    initial = (first_spread(graph,source),edge_order(graph))
    pending = deque([(initial,())])
    seen = {initial}
    value,actions = -1,()
    while pending:
        (infected,edges),path = pending.popleft()
        if spread(nodes,edges,infected)==infected:
            saved = len(nodes)-len(infected)
            if saved>value:
                value,actions = saved,path
            continue
        eligible = [e for e in edges if
                    ((e[0] in infected)!=(e[1] in infected) if threshold else
                     not (e[0] in infected and e[1] in infected))]
        for action in eligible+[None]:
            remaining = tuple(e for e in edges if e!=action)
            after = spread(nodes,remaining,infected)
            state = (after,remaining)
            if state not in seen:
                seen.add(state)
                pending.append((state,path+(action,)))
    return value,list(actions),len(seen)

def replay(graph,source,actions,threshold=False):
    index = 0
    def choose(policy,edges,infected,*args):
        nonlocal index
        assert index<len(actions)
        action = actions[index]
        index += 1
        if action is not None:
            assert action in edges
            assert not threshold or (action[0] in infected)!=(action[1] in infected)
        return action
    with patch("safezones.choose_edge",side_effect=choose):
        value,cuts,states = simulate(graph,source,"scripted",bc={},bcw={},trace=True)
    assert index==len(actions) and cuts==[e for e in actions if e is not None]
    trace = [dict(round=i,before=sorted(states[i][0]),action=action,
                  after=sorted(states[i+1][0]),added=sorted(states[i+1][0]-states[i][0]))
             for i,action in enumerate(actions,1)]
    return value,sorted(set(graph)-states[-1][0]),trace

def cactus(graph):
    return nx.is_connected(graph) and all(len(block)==2 or
        graph.subgraph(block).number_of_edges()==len(block)
        for block in nx.biconnected_components(graph))

def unique(candidates):
    buckets = {}
    for graph in candidates:
        key = nx.weisfeiler_lehman_graph_hash(graph)
        bucket = buckets.setdefault(key,[])
        if not any(nx.is_isomorphic(graph,other) for other in bucket):
            bucket.append(graph)
    return sorted((graph for bucket in buckets.values() for graph in bucket),
                  key=lambda graph:(graph.number_of_edges(),edge_order(graph)))

def extend(graphs):
    for graph in graphs:
        n = len(graph)
        for mask in range(1,1<<n):
            larger = graph.copy()
            larger.add_edges_from((v,n) for v in range(n) if mask>>v & 1)
            yield larger

def cactusgraphs(limit):
    groups = {1:[nx.empty_graph(1)]}
    for n in range(2,limit+1):
        def candidates():
            for graph in groups[n-1]:
                for v in graph:
                    larger = graph.copy()
                    larger.add_edge(v,n-1)
                    yield larger
            for length in range(3,n+1):
                size = n-length+1
                for graph in groups[size]:
                    for v in graph:
                        larger = graph.copy()
                        nx.add_cycle(larger,[v]+list(range(size,n)))
                        yield larger
        groups[n] = unique(candidates())
        assert all(cactus(graph) for graph in groups[n])
    return groups

def compare(graph,label):
    start = perf_counter()
    rows = []
    maximum = 0
    original = edge_order(graph)
    for source in sorted(graph):
        unrestricted = exact(graph,source)
        threshold,actions,states = reference(graph,source)
        assert threshold==exact(graph,source,threshold_only=True),(label,source)
        assert threshold<=unrestricted
        if nx.is_tree(graph):
            assert threshold==unrestricted,(label,source)
        rows.append((source,unrestricted,threshold))
        maximum = max(maximum,states)
    assert edge_order(graph)==original
    return dict(graph=graph,label=label,n=len(graph),rank=graph.number_of_edges()-len(graph)+1,
                cactus=cactus(graph),rows=rows,seconds=perf_counter()-start,states=maximum)

def summary(results):
    rows = [row for result in results for row in result["rows"]]
    failures = sum(t<u for s,u,t in rows)
    ratios = [Fraction(t,u) for s,u,t in rows if u>0]
    return dict(graphs=len(results),cases=len(rows),failures=failures,
                failinggraphs=sum(any(t<u for s,u,t in result["rows"]) for result in results),
                nontrivial=len(ratios),rate=100*failures/len(rows) if rows else 0,
                worst=str(min(ratios)) if ratios else None)

def minimum(results):
    failing = [(result,source,u,t) for result in results
               for source,u,t in result["rows"] if t<u]
    if not failing:
        return []
    n = min(result["n"] for result,s,u,t in failing)
    return [item for item in failing if item[0]["n"]==n]

def detail(result,source,u,t):
    graph = result["graph"]
    value,actions,states = reference(graph,source,False)
    thvalue,thactions,thstates = reference(graph,source)
    assert value==u and thvalue==t
    saved,safe,trace = replay(graph,source,actions)
    thsaved,thsafe,thtrace = replay(graph,source,thactions,True)
    assert saved==u and thsaved==t
    return dict(label=result["label"],n=len(graph),edges=edge_order(graph),source=source,
                unrestricted=u,threshold=t,actions=actions,thresholdactions=thactions,
                cuts=[e for e in actions if e is not None],
                thresholdcuts=[e for e in thactions if e is not None],
                rank=result["rank"],unicyclic=result["rank"]==1,cactus=result["cactus"],
                bipartite=nx.is_bipartite(graph),degrees=sorted((d for v,d in graph.degree()),reverse=True),
                articulations=sorted(nx.articulation_points(graph)),
                bridges=sorted(tuple(sorted(e)) for e in nx.bridges(graph)),
                first=sorted(first_spread(graph,source)),safe=safe,thresholdsafe=thsafe,
                trace=trace,thresholdtrace=thtrace,states=states,thresholdstates=thstates)

def search(limit=8,classlimit=9):
    start = perf_counter()
    atlas = nx.graph_atlas_g()
    groups = {n:[g for g in atlas if len(g)==n and nx.is_connected(g)] for n in range(3,8)}
    if limit>7:
        generated = perf_counter()
        groups[8] = unique(extend(groups[7]))
        assert len(groups[8])==11117
        print(f"Generated all {len(groups[8])} connected order-8 graphs in {perf_counter()-generated:.6f}s",flush=True)
    assert 3<=limit<=8 and classlimit>=limit
    general = []
    for n in range(3,limit+1):
        for i,graph in enumerate(groups[n]):
            label = graph.name if n<=7 else f"n8:{i}"
            general.append(compare(graph,label))
        print("GENERAL",n,json.dumps(summary([r for r in general if r["n"]==n])),flush=True)
    generated = perf_counter()
    cacti = cactusgraphs(classlimit)
    for n in range(3,limit+1):
        expected = [r["graph"] for r in general if r["n"]==n and r["cactus"]]
        assert len(cacti[n])==len(expected)
        assert len(unique(expected+cacti[n]))==len(expected)
    print(f"Cactus generation and overlap verification: {perf_counter()-generated:.6f}s",flush=True)
    classes = [r for r in general if r["cactus"]]
    for n in range(limit+1,classlimit+1):
        for i,graph in enumerate(cacti[n]):
            classes.append(compare(graph,f"cactus{n}:{i}"))
    for n in range(3,classlimit+1):
        print("UNICYCLIC",n,json.dumps(summary([r for r in classes if r["n"]==n and r["rank"]==1])))
        print("CACTUS",n,json.dumps(summary([r for r in classes if r["n"]==n])))
    print("GENERAL TOTAL",json.dumps(summary(general)))
    print("UNICYCLIC TOTAL",json.dumps(summary([r for r in classes if r["rank"]==1])))
    print("CACTUS TOTAL",json.dumps(summary(classes)))
    print("CYCLIC CACTUS TOTAL",json.dumps(summary([r for r in classes if r["rank"]>0])))
    print(f"CYCLE RANK: complete connected graphs, 3<=n<={limit}; rate uses all source cases")
    for rank in range(4):
        print("RANK",rank,json.dumps(summary([r for r in general if r["rank"]==rank])))
    for name,results in (("GENERAL",general),("UNICYCLIC",[r for r in classes if r["rank"]==1]),
                         ("CACTUS",classes)):
        cases = minimum(results)
        print("MINIMUM",name,"cases",len(cases))
        for result,source,u,t in cases:
            print(json.dumps(detail(result,source,u,t),sort_keys=True))
    trees = [r for r in classes if r["rank"]==0]
    print("EXHAUSTIVE TREES",json.dumps(summary(trees)))
    current = [compare(graph,f"{family}:{label}") for family,graphs in families()
               for label,graph in graphs if nx.is_tree(graph)]
    print("CURRENT TREES",json.dumps(summary(current)))
    slowest = max(general+classes+current,key=lambda r:r["seconds"])
    print("SLOWEST GRAPH",slowest["label"],f"{slowest['seconds']:.6f}s")
    print("MAXIMUM INDEPENDENT THRESHOLD STATES",max(r["states"] for r in general+classes+current))
    print(f"TOTAL SECONDS {perf_counter()-start:.6f}")
    return general,classes

def regression():
    graph = nx.Graph([(0,2),(0,3),(0,4),(0,5),(1,2),(1,3)])
    original = graph.copy()
    for threshold,target in ((False,2),(True,1)):
        value,actions,states = reference(graph,1,threshold)
        assert value==target==exact(graph,1,threshold_only=threshold)
        saved,safe,trace = replay(graph,1,actions,threshold)
        assert saved==target
    assert cactus(graph) and graph.number_of_edges()==len(graph)
    assert nx.utils.graphs_equal(graph,original)
    atlas = nx.graph_atlas_g()
    for graph in atlas:
        if 3<=len(graph)<=5 and nx.is_connected(graph):
            for source in graph:
                assert reference(graph,source)[0]==exact(graph,source)
    for index,source,u,t in ((126,0,2,1),(154,0,1,0)):
        graph = atlas[index]
        for threshold,target in ((False,u),(True,t)):
            value,actions,states = reference(graph,source,threshold)
            assert value==target==exact(graph,source,threshold_only=threshold)
            assert replay(graph,source,actions,threshold)[0]==value
    graph = nx.path_graph(4)
    assert replay(graph,0,[None,(2,3)],True)[0]==1
    for family,graphs in families():
        for label,graph in graphs:
            if nx.is_tree(graph):
                for source in graph:
                    value,actions,states = reference(graph,source)
                    assert value==exact(graph,source)==exact(graph,source,threshold_only=True)
                    assert replay(graph,source,actions,True)[0]==value
    small = cactusgraphs(6)
    for n in range(3,7):
        expected = [g for g in nx.graph_atlas_g() if len(g)==n and nx.is_connected(g) and cactus(g)]
        assert len(small[n])==len(expected)==len(unique(small[n]+expected))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--general",type=int,default=8,choices=range(3,9))
    parser.add_argument("--classes",type=int,default=9)
    args = parser.parse_args()
    regression()
    search(args.general,args.classes)
