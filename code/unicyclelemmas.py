# Structural checks for unicyclic threshold strategies
import csv
import json
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from random import Random
from time import perf_counter
from unittest.mock import patch
import networkx as nx
from safezones import edge_order,first_spread,simulate
from cyclesearch import oracle
from thresholdsearch import replay

def weighted(graph,weights,threshold=False):
    nodes = sorted(graph)
    bits = {v:1<<i for i,v in enumerate(nodes)}
    allnodes = (1<<len(nodes))-1
    edges = edge_order(graph)
    ends = [(bits[u],bits[v]) for u,v in edges]
    full = (1<<len(edges))-1
    choice = {}
    def advance(infected,remaining):
        after = infected
        while remaining:
            bit = remaining & -remaining
            u,v = ends[bit.bit_length()-1]
            if infected & u: after |= v
            if infected & v: after |= u
            remaining ^= bit
        return after
    def trim(infected,remaining):
        reached = infected
        while True:
            after = advance(reached,remaining)
            if after==reached: break
            reached = after
        active = 0
        while remaining:
            bit = remaining & -remaining
            u,v = ends[bit.bit_length()-1]
            if reached & (u|v) and infected & (u|v)!=(u|v): active |= bit
            remaining ^= bit
        return active
    @lru_cache(None)
    def best(infected,remaining):
        if advance(infected,remaining)==infected:
            safe = allnodes ^ infected
            return sum(weights[v] for v in nodes if safe & bits[v]),0,0,safe,safe
        candidates = [0]
        pending = remaining
        while pending:
            bit = pending & -pending
            u,v = ends[bit.bit_length()-1]
            if not threshold or bool(infected & u)!=bool(infected & v): candidates.append(bit)
            pending ^= bit
        result = (-1,0,0)
        possible,necessary = 0,allnodes
        for action in candidates:
            following = remaining ^ action
            after = advance(infected,following)
            value,proactive,cuts,union,intersection = best(after,trim(after,following))
            if action:
                u,v = ends[action.bit_length()-1]
                proactive += not bool(infected & (u|v))
                cuts += 1
            if value>result[0]: possible,necessary = union,intersection
            elif value==result[0]: possible,necessary = possible|union,necessary&intersection
            if (-value,proactive,cuts)<(-result[0],result[1],result[2]):
                result = value,proactive,cuts
                choice[infected,remaining] = action
        return (*result,possible,necessary)
    def solve(source):
        infected = sum(bits[v] for v in first_spread(graph,source))
        remaining = trim(infected,full)
        value,proactive,cuts,possible,necessary = best(infected,remaining)
        actions = []
        while advance(infected,remaining)!=infected:
            action = choice[infected,remaining]
            actions.append(edges[action.bit_length()-1] if action else None)
            remaining ^= action
            infected = advance(infected,remaining)
            remaining = trim(infected,remaining)
        return dict(value=value,proactive=proactive,actions=actions,
                    possible=[v for v in nodes if possible & bits[v]],
                    necessary=[v for v in nodes if necessary & bits[v]])
    solve.cache = best.cache_info
    return solve

def controlled(graph,source,chooser):
    actions = []
    def choose(policy,edges,infected,*args):
        action = chooser(edges,infected,len(actions)+1)
        if action is not None:
            assert action in edges and (action[0] in infected)!=(action[1] in infected)
        actions.append(action)
        return action
    with patch("safezones.choose_edge",side_effect=choose):
        value,cuts,states = simulate(graph,source,"scripted",bc={},bcw={},trace=True)
    safe = sorted(set(graph)-states[-1][0])
    return dict(value=value,safe=safe,actions=actions)

def copycuts(graph,source,actions):
    def choose(edges,infected,step):
        action = actions[step-1] if step<=len(actions) else None
        if action in edges and (action[0] in infected)!=(action[1] in infected): return action
        return None
    return controlled(graph,source,choose)

def saveside(graph,source,edge):
    def choose(edges,infected,step):
        if edge in edges and (edge[0] in infected)!=(edge[1] in infected): return edge
        return None
    return controlled(graph,source,choose)

def cycledata(graph,source):
    cycle = nx.cycle_basis(graph)[0]
    edges = {tuple(sorted(e)) for e in zip(cycle,cycle[1:]+cycle[:1])}
    distances = nx.single_source_shortest_path_length(graph,source)
    entry = min(cycle,key=lambda v:distances[v])
    start = cycle.index(entry)
    cycle = cycle[start:]+cycle[:start]
    length = len(cycle)
    arrivals = {v:distances[entry]+min(i,length-i) for i,v in enumerate(cycle)}
    assert all(arrivals[v]==distances[v] for v in cycle)
    collision = [cycle[length//2]] if length%2==0 else [cycle[length//2],cycle[length//2+1]]
    double = [v for v in cycle if sum(distances[u]==distances[v]-1
              for u in graph[v] if u in cycle)==2]
    assert double==(collision if length%2==0 else [])
    forest = graph.copy()
    forest.remove_edges_from(edges)
    roots = {v:next(u for u in component if u in cycle)
             for component in nx.connected_components(forest) for v in component}
    return dict(cycle=cycle,edges=edges,entry=entry,offset=distances[entry],
                arrivals=arrivals,collision=collision,double=double,roots=roots)

def diagnose(graph,source,result):
    value,safe,trace = replay(graph,source,result["actions"])
    safe = set(safe)
    cuts = [step for step in trace if step["action"] is not None]
    ahead = [step for step in cuts if not any(v in step["before"] for v in step["action"])]
    data = cycledata(graph,source)
    findings = dict(source=source,edges=edge_order(graph),safe=sorted(safe),
                    actions=result["actions"],proactive=len(ahead),cycle=data["cycle"],
                    entry=data["entry"],offset=data["offset"],arrivals=data["arrivals"],
                    collision=data["collision"],double=data["double"],
                    cyclecuts=[step["action"] for step in cuts if step["action"] in data["edges"]])
    if len(ahead)!=1: return findings
    step = ahead[0]
    edge = step["action"]
    graphminus = graph.copy()
    graphminus.remove_edge(*edge)
    bridge = not nx.is_connected(graphminus)
    findings.update(edge=edge,round=step["round"],bridge=bridge)
    if not bridge: return findings
    root = data["roots"][edge[0]]
    whole = set(graph)-nx.node_connected_component(graphminus,source)
    assert data["roots"][edge[1]]==root
    current = graph.copy()
    for earlier in trace[:step["round"]]:
        if earlier["action"] is not None: current.remove_edge(*earlier["action"])
    child = next(v for v in edge if v in whole)
    side = set(nx.node_connected_component(current,child))
    final = graph.copy()
    final.remove_edges_from(c["action"] for c in cuts)
    adjacent = set(nx.node_connected_component(final,child))
    remainder = safe-side
    omitted = list(result["actions"])
    omitted[step["round"]-1] = None
    copied = copycuts(graph,source,omitted)
    compressed = copycuts(graph,source,omitted[:step["round"]-1]+omitted[step["round"]:])
    devoted = saveside(graph,source,edge)
    lost = safe-set(copied["safe"])
    outside = set(graph)-safe
    complement = graph.subgraph(outside)
    initial = first_spread(graph,source)
    distances = {v:min(nx.shortest_path_length(complement,u,v) for u in initial) for v in outside}
    arrivals = {v:0 for v in initial}
    for event in trace:
        for v in event["added"]: arrivals[v] = event["round"]
    jobs = []
    for action in cuts:
        e = action["action"]
        if sum(v in safe for v in e)!=1: continue
        parent = next(v for v in e if v in outside)
        protected = next(v for v in e if v in safe)
        component = nx.node_connected_component(final,protected)
        deadline = distances[parent]+1
        jobs.append(dict(edge=e,benefit=len(component),deadline=deadline,
                         release=arrivals[parent]+1,scheduled=action["round"],
                         early=action["round"]<deadline))
    findings.update(root=root,unavoidable=root not in result["possible"],
                    collisionroot=root in data["collision"],incident=root in edge,
                    whole=sorted(whole),side=sorted(side),adjacent=sorted(adjacent),
                    remainder=sorted(remainder),lost=sorted(lost),copied=copied,
                    compressed=compressed,devoted=devoted,
                    same=whole==side==adjacent==lost,
                    lossintree=all(data["roots"][v]==root for v in lost),
                    preserves=remainder<=set(copied["safe"]),
                    preservescompressed=remainder<=set(compressed["safe"]),
                    savesside=side<=set(devoted["safe"]),jobs=jobs,
                    rootarrival=arrivals[root],
                    arrivalmatch=arrivals[root]==max(0,data["arrivals"][root]-1),
                    allboundary=all(sum(v in safe for v in action["action"])==1 for action in cuts),
                    allbridges=all(e["action"] not in data["edges"] for e in cuts))
    return findings

def assignments(graph):
    nodes = sorted(graph)
    yield "ones",dict.fromkeys(nodes,1)
    yield "ascending",{v:1+i%5 for i,v in enumerate(nodes)}
    yield "descending",{v:5-i%5 for i,v in enumerate(nodes)}
    cycle = set(nx.cycle_basis(graph)[0])
    yield "cycleheavy",{v:5 if v in cycle else 1 for v in nodes}
    yield "treeheavy",{v:1 if v in cycle else 5 for v in nodes}
    for seed in (17,29,43):
        rng = Random(20261009+100*len(graph)+seed)
        yield f"random{seed}",{v:rng.randint(1,5) for v in nodes}

def terminalsets(graph,source,threshold=False):
    initial = (first_spread(graph,source),edge_order(graph))
    pending = [initial]
    seen = {initial}
    terminal = set()
    while pending:
        infected,edges = pending.pop()
        after = infected|{v for u,v in edges if u in infected}|{u for u,v in edges if v in infected}
        if after==infected:
            terminal.add(frozenset(set(graph)-infected))
            continue
        actions = [None]+[e for e in edges if ((e[0] in infected)!=(e[1] in infected)
                   if threshold else not all(v in infected for v in e))]
        for action in actions:
            remaining = tuple(e for e in edges if e!=action)
            after = infected|{v for u,v in remaining if u in infected}|{u for u,v in remaining if v in infected}
            state = after,remaining
            if state not in seen:
                seen.add(state)
                pending.append(state)
    return terminal

def regression():
    atlas = [g for g in nx.graph_atlas_g() if 3<=len(g)<=5 and
             g.number_of_edges()==len(g) and nx.is_connected(g)]
    for graph in atlas:
        for threshold in (False,True):
            terminals = {s:terminalsets(graph,s,threshold) for s in graph}
            for label,weights in assignments(graph):
                solve = weighted(graph,weights,threshold)
                for source in graph:
                    result = solve(source)
                    best = max(sum(weights[v] for v in safe) for safe in terminals[source])
                    optimal = [safe for safe in terminals[source] if sum(weights[v] for v in safe)==best]
                    assert result["value"]==best
                    assert set(result["possible"])==set().union(*optimal)
                    assert set(result["necessary"])==set(optimal[0]).intersection(*optimal)
                    value,safe,trace = replay(graph,source,result["actions"],threshold)
                    assert sum(weights[v] for v in safe)==best
    graph = nx.graph_atlas_g()[96]
    result = weighted(graph,dict.fromkeys(graph,1))(1)
    findings = diagnose(graph,1,result)
    assert findings["same"] and findings["preserves"] and findings["savesside"]
    assert findings["unavoidable"] and findings["collisionroot"]
    assert not findings["preservescompressed"]
    assert all(job["deadline"]==job["release"]==2 for job in findings["jobs"])
    graph = nx.Graph([(0,1),(1,2),(2,3),(3,4),(2,4)])
    assert cycledata(graph,0)["offset"]==2
    assert copycuts(graph,0,[(1,2)])["safe"]==[2,3,4]

def search(output="data/unicyclelemmas.json"):
    start = perf_counter()
    with open("data/cyclesearch.csv",newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["cohort"]=="exhaustive" and
                r["rank"]=="1" and int(r["n"])<=12]
    assert len(rows)==89965 and len({r["label"] for r in rows})==7872
    counts = Counter()
    examples = {}
    details = []
    graphs = {}
    checked = {}
    for row in rows:
        label = row["label"]
        source = int(row["source"])
        graph = graphs.setdefault(label,nx.from_graph6_bytes(row["graph6"].encode()))
        cycledata(graph,source)
        counts["arrivalchecks"] += 1
        if not int(row["proactive"]): continue
        if label not in checked: checked[label] = weighted(graph,dict.fromkeys(graph,1))
        result = checked[label](source)
        original,actions = oracle(graph)(source)
        assert result["value"]==int(row["opt"])==original[0]
        assert result["proactive"]==int(row["proactive"])==original[1]
        assert result["actions"]==actions
        finding = diagnose(graph,source,result)
        finding.update(label=label,graph6=row["graph6"],n=len(graph),
                       opt=result["value"],thr=int(row["thr"]),possible=result["possible"])
        details.append(finding)
        counts["proactivecases"] += 1
        counts[f"cycle{len(finding['cycle'])}:offset{finding['offset']}"] += 1
        predicates = {"one":finding["proactive"]<=1,"bridge":finding.get("bridge",False),
                      "attached":finding.get("bridge",False) and "root" in finding,
                      "unavoidable":finding.get("unavoidable",False),
                      "collision":finding.get("collisionroot",False),
                      "incident":finding.get("incident",False),"definitions":finding.get("same",False),
                      "copy":finding.get("preserves",False),"tree":finding.get("lossintree",False),
                      "save":finding.get("savesside",False),"compressed":finding.get("preservescompressed",False),
                      "allboundary":finding.get("allboundary",False),"allbridges":finding.get("allbridges",False),
                      "arrivalmatch":finding.get("arrivalmatch",False),
                      "release":all(j["release"]==j["deadline"] for j in finding.get("jobs",[])),
                      "partitionvalue":int(row["thr"])>=max(len(finding.get("side",[])),
                                                            len(finding.get("remainder",[])))}
        for name,success in predicates.items():
            counts[name+"pass" if success else name+"fail"] += 1
            if not success: examples.setdefault(name,finding)
        counts["cyclecutcases"] += bool(finding["cyclecuts"])
        counts["earlyjobs"] += sum(job["early"] for job in finding.get("jobs",[]))
        multiplicities = Counter(job["deadline"] for job in finding.get("jobs",[]))
        counts["onedoubleddeadline"] += sorted(v for v in multiplicities.values() if v>1)==[2]
        counts["benefitspass"] += sum(job["benefit"] for job in finding.get("jobs",[]))==result["value"]
    print("UNWEIGHTED",json.dumps(dict(counts),sort_keys=True),flush=True)
    for name,finding in examples.items(): print("COUNTEREXAMPLE",name,json.dumps(finding),flush=True)
    stress = []
    minima = Fraction(1)
    weightedfailures = []
    structures = Counter()
    selected = {r["label"]:r for r in rows if int(r["n"])<=8}
    for label,row in selected.items():
        graph = graphs[label]
        for assignment,weights in assignments(graph):
            unrestricted = weighted(graph,weights)
            threshold = weighted(graph,weights,True)
            for source in sorted(graph):
                result = unrestricted(source)
                restricted = threshold(source)
                value,thvalue = result["value"],restricted["value"]
                for solution,only in ((result,False),(restricted,True)):
                    saved,safe,trace = replay(graph,source,solution["actions"],only)
                    assert sum(weights[v] for v in safe)==solution["value"]
                item = dict(label=label,graph6=row["graph6"],source=source,assignment=assignment,
                            weights=[weights[v] for v in sorted(graph)],opt=value,thr=thvalue,
                            proactive=result["proactive"])
                if value: minima = min(minima,Fraction(thvalue,value))
                if 2*thvalue<value:
                    item.update(actions=result["actions"],thactions=restricted["actions"])
                    weightedfailures.append(item)
                if result["proactive"]:
                    finding = diagnose(graph,source,result)
                    item["structure"] = finding
                    if finding.get("same"):
                        finding["sideweight"] = sum(weights[v] for v in finding["side"])
                        finding["remainderweight"] = sum(weights[v] for v in finding["remainder"])
                        assert finding["sideweight"]+finding["remainderweight"]==value
                        assert sum(weights[v] for v in finding["copied"]["safe"])==finding["remainderweight"]
                        assert sum(weights[v] for v in finding["devoted"]["safe"])>=finding["sideweight"]
                    for name,key in (("one","proactive"),("bridge","bridge"),("unavoidable","unavoidable"),
                                     ("collision","collisionroot"),("incident","incident"),
                                     ("copy","preserves"),("save","savesside")):
                        success = finding[key]<=1 if name=="one" else finding.get(key,False)
                        structures[name+"pass" if success else name+"fail"] += 1
                stress.append(item)
    weightedsummary = dict(graphs=len(selected),assignments=8,cases=len(stress),
                           positive=sum(r["opt"]>0 for r in stress),
                           failures=sum(r["thr"]<r["opt"] for r in stress),
                           belowhalf=len(weightedfailures),minimum=str(minima),
                           proactive=max(r["proactive"] for r in stress),structure=dict(structures))
    print("WEIGHTED",json.dumps(weightedsummary,sort_keys=True),flush=True)
    for item in weightedfailures[:20]: print("WEIGHTED COUNTEREXAMPLE",json.dumps(item),flush=True)
    report = dict(unweighted=dict(counts),details=details,examples=examples,
                  weightedsummary=weightedsummary,weighted=stress,seconds=perf_counter()-start)
    with open(output,"w") as stream: json.dump(report,stream,separators=(",",":"))
    print("OUTPUT",output,"BYTES",Path(output).stat().st_size,flush=True)
    print(f"TOTAL SECONDS {perf_counter()-start:.6f}",flush=True)

if __name__=="__main__":
    regression()
    search()
