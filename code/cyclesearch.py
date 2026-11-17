# Exact sparse-cycle comparisons and family checks
import argparse
import records as savedreports
import csv
import json
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import product
from pathlib import Path
from random import Random
from time import perf_counter
import networkx as nx
from safezones import centralities,edge_order,exact,first_spread,simulate
from thresholdsearch import cactus,cactusgraphs,extend,replay,unique

def oracle(graph,threshold=False):
    nodes = sorted(graph)
    bits = {v:1<<i for i,v in enumerate(nodes)}
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
            if reached & (u|v) and infected & (u|v)!=(u|v):
                active |= bit
            remaining ^= bit
        return active
    @lru_cache(None)
    def best(infected,remaining):
        if advance(infected,remaining)==infected:
            return len(nodes)-infected.bit_count(),0,0
        candidates = [0]
        pending = remaining
        while pending:
            bit = pending & -pending
            u,v = ends[bit.bit_length()-1]
            if not threshold or bool(infected & u)!=bool(infected & v):
                candidates.append(bit)
            pending ^= bit
        result = (-1,0,0)
        for action in candidates:
            following = remaining ^ action
            after = advance(infected,following)
            value,proactive,cuts = best(after,trim(after,following))
            if action:
                u,v = ends[action.bit_length()-1]
                proactive += not bool(infected & (u|v))
                cuts += 1
            candidate = value,proactive,cuts
            if (-value,proactive,cuts)<(-result[0],result[1],result[2]):
                result = candidate
                choice[infected,remaining] = action
        return result
    def solve(source):
        infected = sum(bits[v] for v in first_spread(graph,source))
        remaining = trim(infected,full)
        result = best(infected,remaining)
        actions = []
        while advance(infected,remaining)!=infected:
            action = choice[infected,remaining]
            actions.append(edges[action.bit_length()-1] if action else None)
            remaining ^= action
            infected = advance(infected,remaining)
            remaining = trim(infected,remaining)
        return result,actions
    solve.cache = best.cache_info
    return solve

def attach(graph,parent,tree,root=0):
    labels = {v:len(graph)+i for i,v in enumerate(sorted(tree))}
    graph.add_nodes_from(labels.values())
    graph.add_edges_from((labels[u],labels[v]) for u,v in tree.edges)
    graph.add_edge(parent,labels[root])
    return labels

def branches(left,right,leftroot=0,rightroot=0):
    graph = nx.cycle_graph(4)
    a = attach(graph,2,left,leftroot)
    b = attach(graph,2,right,rightroot)
    return graph,a[leftroot],b[rightroot]

def paths(a,b):
    return branches(nx.path_graph(a),nx.path_graph(b))

def zero(m):
    graph = nx.Graph([(0,1),(0,2),(1,3),(1,4),(2,3),(2,4),(3,5),(4,5)])
    graph.add_edges_from((5,v) for v in range(6,m+5))
    return graph

def rankexample():
    return nx.Graph([(0,1),(0,3),(0,4),(0,5),(0,6),(0,7),(1,2),(2,3),(2,4),(2,8)])

def cactusexample():
    return nx.Graph([(0,1),(0,3),(0,4),(0,5),(0,6),(0,7),(1,2),(2,3),
                     (4,5),(4,8),(5,9),(6,7),(6,10)])

def unicyclicgraphs(limit):
    groups = {3:[nx.cycle_graph(3)]}
    for n in range(4,limit+1):
        def candidates():
            yield nx.cycle_graph(n)
            for graph in groups[n-1]:
                for v in graph:
                    larger = graph.copy()
                    larger.add_edge(v,n-1)
                    yield larger
        groups[n] = unique(candidates())
    return groups

def ranktwo(graphs):
    def candidates():
        for graph in graphs:
            for u,v in nx.non_edges(graph):
                larger = graph.copy()
                larger.add_edge(u,v)
                yield larger
    return unique(candidates())

def regression():
    for graph in nx.graph_atlas_g():
        if not 3<=len(graph)<=6 or not nx.is_connected(graph): continue
        for threshold in (False,True):
            solve = oracle(graph,threshold)
            for source in graph:
                result,actions = solve(source)
                assert result[0]==exact(graph,source,threshold_only=threshold)
                assert replay(graph,source,actions,threshold)[0]==result[0]
    for a,b in product(range(1,5),repeat=2):
        graph,left,right = paths(a,b)
        assert oracle(graph)(0)[0][0]==a+b
        assert oracle(graph,True)(0)[0][0]==a+b-1
    for m in range(1,7):
        graph = zero(m)
        assert oracle(graph)(0)[0][0]==m
        assert oracle(graph,True)(0)[0][0]==min(1,m-1)
        graph,a,b = branches(nx.star_graph(m-1),nx.star_graph(m-1))
        assert oracle(graph)(0)[0][0]==2*m
        assert oracle(graph,True)(0)[0][0]==(1 if m==1 else m+1)
    for graph,source,u,t,proactive in ((rankexample(),8,3,1,2),(cactusexample(),2,5,2,1)):
        original = graph.copy()
        result,actions = oracle(graph)(source)
        thresult,thactions = oracle(graph,True)(source)
        assert result[:2]==(u,proactive) and thresult[0]==t
        assert exact(graph,source)==u and exact(graph,source,threshold_only=True)==t
        assert replay(graph,source,actions)[0]==u
        assert replay(graph,source,thactions,True)[0]==t
        assert nx.utils.graphs_equal(graph,original)
    graph = nx.graph_atlas_g()[96]
    assert oracle(graph)(2)[0][0]==oracle(graph,True)(2)[0][0]==1
    assert simulate(graph,2,"threshold")==0

def cycleblocks(count,length,shape,gap=0):
    graph = nx.cycle_graph(length)
    tip = length//2
    for i in range(1,count):
        anchor = 0 if shape=="star" else tip
        for j in range(gap):
            following = len(graph)
            graph.add_edge(anchor,following)
            anchor = following
        vertices = [anchor]+list(range(len(graph),len(graph)+length-1))
        nx.add_cycle(graph,vertices)
        tip = vertices[length//2]
    return graph

def targeted():
    for m in range(1,13):
        graph,a,b = paths(m,m)
        yield f"paths({m},{m})",graph,[0],(2*m,2*m-1)
        graph,a,b = branches(nx.star_graph(m-1),nx.star_graph(m-1))
        yield f"stars({m},{m})",graph,[0],(2*m,1 if m==1 else m+1)
        yield f"zero({m})",zero(m),[0],(m,min(1,m-1))
        graph = nx.cycle_graph(4)
        graph.add_edges_from((2,v) for v in range(4,m+5))
        yield f"tight({m+1})",graph,[0],(2,1)
    for a,b in product(range(1,7),repeat=2):
        if a==b: continue
        graph,left,right = paths(a,b)
        yield f"paths({a},{b})",graph,[0],(a+b,a+b-1)
    rooted = [(nx.empty_graph(1),0)]
    for n in range(2,5):
        for tree in nx.nonisomorphic_trees(n):
            rooted.extend((tree,v) for v in tree)
    for i,(left,a) in enumerate(rooted):
        for j,(right,b) in enumerate(rooted):
            graph,u,v = branches(left,right,a,b)
            yield f"rooted({i},{j})",graph,[0],None
    for count,length,shape,gap in product((2,3,4),(3,4,5),("chain","star"),(0,2)):
        graph = cycleblocks(count,length,shape,gap)
        yield f"blocks({count},{length},{shape},{gap})",graph,None,None
    rng = Random(20261008)
    for i in range(48):
        n = 13+i%8
        length = 3+i%7
        graph = nx.cycle_graph(length)
        for v in range(length,n):
            graph.add_edge(v,rng.randrange(v))
        yield f"randomuni({i})",graph,None,None
    for i in range(64):
        n = 11+i%8
        graph = nx.from_prufer_sequence([rng.randrange(n) for j in range(n-2)])
        for j in range(2):
            graph.add_edge(*rng.choice(sorted(nx.non_edges(graph))))
        yield f"randomranktwo({i})",graph,None,None
    for lengths in ((1,2,4),(2,2,2),(2,3,4),(3,3,3),(1,3,6)):
        base = nx.empty_graph(2)
        for length in lengths:
            nx.add_path(base,[0]+list(range(len(base),len(base)+length-1))+[1])
        for shape in ("path","star"):
            graph = base.copy()
            tree = nx.path_graph(5) if shape=="path" else nx.star_graph(4)
            attach(graph,0,tree)
            attach(graph,1,nx.path_graph(3))
            yield f"theta({lengths},{shape})",graph,None,None

def statistics(rows):
    positive = [r for r in rows if r["opt"]]
    return dict(graphs=len({r["label"] for r in rows}),cases=len(rows),
                nontrivial=len(positive),failures=sum(r["thr"]<r["opt"] for r in rows),
                tbcfailures=sum(r["tbc"]<r["thr"] for r in rows),
                belowhalf=sum(2*r["thr"]<r["opt"] for r in positive),
                tbcbelowhalf=sum(2*r["tbc"]<r["opt"] for r in positive),
                minimumthr=str(min(Fraction(r["thr"],r["opt"]) for r in positive)) if positive else None,
                minimumtbc=str(min(Fraction(r["tbc"],r["opt"]) for r in positive)) if positive else None,
                maxproactive=max((r["proactive"] for r in rows),default=0))

def witness(row):
    graph = nx.from_graph6_bytes(row["graph6"].encode())
    source = row["source"]
    result,actions = oracle(graph)(source)
    thresult,thactions = oracle(graph,True)(source)
    saved,safe,trace = replay(graph,source,actions)
    thsaved,thsafe,thtrace = replay(graph,source,thactions,True)
    assert saved==row["opt"] and thsaved==row["thr"]
    bc,bcw = centralities(graph,dict.fromkeys(edge_order(graph),1))
    tbc,cuts,states = simulate(graph,source,"threshold",bc=bc,bcw=bcw,trace=True)
    assert tbc==row["tbc"]
    return dict(row,edges=edge_order(graph),actions=actions,thactions=thactions,
                safe=safe,thsafe=thsafe,trace=trace,thtrace=thtrace,
                tbccuts=cuts,tbcsafe=sorted(set(graph)-states[-1][0]))

def proactivecuts(rows):
    counts = defaultdict(int)
    cases = 0
    for row in rows:
        if row["rank"]!=1 or not row["proactive"]: continue
        graph = nx.from_graph6_bytes(row["graph6"].encode())
        result,actions = oracle(graph)(row["source"])
        value,safe,trace = replay(graph,row["source"],actions)
        bridges = {tuple(sorted(e)) for e in nx.bridges(graph)}
        found = 0
        for step in trace:
            action = step["action"]
            if action is None or any(v in step["before"] for v in action): continue
            found += 1
            kind = "bridge" if action in bridges else "cycle"
            kind += " boundary" if sum(v in safe for v in action)==1 else " internal"
            counts[kind] += 1
        assert value==row["opt"] and found==row["proactive"]
        cases += 1
    return dict(cases=cases,cuts=dict(counts))

def search(unilimit=12,cactuslimit=11,ranklimit=10,output="data/cyclesearch.csv"):
    savedreports.guard(output)
    start = perf_counter()
    atlas = nx.graph_atlas_g()
    groups = {n:[g for g in atlas if len(g)==n and nx.is_connected(g)] for n in range(3,8)}
    groups[8] = unique(extend(groups[7]))
    assert len(groups[8])==11117
    cacti = cactusgraphs(cactuslimit)
    uni = unicyclicgraphs(unilimit)
    for n in range(3,min(cactuslimit,unilimit)+1):
        expected = [g for g in cacti[n] if g.number_of_edges()==n]
        assert len(uni[n])==len(expected)==len(unique(uni[n]+expected))
    for n in range(3,9):
        expected = [g for g in groups[n] if cactus(g)]
        assert len(expected)==len(cacti[n])==len(unique(expected+cacti[n]))
    print(f"ENUMERATION baseline8={len(groups[8])} seconds={perf_counter()-start:.6f}",flush=True)
    rows = []
    maximum = 0
    slowest = (0,None,None)
    checks = 0
    replays = 0
    columns = ["label","cohort","exhaustive","coverage","graph6","source","n","m","rank","cactus",
               "density","bridges","articulations","girth","degree","eccentricity",
               "opt","thr","tbc","rhothr","rhotbc","proactive","seconds","optstates","thrstates"]
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with open(output,"x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        def scan(graph,label,cohort,sources=None,expected=None):
            nonlocal maximum,slowest,checks,replays
            original = edge_order(graph)
            n,m = len(graph),graph.number_of_edges()
            iscactus = cactus(graph)
            girth = nx.girth(graph) if m>=n else ""
            distances = nx.eccentricity(graph)
            bc,bcw = centralities(graph,dict.fromkeys(original,1))
            common = dict(label=label,cohort=cohort,exhaustive=cohort=="exhaustive",
                          coverage=";".join(name for name,limit in (("unicyclic",unilimit),
                              ("cactus",cactuslimit),("ranktwo",ranklimit)) if n<=limit)
                              if cohort=="exhaustive" else "targeted",
                          graph6=nx.to_graph6_bytes(graph,header=False).decode().strip(),
                          n=n,m=m,rank=m-n+1,cactus=iscactus,density=nx.density(graph),
                          bridges=len(list(nx.bridges(graph))),
                          articulations=len(list(nx.articulation_points(graph))),girth=girth)
            assert edge_order(nx.from_graph6_bytes(common["graph6"].encode()))==original
            unrestricted = oracle(graph)
            threshold = oracle(graph,True)
            for source in sorted(graph) if sources is None else sources:
                tick = perf_counter()
                result,actions = unrestricted(source)
                thresult,thactions = threshold(source)
                u,t = result[0],thresult[0]
                h = simulate(graph,source,"threshold",bc=bc,bcw=bcw)
                assert 0<=h<=t<=u<=n-len(first_spread(graph,source))
                assert m!=n-1 or t==u
                if expected is not None: assert (u,t)==expected,(label,source,u,t,expected)
                if n<=7 or cohort=="targeted" and (n<=16 or
                    label.startswith(("paths(","stars(","zero(","tight("))):
                    assert u==exact(graph,source)
                    assert t==exact(graph,source,threshold_only=True)
                    checks += 1
                assert replay(graph,source,actions)[0]==u
                assert replay(graph,source,thactions,True)[0]==t
                replays += 2
                elapsed = perf_counter()-tick
                if elapsed>slowest[0]: slowest = elapsed,label,source
                counts = unrestricted.cache().currsize,threshold.cache().currsize
                maximum = max(maximum,*counts)
                row = dict(common,source=source,degree=graph.degree(source),eccentricity=distances[source],
                           opt=u,thr=t,tbc=h,rhothr=t/u if u else "",rhotbc=h/u if u else "",
                           proactive=result[1],seconds=elapsed,optstates=counts[0],thrstates=counts[1])
                writer.writerow(row)
                rows.append(row)
            assert edge_order(graph)==original
            stream.flush()
        for n in range(3,max(unilimit,cactuslimit,ranklimit)+1):
            tick = perf_counter()
            if n<=8:
                selected = [(g.name if n<=7 else f"n8:{i}",g) for i,g in enumerate(groups[n])
                            if cactus(g) or g.number_of_edges()-n+1==2]
            else:
                selected = [(f"cactus{n}:{i}",g) for i,g in enumerate(cacti[n])] if n<=cactuslimit else []
                if n>cactuslimit and n<=unilimit:
                    selected.extend((f"uni{n}:{i}",g) for i,g in enumerate(uni[n]))
                if n<=ranklimit:
                    bicyclic = ranktwo(uni[n])
                    selected.extend((f"ranktwo{n}:{i}",g) for i,g in enumerate(bicyclic) if not cactus(g))
                    print("RANKTWO ENUMERATED",n,len(bicyclic),flush=True)
            for label,graph in selected:
                scan(graph,label,"exhaustive")
            for name,condition in (("UNICYCLIC",lambda r:r["rank"]==1),
                                   ("CACTUS",lambda r:r["cactus"]),
                                   ("RANKTWO",lambda r:r["rank"]==2)):
                subset = [r for r in rows if r["n"]==n and condition(r)]
                limit = {"UNICYCLIC":unilimit,"CACTUS":cactuslimit,"RANKTWO":ranklimit}[name]
                if subset and n<=limit: print(name,n,json.dumps(statistics(subset)),flush=True)
            print("ORDER SECONDS",n,f"{perf_counter()-tick:.6f}",flush=True)
        for i,(label,graph,sources,expected) in enumerate(targeted()):
            scan(graph,label,"targeted",sources,expected)
            if i<48:
                row = rows[-1]
                print("FAMILY",label,"OPT",row["opt"],"THR",row["thr"],"TBC",row["tbc"],
                      "rho",str(Fraction(row["thr"],row["opt"])),flush=True)
            if i%50==0: print("TARGETED PROGRESS",i,flush=True)
    report(rows,unilimit,cactuslimit,ranklimit)
    print("DATASET",output,"rows",len(rows),"graphs",len({r["label"] for r in rows}),flush=True)
    print("PRODUCTION CROSSCHECKS",checks,"SCHEDULE REPLAYS",replays,flush=True)
    print("MAXIMUM CACHED STATES PER GRAPH",maximum,"SLOWEST",slowest,flush=True)
    print(f"TOTAL SECONDS {perf_counter()-start:.6f}",flush=True)

def report(rows,unilimit=12,cactuslimit=11,ranklimit=10):
    for cohort in ("previous","exhaustive","targeted"):
        base = [r for r in rows if r["cohort"]==("exhaustive" if cohort=="previous" else cohort)
                and (cohort!="previous" or r["n"]<=9 and (r["cactus"] or r["n"]<=8))]
        for name,condition in (("UNICYCLIC",lambda r:r["rank"]==1),
                               ("CACTUS",lambda r:r["cactus"]),
                               ("RANKTWO",lambda r:r["rank"]==2)):
            limit = {"UNICYCLIC":unilimit,"CACTUS":cactuslimit,"RANKTWO":ranklimit}[name]
            subset = [r for r in base if condition(r) and (cohort!="exhaustive" or r["n"]<=limit)]
            print("SUMMARY",cohort,name,json.dumps(statistics(subset)),flush=True)
            for description,predicate in (("TBC BELOW THR",lambda r:r["tbc"]<r["thr"]),
                                           ("TBC BELOW HALF",lambda r:2*r["tbc"]<r["opt"]),
                                           ("THR BELOW HALF",lambda r:2*r["thr"]<r["opt"]),
                                           ("MULTIPLE PROACTIVE",lambda r:r["proactive"]>1)):
                cases = [r for r in subset if predicate(r)]
                if cases:
                    row = min(cases,key=lambda r:(r["n"],r["label"],r["source"]))
                    print(description,cohort,name,json.dumps(witness(row),sort_keys=True),flush=True)
    print("UNICYCLIC PROACTIVE CLASSIFICATION",json.dumps(proactivecuts(rows)),flush=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unicyclic",type=int,default=12)
    parser.add_argument("--cactus",type=int,default=11)
    parser.add_argument("--ranktwo",type=int,default=10)
    parser.add_argument("--output",default="data/cyclesearch.csv")
    parser.add_argument("--report",action="store_true")
    args = parser.parse_args()
    if args.report:
        with open(args.output,newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ("n","m","rank","source","opt","thr","tbc","proactive"):
                row[key] = int(row[key])
            row["cactus"] = row["cactus"]=="True"
        report(rows,args.unicyclic,args.cactus,args.ranktwo)
    else:
        search(args.unicyclic,args.cactus,args.ranktwo,args.output)
