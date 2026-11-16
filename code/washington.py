# Washington topology counterfactual pilot
import records as savedreports
import csv
import hashlib
import json
import platform
import random
import signal
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from time import perf_counter
import networkx as nx

root = Path(__file__).resolve().parent.parent
folder = root/'data/results'
import air
from hospital import components,score

R = frozenset(air.washington)
A = frozenset(('BFI','ESD','FRD','RNT'))
U = R|A
levels = {'internal':(9,15,21,27,33,36),'boundary':(1,2,4,5,6),'competing':(3,4,5,7,8,9)}
seconds = 20
limit = 200000

def boundary(graph,vertices):
    return tuple(e for e in air.edges(graph) if (e[0] in vertices)!=(e[1] in vertices))

def signature(graph):
    return hashlib.sha256(json.dumps([sorted(graph),air.edges(graph)],separators=(',',':')).encode()).hexdigest()

def original():
    path = root/'data/raw/t100_domestic_segment_2025.zip'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==air.sourcehash
    graph,audit = air.readgraph(path)
    assert (len(graph),graph.number_of_edges())==(443,7160)
    assert (len(R),len(boundary(graph,R)),len(U),len(boundary(graph,U)))==(9,3,13,6)
    assert graph.subgraph(R).number_of_edges()==30 and graph.subgraph(U).number_of_edges()==37
    return graph

def perturb(base,feature,level,seed):
    graph = base.copy()
    rng = random.Random(seed)
    if feature=='original': return graph
    if feature=='internal':
        present = set(air.edges(base.subgraph(R)))
        pool = sorted(present if level<30 else set(combinations(sorted(R),2))-present)
        rng.shuffle(pool)
        for edge in pool:
            if graph.subgraph(R).number_of_edges()==level: break
            if level<30:
                data = dict(graph.edges[edge])
                graph.remove_edge(*edge)
                if not nx.is_connected(graph.subgraph(R)): graph.add_edge(*edge,**data)
            else: graph.add_edge(*edge,synthetic=True)
    else:
        inside,outside,current = (R,A,3) if feature=='boundary' else (A,set(base)-U,6)
        present = boundary(base,R if feature=='boundary' else U)
        if level<current:
            pool = list(present)
            rng.shuffle(pool)
            graph.remove_edges_from(pool[:current-level])
        else:
            pool = sorted(tuple(sorted((u,v))) for u in inside for v in outside if not base.has_edge(u,v))
            rng.shuffle(pool)
            graph.add_edges_from((u,v,dict(synthetic=True)) for u,v in pool[:level-current])
    result = nx.Graph()
    result.add_nodes_from((v,dict(graph.nodes[v])) for v in sorted(graph))
    result.add_edges_from((u,v,dict(graph.edges[u,v])) for u,v in air.edges(graph))
    return result

def validate(base,graph,feature,level):
    assert set(graph)==set(base) and nx.is_connected(graph)
    for vertices in (R,U,set(graph)-R,set(graph)-U): assert nx.is_connected(graph.subgraph(vertices))
    changed = set(air.edges(base))^set(air.edges(graph))
    if feature=='internal':
        assert all(set(e)<=R for e in changed)
        assert graph.subgraph(R).number_of_edges()==level
        assert boundary(graph,R)==boundary(base,R) and boundary(graph,U)==boundary(base,U)
    elif feature=='boundary':
        assert all((u in R and v in A) or (v in R and u in A) for u,v in changed)
        assert len(boundary(graph,R))==level and boundary(graph,U)==boundary(base,U)
        assert air.edges(graph.subgraph(R))==air.edges(base.subgraph(R))
    elif feature=='competing':
        assert all((u in A and v not in U) or (v in A and u not in U) for u,v in changed)
        assert len(boundary(graph,U))==level and boundary(graph,R)==boundary(base,R)
        assert air.edges(graph.subgraph(U))==air.edges(base.subgraph(U))
    else: assert not changed

def variants(base):
    cases = [('original',0,0)]+[(feature,level,seed) for feature,values in levels.items() for level in values for seed in (0,1)]
    found = {}
    for feature,level,seed in cases:
        graph = perturb(base,feature,level,seed)
        validate(base,graph,feature,level)
        key = signature(graph)
        if key in found:
            found[key]['seeds'].append(seed)
            continue
        found[key] = dict(variant='original' if feature=='original' else f'{feature}{level}s{seed}',feature=feature,level=level,
                          seeds=[seed],graph=graph,graphhash=key,
                          added=sorted(set(air.edges(graph))-set(air.edges(base))),removed=sorted(set(air.edges(base))-set(air.edges(graph))))
    return list(found.values())

def adapted(graph):
    result = graph.copy()
    nx.set_node_attributes(result,1,'inbound')
    nx.set_edge_attributes(result,1,'passengers')
    return result

def expired(signum,frame):
    raise TimeoutError('Exact search exceeded its per-graph time cap')

def solve(graph,b,cap=seconds,work=limit):
    start = perf_counter()
    previous = signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,cap)
    try:
        catalog = air.partitions(adapted(graph),b,limit=work)
        values,stats = air.optimize(catalog,'uu',limit=work)
        return dict(status='exact',values={j:(Fraction(air.numerator(catalog,'uu',state),len(graph)),state[4]) for j,state in values.items()},
                    subsets=catalog['stats']['localsubsets'],states=stats['maximumstates'],error='',seconds=perf_counter()-start)
    except (RuntimeError,TimeoutError) as error:
        return dict(status='unverified',values={},subsets='',states='',error=str(error),seconds=perf_counter()-start)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)

def evaluate(case):
    graph = case['graph']
    start = perf_counter()
    cuts = boundary(graph,R)
    budgets = sorted({3,len(cuts)}) if case['feature']=='boundary' else [3]
    order = air.rankings(adapted(graph))['bc']
    ranks = {edge:i+1 for i,edge in enumerate(order)}
    p,a = air.objectives(graph,'uniform','airports')
    candidate = score(graph,cuts,p,a)
    assert candidate==Fraction(7812,443)
    exact = solve(graph,max(budgets))
    rows = []
    for b in budgets:
        chosen = order[:b]
        value = score(graph,chosen,p,a)
        feasible = len(cuts)<=b
        row = {key:val for key,val in case.items() if key!='graph'}
        row.update(synthetic=case['feature']!='original',n=len(graph),m=graph.number_of_edges(),budget=b,
                   rvertices=sorted(R),uvertices=sorted(U),rinternaledges=graph.subgraph(R).number_of_edges(),
                   uinternaledges=graph.subgraph(U).number_of_edges(),rdensity=str(Fraction(graph.subgraph(R).number_of_edges(),36)),
                   udensity=str(Fraction(graph.subgraph(U).number_of_edges(),78)),rboundary=cuts,rboundarysize=len(cuts),
                   uboundary=boundary(graph,U),uboundarysize=len(boundary(graph,U)),rboundaryranks=[ranks[e] for e in cuts],
                   graphconnected=nx.is_connected(graph),rconnected=nx.is_connected(graph.subgraph(R)),
                   rcomplementconnected=nx.is_connected(graph.subgraph(set(graph)-R)),uconnected=nx.is_connected(graph.subgraph(U)),
                   ucomplementconnected=nx.is_connected(graph.subgraph(set(graph)-U)),
                   bccuts=chosen,bcscore=str(value),bcdecimal=float(value),bccomponents=sorted(map(len,components(graph,chosen)),reverse=True),
                   candidatescore=str(candidate),candidatedecimal=float(candidate),candidatefeasible=feasible,
                   exactstatus=exact['status'],exact='',exactdecimal='',optimalcut=[],optimalpartition=[],optimalcomponents=[],
                   bcratio='',bcratiodecimal='',candidateoptimal=False if not feasible else '',rinoptimal='',
                   exactsubsets=exact['subsets'],exactstates=exact['states'],exactseconds=exact['seconds'],exacterror=exact['error'])
        if b in exact['values']:
            optimum,optimalcut = exact['values'][b]
            groups = sorted(components(graph,optimalcut),key=lambda c:(-len(c),c))
            assert score(graph,optimalcut,p,a)==optimum>=value
            assert not feasible or candidate<=optimum
            row.update(exact=str(optimum),exactdecimal=float(optimum),optimalcut=optimalcut,optimalpartition=groups,
                       optimalcomponents=list(map(len,groups)),bcratio=str(value/optimum),bcratiodecimal=float(value/optimum),
                       candidateoptimal=feasible and candidate==optimum,rinoptimal=any(set(c)==R for c in groups))
        row['seconds'] = perf_counter()-start
        rows.append(row)
    return rows

def report(rows,base,elapsed):
    originalrow = next(row for row in rows if row['feature']=='original')
    paths = [root/'data/raw/t100_domestic_segment_2025.zip',root/'code/air.py',root/'code/hospital.py',Path(__file__)]
    lines = ['WASHINGTON AIR-NETWORK GENERALITY PILOT','',
             'Scope: synthetic topology counterfactuals of the original tau=1 graph, not measured alternative routes.',
             'Original graph: 443 airports, 7160 routes; R='+','.join(sorted(R))+'; U=R plus BFI, ESD, FRD, RNT.',
             'Original boundaries: |delta(R)|=3, |delta(U)|=6. No vertices are dropped and no giant is reselected.',
             'Uniform source over all 443 airports; unit target weights; static fixed cuts; score=N-sum(component size squared)/N.',
             'Ordinary BC is recomputed once for each counterfactual original graph, then held fixed for its cuts. air.rankings uses 2*NetworkX normalized=False and canonical lexicographic ties.',
             'All graphs retain connected R, U, V-R and V-U. The graph is connected before intervention.',
             'Seeds 0 and 1 shuffle canonically sorted eligible edges. Duplicate resulting graphs are evaluated once; seeds lists retain aliases.',
             'Internal levels: 9,15,21,27,33,36 edges. Remove shuffled internal edges only if R stays connected, or add shuffled missing internal edges.',
             'R-boundary levels: 1,2,4,5,6. Remove existing boundary edges or add absent R-to-{BFI,ESD,FRD,RNT} edges; delta(U) is unchanged.',
             'U-boundary levels: 3,4,5,7,8,9. Modify only edges between {BFI,ESD,FRD,RNT} and V-U; delta(R) and U internal edges are unchanged.',
             'Internal and R-boundary changes also change U internal density as a derived quantity because R is contained in U. This is not an independently fixed U-density experiment.',
             'Budget=3 in every graph; R-boundary variants also use budget=|delta(R)|, without duplicate rows.',
             'The full R-cut value is 7812/443 in every graph. It is an evaluated intervention, never a substitute for OPT; feasibility is recorded separately.',
             'candidateoptimal means the R boundary ALONE attains OPT within the row budget. rinoptimal refers only to the saved representative partition, not all co-optima.',
             'Additions are marked synthetic in memory. No passenger volume is invented. A solver/ranking copy sets all node masses and passenger fields to one; these fields are bookkeeping for the uniform/topological objective, not traffic estimates.',
             'Exact method: unchanged air.partitions and air.optimize(uu), with unit-capacity Gomory-Hu contraction, exhaustive local partition enumeration and the existing exact dominance DP.',
             'Completeness is the existing air.py proof: inseparable atoms preserve all <=b cuts; enumerate all affordable bundle deletions per branch; combine every feasible local partition. No restriction to a single tree edge or to R/U cuts.',
             f'Exact limits: {seconds} seconds per graph; {limit} local visits and {limit} DP combinations. Any limit failure leaves exact score and BC/OPT blank and records unverified plus its reason.',
             f'Python {platform.python_version()}; NetworkX {nx.__version__}; platform {platform.platform()}',
             'Input/code SHA256:']
    lines.extend(str(path.relative_to(root))+' '+hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)
    lines += ['',f'Distinct graphs: {len({r["graphhash"] for r in rows})}; objective/budget rows: {len(rows)}.',
              f'Exact rows: {sum(r["exactstatus"]=="exact" for r in rows)}/{len(rows)}; total wall seconds: {elapsed:.3f}.',
              f'Original: OPT={originalrow["exact"]}; BC={originalrow["bcscore"]}; boundary ranks={originalrow["rboundaryranks"]}.',
              '', 'RESULTS', 'variant | seeds | parameter | budget | R/U boundaries | R density | OPT | BC | BC/OPT | R alone optimal | R in representative | optimal sizes']
    for row in rows:
        lines.append(f'{row["variant"]} | {row["seeds"]} | {row["level"]} | {row["budget"]} | {row["rboundarysize"]}/{row["uboundarysize"]} | {row["rdensity"]} | {row["exact"] or "unverified"} | {row["bcscore"]} | {row["bcratio"] or "unverified"} | {row["candidateoptimal"]} | {row["rinoptimal"]} | {row["optimalcomponents"]}')
        if row['exacterror']: lines.append('  Exact search: '+row['exacterror'])
    lines += ['', 'CONTROLLED COMPARISONS AT FIXED BUDGET 3', 'feature | level | distinct graphs | R alone optimal / verified | BC/OPT range among verified']
    for feature in levels:
        for level in levels[feature]:
            group = [r for r in rows if r['feature']==feature and r['level']==level and r['budget']==3]
            checked = [r for r in group if r['exactstatus']=='exact']
            ratios = [Fraction(r['bcratio']) for r in checked]
            lines.append(f'{feature} | {level} | {len(group)} | {sum(r["candidateoptimal"] for r in checked)}/{len(checked)} | '+(f'{float(min(ratios)):.6f}..{float(max(ratios)):.6f}' if ratios else 'unverified'))
    lines += ['', 'COMPLETE SYNTHETIC MODIFICATION LOG']
    seen = set()
    for row in rows:
        if row['variant'] in seen: continue
        seen.add(row['variant'])
        lines.append(row['variant']+' seeds='+str(row['seeds'])+' add='+json.dumps(row['added'])+' remove='+json.dumps(row['removed']))
    lines += ['', 'RECOMMENDED FIGURE',
              'One three-panel plot with x=R internal density, R boundary width, and U boundary width, respectively; y=BC/OPT. Show separate seed traces at budget 3 and mark the original measured graph.',
              'In the R-width panel, distinguish the matched-budget b=|delta(R)| traces. Mark candidate-cut infeasibility separately from feasible suboptimality; show unverified optima as missing, never as zero or candidate-normalized ratios.',
              'Caption every perturbed graph as a synthetic counterfactual. Two deterministic seeds are placement sensitivity checks, not a representative statistical sample or evidence of a universal phase transition.',
              '', 'REPRODUCTION',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py washington',
              'PYTHONDONTWRITEBYTECODE=1 python code/washington.py',
              'The pilot refuses to overwrite data/results/washington.csv or the washington report entry in data/records.json. Existing application files and results are never written.']
    return '\n'.join(lines)+'\n'

def main():
    outputs = (folder/'washington.csv',)
    if (any(path.exists() for path in outputs) or savedreports.exists('washington')): raise FileExistsError('Pilot output already exists; refusing to overwrite')
    start = perf_counter()
    base = original()
    cases = variants(base)
    rows = []
    print(f'Original graph verified; {len(cases)} distinct graphs',flush=True)
    for i,case in enumerate(cases,1):
        result = evaluate(case)
        rows.extend(result)
        print(f'{i}/{len(cases)} {case["variant"]}: '+', '.join(f'b={r["budget"]} OPT={r["exact"] or "unverified"} BC={r["bcscore"]}' for r in result),flush=True)
    elapsed = perf_counter()-start
    with outputs[0].open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(tuple,list,dict)) else v for k,v in row.items()})
    savedreports.save('washington',report(rows,base,elapsed),exclusive=True)
    print(f'Wrote {len(rows)} rows; {sum(r["exactstatus"]=="exact" for r in rows)} verified exact; {elapsed:.3f} seconds',flush=True)

if __name__=='__main__':
    main()
