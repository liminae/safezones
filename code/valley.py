# San Luis Valley topology counterfactual pilot
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
import commuting
from hospital import edges,components,score

R = frozenset(('08003','08021','08023','08079','08105','08109'))
anchors = ('08015','35055')
originalcuts = (('08015','08109'),('08023','35055'))
levels = (5,6,7,8,10,12,15)
seconds = 90
limit = 200000
source = root/'data/raw/table1.xlsx'

def signature(graph):
    return hashlib.sha256(json.dumps([sorted(graph),edges(graph)],separators=(',',':')).encode()).hexdigest()

def original():
    assert hashlib.sha256(source.read_bytes()).hexdigest()==commuting.sourcehash
    nodes,pairs,audit = commuting.readflows(commuting.workbookrows(source))
    graph = commuting.buildgraph(nodes,pairs,1,True)
    assert (len(nodes),len(graph),graph.number_of_edges())==(3109,3105,24338)
    assert graph.subgraph(R).number_of_edges()==8 and commuting.boundary(graph,R)==originalcuts
    assert nx.is_connected(graph.subgraph(R)) and nx.is_connected(graph.subgraph(set(graph)-R))
    return graph

def perturb(base,feature,level,seed):
    graph = base.copy()
    rng = random.Random(seed)
    regional = sorted(R)
    if feature=='internal':
        if level==8:
            graph.remove_edges_from(edges(graph.subgraph(R)))
            rng.shuffle(regional)
            graph.add_edges_from((regional[i-1],regional[i],dict(synthetic=True)) for i in range(1,len(regional)))
            pool = sorted(e for e in combinations(sorted(R),2) if not graph.has_edge(*e))
            rng.shuffle(pool)
            graph.add_edges_from((u,v,dict(synthetic=True)) for u,v in pool[:3])
        else:
            present = set(edges(base.subgraph(R)))
            pool = sorted(present if level<8 else set(combinations(sorted(R),2))-present)
            rng.shuffle(pool)
            for edge in pool:
                if graph.subgraph(R).number_of_edges()==level: break
                if level<8:
                    data = dict(graph.edges[edge])
                    graph.remove_edge(*edge)
                    if not nx.is_connected(graph.subgraph(R)): graph.add_edge(*edge,**data)
                else: graph.add_edge(*edge,synthetic=True)
    elif feature=='boundary':
        if level==1:
            pool = list(originalcuts)
            rng.shuffle(pool)
            graph.remove_edge(*pool[0])
        else:
            used = {v for e in originalcuts for v in e if v in R}
            rng.shuffle(regional)
            for v in regional:
                if v in used: continue
                graph.add_edge(v,anchors[rng.randrange(2)],synthetic=True)
                used.add(v)
                if len(commuting.boundary(graph,R))==level: break
    elif feature=='placement':
        graph.remove_edges_from(originalcuts)
        rng.shuffle(regional)
        if level=='distinct': links = zip(regional[:2],anchors)
        elif level=='region': links = [(regional[0],v) for v in anchors]
        elif level=='outside': links = [(v,anchors[seed]) for v in ('08023','08109')]
        elif level=='alternate':
            pool = sorted({v for u in anchors for v in base[u]}-R-set(anchors))
            rng.shuffle(pool)
            links = zip(regional[:2],pool[:2])
        else: raise ValueError('Unknown placement')
        graph.add_edges_from((u,v,dict(synthetic=True)) for u,v in links)
    elif feature!='original': raise ValueError('Unknown feature')
    result = nx.Graph()
    result.add_nodes_from((v,dict(base.nodes[v])) for v in sorted(base))
    for u,v in edges(graph):
        result.add_edge(u,v,**(dict(base.edges[u,v]) if base.has_edge(u,v) else dict(synthetic=True)))
    return result

def validate(base,graph,feature,level):
    assert set(graph)==set(base) and nx.is_connected(graph)
    assert nx.is_connected(graph.subgraph(R)) and nx.is_connected(graph.subgraph(set(graph)-R))
    assert edges(graph.subgraph(set(graph)-R))==edges(base.subgraph(set(base)-R))
    changed = set(edges(base))^set(edges(graph))
    if feature=='internal':
        assert graph.subgraph(R).number_of_edges()==level
        assert all(set(e)<=R for e in changed)
        assert commuting.boundary(graph,R)==originalcuts
    elif feature in ('boundary','placement'):
        assert all((u in R)!=(v in R) for u,v in changed)
        assert edges(graph.subgraph(R))==edges(base.subgraph(R))
        assert len(commuting.boundary(graph,R))==(level if feature=='boundary' else 2)
    else: assert not changed

def variants(base):
    specs = [('original',0,0)]+[('internal',level,seed) for level in levels for seed in (0,1)]
    specs += [('boundary',level,seed) for level in (1,3,4) for seed in (0,1)]
    specs += [('placement',level,seed) for level in ('distinct','region','outside','alternate') for seed in (0,1)]
    found = {}
    for feature,level,seed in specs:
        graph = perturb(base,feature,level,seed)
        validate(base,graph,feature,level)
        key = signature(graph)
        if key in found:
            found[key]['aliases'].append(dict(feature=feature,level=level,seed=seed))
            continue
        found[key] = dict(variant='original' if feature=='original' else f'{feature}{level}s{seed}',feature=feature,level=level,seed=seed,
                          aliases=[dict(feature=feature,level=level,seed=seed)],graphhash=key,
                          added=sorted(set(edges(graph))-set(edges(base))),removed=sorted(set(edges(base))-set(edges(graph))))
    assert 20<=len(found)<=30
    return list(found.values())

def bcorder(graph):
    values = {tuple(sorted(e)):2*v for e,v in nx.edge_betweenness_centrality(graph,normalized=False).items()}
    return tuple(sorted(values,key=lambda e:(-values[e],e))),values

def adapted(graph):
    result = graph.copy()
    nx.set_node_attributes(result,1,'inbound')
    nx.set_edge_attributes(result,1,'passengers')
    return result

def expired(signum,frame):
    raise TimeoutError('Exact search exceeded its per-variant time cap')

def solve(graph,b,cap=seconds,work=limit):
    start = perf_counter()
    previous = signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,cap)
    try:
        data = air.partitions(adapted(graph),b,limit=work)
        values,stats = air.optimize(data,'uu',limit=work)
        return dict(status='exact',values={j:(Fraction(air.numerator(data,'uu',state),len(graph)),state[4]) for j,state in values.items()},
                    stats={**{k:v for k,v in data['stats'].items() if k!='core'},**stats},error='',seconds=perf_counter()-start)
    except (RuntimeError,TimeoutError) as error:
        return dict(status='unverified',values={},stats={},error=str(error),seconds=perf_counter()-start)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)

def protection(graph,cuts):
    groups = sorted(components(graph,cuts),key=lambda c:(-len(c),c))
    value = Fraction(len(graph)**2-sum(len(c)**2 for c in groups),len(graph))
    p,a = commuting.weights(graph,'uniform','counties')
    assert score(graph,cuts,p,a)==value
    return value,groups

def evaluate(base,case):
    start = perf_counter()
    graph = perturb(base,case['feature'],case['level'],case['seed'])
    validate(base,graph,case['feature'],case['level'])
    assert signature(graph)==case['graphhash']
    boundary = commuting.boundary(graph,R)
    budgets = sorted({2,len(boundary)}) if case['feature']=='boundary' else [2]
    exact = solve(graph,max(budgets))
    print(f'  Exact {exact["status"]}: {exact["seconds"]:.2f}s; computing ordinary BC',flush=True)
    bcstart = perf_counter()
    order,values = bcorder(graph)
    bcseconds = perf_counter()-bcstart
    ranks = {e:i+1 for i,e in enumerate(order)}
    candidate,groups = protection(graph,boundary)
    assert candidate==Fraction(37188,3105) and list(map(len,groups))==[3099,6]
    inside = sorted({v for e in boundary for v in e if v in R})
    outside = sorted({v for e in boundary for v in e if v not in R})
    rows = []
    for b in budgets:
        chosen = order[:b]
        value,groups = protection(graph,chosen)
        feasible = len(boundary)<=b
        row = dict(case,synthetic=case['feature']!='original',n=len(graph),m=graph.number_of_edges(),budget=b,
                   budgettype='fixed and matching' if b==2 and len(boundary)==2 else 'fixed' if b==2 else 'matching',
                   rvertices=sorted(R),rinternaledges=graph.subgraph(R).number_of_edges(),rdensity=str(Fraction(graph.subgraph(R).number_of_edges(),15)),
                   rboundary=boundary,rboundarysize=len(boundary),rboundaryranks=[ranks[e] for e in boundary],rboundarybc=[values[e] for e in boundary],
                   rendpoints=inside,oendpoints=outside,rendpointcount=len(inside),oendpointcount=len(outside),
                   endpointconfiguration=f'{len(inside)} regional / {len(outside)} outside endpoints',
                   graphconnected=nx.is_connected(graph),rconnected=nx.is_connected(graph.subgraph(R)),
                   rcomplementconnected=nx.is_connected(graph.subgraph(set(graph)-R)),bccuts=chosen,bccutranks=list(range(1,b+1)),
                   bccutvalues=[values[e] for e in chosen],bcboundaryselected=sum(e in boundary for e in chosen),
                   bcscore=str(value),bcdecimal=float(value),bccomponents=list(map(len,groups)),
                   candidatefeasible=feasible,candidatescore=str(candidate) if feasible else 'infeasible',candidatedecimal=float(candidate) if feasible else '',
                   exactstatus=exact['status'],exact='',exactdecimal='',optimalcut=[],optimalpartition=[],optimalcomponents=[],optimalregions=[],
                   bcratio='',bcratiodecimal='',bcoptimal='',candidateoptimal='' if feasible else 'infeasible',rinoptimal='',
                   anotherbetter='' if feasible else 'infeasible comparison',otherregionbetter='',exactstats=exact['stats'],exactseconds=exact['seconds'],exacterror=exact['error'],bcseconds=bcseconds)
        if b in exact['values']:
            optimum,cuts = exact['values'][b]
            replay,groups = protection(graph,cuts)
            assert replay==optimum>=value and len(cuts)<=b
            assert not feasible or optimum>=candidate
            regions = [dict(fips=list(c),names=[graph.nodes[v]['name'] for v in c],states=sorted({graph.nodes[v]['state'] for v in c})) for c in groups[1:]]
            otherbetter = []
            for c in groups[1:]:
                cutsforregion = commuting.boundary(graph,c)
                regionvalue,unused = protection(graph,cutsforregion)
                if set(c)!=R and len(cutsforregion)<=b and regionvalue>candidate:
                    otherbetter.append(dict(fips=list(c),score=str(regionvalue),boundary=cutsforregion))
            row.update(exact=str(optimum),exactdecimal=float(optimum),optimalcut=cuts,optimalpartition=groups,
                       optimalcomponents=list(map(len,groups)),optimalregions=regions,
                       bcratio=str(value/optimum) if optimum else '',bcratiodecimal=float(value/optimum) if optimum else '',bcoptimal=value==optimum,
                       candidateoptimal=candidate==optimum if feasible else 'infeasible',rinoptimal=any(set(c)==R for c in groups),
                       anotherbetter=optimum>candidate if feasible else 'infeasible comparison',otherregionbetter=otherbetter)
        row['seconds'] = perf_counter()-start
        rows.append(row)
    return rows

def report(rows,base,elapsed,runhash=None):
    verified = [r for r in rows if r['exactstatus']=='exact']
    feasible = [r for r in verified if r['candidatefeasible']]
    originalrow = next(r for r in rows if r['feature']=='original')
    runhash = runhash or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    paths = [source,root/'code/air.py',root/'code/commuting.py',root/'code/hospital.py',Path(__file__),root/'code/tests.py',root/'data/results/washington.csv']
    lines = ['SAN LUIS VALLEY GENERALITY PILOT','',
             'Scope: deterministic synthetic topology counterfactuals, not measured commuting changes, transport plans or geographic conclusions.',
             'Baseline: kappa=1, strict direction-specific flow>MOE filter; aggregate both directed flows; select original largest component once. 3105 vertices, 24338 edges. No vertices are dropped or giant reselected after perturbation.',
             'R: '+json.dumps([dict(fips=v,name=base.nodes[v]['name']) for v in sorted(R)]),
             'Original R: 6 counties, 8 internal edges, density=8/15, boundary='+str(originalcuts)+'. R and its complement are connected.',
             'All variants keep the outside induced graph identical. Both R and its complement remain connected, and so does the full graph.',
             'Objective: uniform source over the same 3105 vertices, unit target, static fixed cuts; score=N-sum(size squared)/N. No changes to sequential semantics. Fractions are authoritative; expected protection is not the deterministic size of the detached region.',
             'Candidate R cut: 37188/3105=12396/1035 when feasible. Infeasible cuts have the literal infeasible score, no decimal, and no BC/OPT substitution.',
             'BC: 2*NetworkX edge_betweenness_centrality(normalized=False), exactly the ordinary-BC expression in commuting.rankings; canonical FIPS ties. Computed once on each variant, not after cutting. BCw/weak/strong are not calculated. Full-network centrality is used; high-BC edges are not assumed to bound a coherent outside region.',
             'Solver: existing air.partitions plus air.optimize(uu), also used by the application sweeps and Washington pilot. Temporary inbound/passengers adapter values are 1; no real flow or county metadata are changed. No invented traffic weights are assigned to synthetic links.',
             'Completeness: unit-capacity Gomory-Hu contraction keeps every <=b partition; all affordable local bundle deletions are enumerated and all global combinations are considered by the existing exact dominance DP. This is not a search restricted to R or single Gomory-Hu edges. See code/air.py:21-68.',
             f'Limits: {seconds} seconds per variant exact search; {limit} local visits / DP combinations. Failure leaves OPT and ratio blank, with exactstatus=unverified and a reason. Limits apply to the largest requested budget of the variant; no smaller-budget optimum is inferred from a timed-out run.',
             'Local workbook is parsed once and the baseline graph reused in memory. The previous /private/tmp application-sweep cache was absent. No original sweep is run or written. The saved baseline exact cut/score and BC choices serve as independent regression anchors.',
             'Internal: counts 5,6,7,8,10,12,15; seeds 0,1. Below 8 remove shuffled original edges only if connected; above 8 add shuffled missing edges. At 8, build a shuffled spanning path plus 3 shuffled absent chords to test different placements at the empirical edge count.',
             'Boundary width: 1,3,4, seeds 0,1, plus measured width 2. At width 1 remove one original link. For larger widths add links on distinct unused regional vertices, to the two original outside anchors. Evaluate fixed b=2 and matching b=width.',
             'Boundary placement at width 2: distinct changes regional endpoints with original outside anchors; region concentrates both links on one regional vertex; outside concentrates both on Chaffee or Taos while retaining original regional endpoints; alternate uses two shuffled one-hop neighbors of the original outside anchors and distinct shuffled regional endpoints. Seeds 0,1; R internal edges stay unchanged.',
             'Duplicate graphs are evaluated once and retain parameter/seed aliases. A baseline control is not relabeled synthetic. All modifications and endpoints are recorded below. Two seeds are placement checks, not representative random samples.',
             'candidateoptimal: R boundary alone reaches OPT. rinoptimal: R occurs in the saved representative. anotherbetter: a strictly better partition than feasible R exists. otherregionbetter lists only non-R components of the saved optimum whose own boundary alone beats the numerical R benchmark; an empty list does not exclude other co-optima. Multiple separated groups can improve the objective without one group alone beating R. No uniqueness claims.',
             f'Python {platform.python_version()}; NetworkX {nx.__version__}; platform {platform.platform()}',
             'Experiment-generation script SHA256: '+runhash,
             'Saved results CSV SHA256: '+hashlib.sha256((folder/'valley.csv').read_bytes()).hexdigest(),
             'INPUT/CURRENT CODE SHA256']
    lines.extend(str(p)+' '+hashlib.sha256(p.read_bytes()).hexdigest() for p in paths)
    lines += ['',f'Distinct graphs: {len({r["graphhash"] for r in rows})}; cases: {len(rows)}; exact verified: {len(verified)}; unverified: {len(rows)-len(verified)}.',
              f'Total wall seconds: {elapsed:.3f}; slowest exact seconds: {max(r["exactseconds"] for r in rows):.3f}; slowest BC seconds: {max(r["bcseconds"] for r in rows):.3f}.',
              f'Baseline OPT={originalrow["exact"] or "unverified"}; BC={originalrow["bcscore"]}; R boundary ranks={originalrow["rboundaryranks"]}.',
              f'BC positive: {sum(Fraction(r["bcscore"])>0 for r in rows)}/{len(rows)}; BC optimal: {sum(r["bcoptimal"] is True for r in verified)}/{len(verified)} verified.',
              f'R optimal: {sum(r["candidateoptimal"] is True for r in feasible)}/{len(feasible)} feasible verified; infeasible: {sum(not r["candidatefeasible"] for r in rows)}.',
              '', 'RESULTS', 'variant | feature/level/seed | b | R internal/boundary | endpoint counts R/O | boundary ranks | BC boundary hits | R score | OPT | BC | ratio | R optimal | optimal sizes']
    for r in rows:
        lines.append(f'{r["variant"]} | {r["feature"]}/{r["level"]}/{r["seed"]} | {r["budget"]} | {r["rinternaledges"]}/{r["rboundarysize"]} | {r["rendpointcount"]}/{r["oendpointcount"]} | {r["rboundaryranks"]} | {r["bcboundaryselected"]} | {r["candidatescore"]} | {r["exact"] or "unverified"} | {r["bcscore"]} | {r["bcratio"] or "undefined"} | {r["candidateoptimal"]} | {r["optimalcomponents"]}')
        if r['exacterror']: lines.append('  Exact issue: '+r['exacterror'])
    lines += ['', 'OPTIMAL REGIONS AND RANKED CUTS']
    for r in verified:
        lines.append(f'{r["variant"]} b={r["budget"]}: optimal cut='+json.dumps(r['optimalcut'])+'; regions='+json.dumps(r['optimalregions'])+'; BC cut='+json.dumps(r['bccuts'])+'; better single non-R regions='+json.dumps(r['otherregionbetter']))
    lines += ['', 'EQUAL INTERNAL COUNTS: PLACEMENT COMPARISONS']
    for level in levels:
        group = [r for r in verified if r['feature']=='internal' and r['level']==level]
        lines.append(f'Internal {level}: '+ '; '.join(f'{r["variant"]} OPT={r["exact"]}, R optimal={r["candidateoptimal"]}, ranks={r["rboundaryranks"]}' for r in group))
    lines += ['', 'COMPLETE MODIFICATION LOG']
    seen = set()
    for r in rows:
        if r['variant'] in seen: continue
        seen.add(r['variant'])
        lines.append(r['variant']+' aliases='+json.dumps(r['aliases'])+' graphhash='+r['graphhash']+' add='+json.dumps(r['added'])+' remove='+json.dumps(r['removed']))
    with (root/'data/results/washington.csv').open(newline='') as stream: washington = list(csv.DictReader(stream))
    zerow = sum(Fraction(r['bcscore'])==0 for r in washington)
    internal = {}
    for r in verified:
        if r['feature']=='internal': internal.setdefault(r['rinternaledges'],set()).add(r['exact'])
    differences = [k for k,values in internal.items() if len(values)>1]
    placement = [r for r in rows if r['feature']=='placement']
    lines += ['', 'COMPARISON WITH WASHINGTON',
              f'BC failure persists: commuting BC=0 in {sum(Fraction(r["bcscore"])==0 for r in rows)}/{len(rows)} cases; Washington {zerow}/{len(washington)}. These case totals have different grids and are not comparative population estimates.',
              ('Equal internal counts give different exact values at commuting counts '+str(differences)+'.' if differences else 'No equal-internal-count optimum differences were found in this commuting pilot. This does not reproduce Washington\'s placement-dependent optima at 9 and 15 internal edges.'),
              'Distinct OPT values over all commuting internal-structure variants: '+str(sorted({r['exact'] for r in verified if r['feature']=='internal'}))+'. The limited placements do not prove density sufficient. Across the full pilot, the same 8 internal edges occur with R optimal, feasible but suboptimal, or infeasible, depending on boundary width and budget; density alone does not describe those changes.',
              'Boundary width: compare fixed b=2 rows below; infeasibility must be separated from a feasible but suboptimal R cut.',
              *[f'  {r["variant"]} b={r["budget"]}: R={r["candidatescore"]}, OPT={r["exact"] or "unverified"}, sizes={r["optimalcomponents"]}' for r in rows if r['feature']=='boundary'],
              f'Boundary placement: {len(placement)} configurations have identical R size, 8 internal edges, boundary size 2 and candidate value 12396/1035; observed canonical R-rank lists='+str([r['rboundaryranks'] for r in placement])+'. Changes in ranks need not change top-2 protection.',
              f'Placement outcomes: R optimal in {sum(r["candidateoptimal"] is True for r in placement)}/{len(placement)}; BC positive in {sum(Fraction(r["bcscore"])>0 for r in placement)}/{len(placement)}.',
              'The width-three matching-budget solution separates groups of sizes 3+2+1 rather than R alone: both detach six counties, but fragmentation lowers the component-square penalty. Width four instead selects 3+2+1+1. These statements refer to the saved representative solutions above, not uniqueness. Width one at b=2 retains R and adds Big Bend, so R alone is suboptimal while R remains in the optimum.',
              'Non-generalizing details: Washington U is a specific larger 13-airport set; this pilot defines no analogous outside cohort. Dispersed high-BC county links are not evidence of a coherent competing region. Washington had 4 positive-BC cases, one BC-optimal case and U-optimal transitions; no corresponding commuting conclusion is assumed. See the case counts and region identities above.',
              'The perturbations are counterfactual additions/removals after fixing kappa=1, not a reliability sweep. Fixed nodes and outside graph rule out the giant-component vertex-loss effect seen in threshold sweeps.',
              '', 'REPRODUCTION',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py valley',
              'PYTHONDONTWRITEBYTECODE=1 python code/valley.py',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py valley',
              'The experiment refuses to overwrite data/results/valley.csv and the valley report entry. Tests replay saved cuts without rerunning full-graph BC or exact searches. Small test graphs check exactness and BC conventions independently.']
    return '\n'.join(lines)+'\n'

def main():
    outputs = (folder/'valley.csv',)
    if (any(p.exists() for p in outputs) or savedreports.exists('valley')): raise FileExistsError('Pilot output exists; refusing to overwrite')
    start = perf_counter()
    runhash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    base = original()
    cases = variants(base)
    rows = []
    print(f'Baseline 3105/24338 verified; {len(cases)} distinct graphs',flush=True)
    with outputs[0].open('x',newline='') as stream:
        writer = None
        for i,case in enumerate(cases,1):
            print(f'{i}/{len(cases)} {case["variant"]}',flush=True)
            result = evaluate(base,case)
            rows.extend(result)
            if writer is None:
                writer = csv.DictWriter(stream,fieldnames=list(result[0]))
                writer.writeheader()
            for row in result: writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,tuple,dict)) else v for k,v in row.items()})
            stream.flush()
            print('  '+', '.join(f'b={r["budget"]} OPT={r["exact"] or "unverified"} BC={r["bcscore"]} ranks={r["rboundaryranks"]}' for r in result),flush=True)
    elapsed = perf_counter()-start
    savedreports.save('valley',report(rows,base,elapsed,runhash),exclusive=True)
    print(f'Wrote {len(rows)} cases; {sum(r["exactstatus"]=="exact" for r in rows)} verified; {elapsed:.3f} seconds',flush=True)

if __name__=='__main__':
    main()
