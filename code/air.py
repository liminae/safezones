# Exact static airport route application
import records as savedreports
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter,defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from time import perf_counter
import networkx as nx
from hospital import components,score

root = Path(__file__).resolve().parent.parent
sourcehash = '602515528bcc33e64eee0a19181795e296586c0379b1a62bfadbf3d7a925af86'
states = set('AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC'.split())
washington = tuple('DHB FBS KEH LKE LPS PUL RCE RSJ WSX'.split())
proof = '''COMPLETENESS AND EXACT ARITHMETIC
1. Give each original route unit capacity. In a Gomory-Hu tree the minimum tree
edge on an s-t path equals their minimum cut value. Retain tree edges of weight
greater than B. Their components are atoms that cannot be split by <=B removals.
An atom need not induce a connected subgraph; its vertices remain connected in
the surviving ORIGINAL graph under every allowed removal set.
2. Restore cuts internal to each final component. This preserves the objective,
reduces or preserves cardinality and passenger disruption, and leaves exactly
the component boundary. Between two atoms either the entire bundle of original
routes crosses the partition or none does. Bundle cost is its route count.
3. Choose the largest atom as root, and remove it from the quotient graph.
Different remaining branches meet only at that root. In each branch plus root,
enumerate EVERY bundle subset with total cardinality <=B. Deduplicate by its
resulting component partition and restore redundant internal bundles. Every
global feasible partition restricts to one of these local partitions; conversely
every choice of local partitions with total cost <=B gives a feasible partition.
All local root components merge into one global component. This is not a search
restricted to individual Gomory-Hu tree cuts.
4. For detached components retain total size n, total inbound mass w, and the
sum q of size squared (uniform/uniform), size times mass (mixed), or mass squared
(inbound/inbound). Let N,W be the corresponding totals over the whole graph.
The three objective numerators are N^2-(N-n)^2-q,
N*W-(N-n)*(W-w)-q, and W^2-(W-w)^2-q, respectively.
Uniform/inbound and inbound/uniform share the same numerator, with different
normalizing denominators. This explains their identical optimal partitions.
5. Dynamic programming combines every branch option at each exact spent budget.
For size squared, equal-n states need only the smallest q. For the mixed case,
compare states with equal n. After a future detached size f their difference is
(N-n-f)*delta(w)-delta(q). It is affine for 0<=f<=remaining branch size.
For mass squared the difference after future mass f is likewise affine, with
each state's contribution 2*W*w-w^2-q-2*f*w. Compare both endpoints of the
interval 0<=f<=remaining branch mass. A state strictly inferior at both endpoints
can never win under ANY common future completion. Equal endpoint pairs are
deduplicated deterministically. Otherwise keep the state. Thus pruning is a
proved dominance rule, not an approximation. The count and mass bounds may be
loose; this only retains extra states. Original graph replay verifies each result.
6. The secondary problem maximizes score/disrupted passenger-segments among
NONEMPTY cuts with <=6 routes. Dinkelbach iteration exactly maximizes
denominator*objective numerator - numerator*disrupted passengers. The penalty
is additive and the same dominance proof applies after adding it to q. Starting
at zero, each positive residual strictly improves the rational ratio. The finite
cut space ensures termination at residual zero, which certifies global ratio
optimality within the six-route cap. This is NOT an unlimited-cardinality cost
optimum or a passenger-budget optimum. All arithmetic used in optimization and
comparisons is integer or Fraction. BCw alone uses NetworkX floating distances.
Deterministic ties use fewer routes, then canonical lexicographic route tuples.
There is no polynomial-time claim: local enumeration and Pareto fronts can grow.
Explicit local-subset/state limits raise an error before any incomplete result is
reported as exact. The report records the actual exhaustive work completed.
Reference: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.gomory_hu_tree.html
'''

def edges(graph):
    return tuple(sorted(tuple(sorted(e)) for e in graph.edges))

def integer(text):
    value = Decimal(text)
    if not value.is_finite() or value<0 or value!=int(value):
        raise ValueError('Traffic quantities must be nonnegative integers')
    return int(value)

def buildgraph(rows):
    graph,counts,months,years = nx.Graph(),Counter(),set(),Counter()
    for row in rows:
        counts['rows'] += 1
        years[row['YEAR']] += 1
        if row['YEAR']!='2025' or row['CLASS']!='F': continue
        passengers = integer(row['PASSENGERS'])
        if not passengers: continue
        counts['positiveF'] += 1
        u,v = row['ORIGIN'],row['DEST']
        if row['ORIGIN_STATE_ABR'] not in states or row['DEST_STATE_ABR'] not in states: continue
        if u==v: continue
        counts['retained'] += 1
        months.add(integer(row['MONTH']))
        for prefix,node in (('ORIGIN',u),('DEST',v)):
            city,state = row[prefix+'_CITY_NAME'],row[prefix+'_STATE_ABR']
            if node not in graph:
                graph.add_node(node,code=node,city=city,state=state,inbound=0,outbound=0)
            elif (graph.nodes[node]['city'],graph.nodes[node]['state'])!=(city,state):
                raise ValueError('Inconsistent airport metadata for '+node)
        graph.nodes[u]['outbound'] += passengers
        graph.nodes[v]['inbound'] += passengers
        if not graph.has_edge(u,v): graph.add_edge(u,v,passengers=0,departures=0,seats=0)
        for name,field in (('passengers','PASSENGERS'),('departures','DEPARTURES_PERFORMED'),('seats','SEATS')):
            graph.edges[u,v][name] += integer(row[field])
    if not graph: raise ValueError('No qualifying routes')
    groups = sorted((tuple(sorted(c)) for c in nx.connected_components(graph)),key=lambda c:(-len(c),c))
    chosen = graph.subgraph(groups[0])
    result = nx.Graph()
    result.add_nodes_from((v,dict(chosen.nodes[v])) for v in sorted(chosen))
    result.add_edges_from((u,v,dict(chosen.edges[u,v])) for u,v in edges(chosen))
    audit = dict(counts,years=dict(years),months=sorted(months),pregiantnodes=len(graph),
                 pregiantedges=graph.number_of_edges(),componentcounts=list(map(len,groups)),
                 excludedairports=[list(c) for c in groups[1:]])
    return result,audit

def readgraph(path):
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [v for v in archive.namelist() if v.lower().endswith('.csv')]
            if len(names)!=1: raise ValueError('Expected one CSV in the archive')
            with archive.open(names[0]) as raw:
                return buildgraph(csv.DictReader(io.TextIOWrapper(raw,encoding='utf-8-sig')))
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return buildgraph(csv.DictReader(stream))

def objectives(graph,pname,aname):
    if pname not in ('uniform','inbound') or aname not in ('airports','inbound'):
        raise ValueError('Unknown objective')
    p = {v:1 if pname=='uniform' else graph.nodes[v]['inbound'] for v in graph}
    total = sum(p.values())
    if total<=0: raise ValueError('Source population is empty')
    return {v:Fraction(w,total) for v,w in p.items()}, {v:1 if aname=='airports' else graph.nodes[v]['inbound'] for v in graph}

def contraction(graph,b):
    if not isinstance(b,int) or isinstance(b,bool) or b<0: raise ValueError('Invalid budget')
    if not graph or graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph) or not nx.is_connected(graph):
        raise ValueError('Require a connected simple undirected graph')
    capacity = nx.Graph()
    capacity.add_nodes_from(sorted(graph))
    capacity.add_edges_from((u,v,dict(capacity=1)) for u,v in edges(graph))
    tree = nx.gomory_hu_tree(capacity,capacity='capacity') if len(graph)>1 else capacity
    forest = nx.Graph()
    forest.add_nodes_from(capacity)
    forest.add_edges_from((u,v) for u,v,d in tree.edges(data=True) if d['weight']>b)
    atoms = tuple(sorted(tuple(sorted(c)) for c in nx.connected_components(forest)))
    owner = {v:i for i,atom in enumerate(atoms) for v in atom}
    bundles = defaultdict(list)
    for u,v in edges(graph):
        if owner[u]!=owner[v]: bundles[tuple(sorted((owner[u],owner[v])))].append((u,v))
    quotient = nx.Graph()
    quotient.add_nodes_from(range(len(atoms)))
    quotient.add_edges_from((u,v,dict(cuts=tuple(cuts),cost=len(cuts))) for (u,v),cuts in sorted(bundles.items()))
    core = min(quotient,key=lambda i:(-len(atoms[i]),atoms[i]))
    return atoms,quotient,core

def partitions(graph,b=6,limit=2000000):
    start = perf_counter()
    atoms,quotient,core = contraction(graph,b)
    outside = quotient.copy()
    outside.remove_node(core)
    branches,visits = [],0
    for branch in sorted(nx.connected_components(outside),key=min):
        local = quotient.subgraph(branch|{core}).copy()
        bundles = [(u,v,d['cost'],d['cuts']) for u,v,d in sorted(local.edges(data=True))]
        found = {}
        def visit(index,spent):
            nonlocal visits
            visits += 1
            if visits>limit: raise RuntimeError(f'Exact local enumeration stopped at {visits} subsets; limit={limit}')
            groups = tuple(sorted(tuple(sorted(c)) for c in nx.connected_components(local)))
            if groups not in found:
                labels = {v:i for i,c in enumerate(groups) for v in c}
                cuts = tuple(sorted(e for u,v,cost,original in bundles if labels[u]!=labels[v] for e in original))
                detached = tuple(sorted(tuple(sorted(v for i in group for v in atoms[i])) for group in groups if core not in group))
                sizes = [len(c) for c in detached]
                masses = [sum(graph.nodes[v]['inbound'] for v in c) for c in detached]
                found[groups] = dict(cuts=cuts,cost=len(cuts),groups=detached,n=sum(sizes),w=sum(masses),
                                     uu=sum(n*n for n in sizes),mixed=sum(n*w for n,w in zip(sizes,masses)),
                                     pp=sum(w*w for w in masses),passengers=sum(graph.edges[e]['passengers'] for e in cuts))
            for j in range(index,len(bundles)):
                u,v,cost,original = bundles[j]
                if spent+cost>b: continue
                local.remove_edge(u,v)
                visit(j+1,spent+cost)
                local.add_edge(u,v)
        visit(0,0)
        vertices = tuple(v for i in sorted(branch) for v in atoms[i])
        branches.append(dict(nodes=vertices,n=len(vertices),w=sum(graph.nodes[v]['inbound'] for v in vertices),
                             options=sorted(found.values(),key=lambda d:(d['cost'],d['cuts']))))
    stats = dict(budget=b,atoms=len(atoms),atomsizes=dict(sorted(Counter(map(len,atoms)).items())),
                 core=atoms[core],bundles=quotient.number_of_edges(),branches=len(branches),
                 maximumbranchblocks=max((len(c) for c in nx.connected_components(outside)),default=0),
                 localsubsets=visits,localpartitions=sum(len(d['options']) for d in branches),seconds=perf_counter()-start)
    return dict(budget=b,n=len(graph),w=sum(graph.nodes[v]['inbound'] for v in graph),branches=branches,stats=stats)

def numerator(catalog,mode,state):
    n,w,q,cost,cuts = state
    N,W = catalog['n'],catalog['w']
    if mode=='uu': return N*N-(N-n)**2-q
    if mode=='mixed': return N*W-(N-n)*(W-w)-q
    if mode=='pp': return W*W-(W-w)**2-q
    raise ValueError('Unknown optimization mode')

def frontier(values,mode,N,W,remainingn,remainingw,penalty):
    num,den = penalty.numerator,penalty.denominator
    buckets = defaultdict(dict)
    for state in values:
        n,w,q,cost,cuts = state
        Q = den*q+num*cost
        if mode=='uu': low = high = -Q
        elif mode=='mixed': low,high = den*(N-n-remainingn)*w-Q,den*(N-n)*w-Q
        else:
            high = den*(2*W*w-w*w)-Q
            low = high-2*den*remainingw*w
        key = n if mode!='pp' else 0
        pair = high,low
        old = buckets[key].get(pair)
        if old is None or cuts<old[4]: buckets[key][pair] = state
    kept = []
    for bucket in buckets.values():
        prior,best,current = None,None,None
        for (high,low),state in sorted(bucket.items(),reverse=True):
            if high!=current:
                prior,best,current = best,best,high
            if prior is None or low>=prior: kept.append(state)
            if best is None or low>best: best = low
    return kept

def optimize(catalog,mode,penalty=Fraction(0),limit=2000000,nonempty=False):
    if mode not in ('uu','mixed','pp'): raise ValueError('Unknown optimization mode')
    b,N,W = catalog['budget'],catalog['n'],catalog['w']
    remainingn = sum(d['n'] for d in catalog['branches'])
    remainingw = sum(d['w'] for d in catalog['branches'])
    table,maximum,combinations = {0:[(0,0,0,0,())]},1,0
    for branch in catalog['branches']:
        remainingn -= branch['n']
        remainingw -= branch['w']
        candidates = defaultdict(list)
        for spent,values in table.items():
            for option in branch['options']:
                cost = spent+option['cost']
                if cost>b: continue
                for n,w,q,passengers,cuts in values:
                    combinations += 1
                    if combinations>limit: raise RuntimeError(f'Exact DP stopped at {combinations} combinations; limit={limit}')
                    candidates[cost].append((n+option['n'],w+option['w'],q+option[mode],passengers+option['passengers'],tuple(sorted(cuts+option['cuts']))))
        table = {cost:frontier(values,mode,N,W,remainingn,remainingw,penalty) for cost,values in candidates.items()}
        maximum = max(maximum,sum(map(len,table.values())))
    result = {}
    for b in range(catalog['budget']+1):
        choices = [state for cost,values in table.items() if cost<=b and (cost or not nonempty) for state in values]
        if not choices: continue
        result[b] = min(choices,key=lambda s:(-(penalty.denominator*numerator(catalog,mode,s)-penalty.numerator*s[3]),len(s[4]),s[4]))
    return result,dict(combinations=combinations,maximumstates=maximum)

def efficiency(catalog,mode):
    ratio,iterations = Fraction(0),0
    while True:
        values,stats = optimize(catalog,mode,ratio,nonempty=True)
        choices = [state for state in values.values() if state[3]>0]
        if not choices: raise ValueError('No nonempty separating cut within budget')
        state = max(choices,key=lambda s:Fraction(numerator(catalog,mode,s),s[3]))
        new = Fraction(numerator(catalog,mode,state),state[3])
        iterations += 1
        if new==ratio:
            assert max(ratio.denominator*numerator(catalog,mode,s)-ratio.numerator*s[3] for s in choices)==0
            return state,iterations
        if new<ratio: raise AssertionError('Efficiency ratio decreased')
        ratio = new

def rankings(graph):
    order = edges(graph)
    if any(graph.edges[e]['passengers']<=0 for e in order): raise ValueError('Passenger weights must be positive')
    bc = {tuple(sorted(e)):2*v for e,v in nx.edge_betweenness_centrality(graph,normalized=False).items()}
    weighted = graph.copy()
    for e in order: weighted.edges[e]['distance'] = 1/graph.edges[e]['passengers']
    bcw = {tuple(sorted(e)):2*v for e,v in nx.edge_betweenness_centrality(weighted,normalized=False,weight='distance').items()}
    return dict(bc=tuple(sorted(order,key=lambda e:(-bc[e],e))),
                bcw=tuple(sorted(order,key=lambda e:(-bcw[e],e))),
                weak=tuple(sorted(order,key=lambda e:(graph.edges[e]['passengers'],e))),
                strong=tuple(sorted(order,key=lambda e:(-graph.edges[e]['passengers'],e))),
                bcperpassenger=tuple(sorted(order,key=lambda e:(-bc[e]/graph.edges[e]['passengers'],e))))

def cutdetails(graph,cuts):
    return [dict(route=e,passengers=graph.edges[e]['passengers'],departures=graph.edges[e]['departures'],seats=graph.edges[e]['seats']) for e in cuts]

def groupdetails(graph,cuts):
    return [dict(airports=c,size=len(c),inbound=sum(graph.nodes[v]['inbound'] for v in c)) for c in sorted(components(graph,cuts),key=lambda c:(-len(c),c))]

def cohort(graph,vertices):
    vertices = set(vertices)
    cuts = tuple(e for e in edges(graph) if (e[0] in vertices)!=(e[1] in vertices))
    internal = graph.subgraph(vertices)
    values = {}
    for pname,aname in (('uniform','airports'),('inbound','airports'),('uniform','inbound'),('inbound','inbound')):
        p,a = objectives(graph,pname,aname)
        value = score(graph,cuts,p,a)
        values[pname+'/'+aname] = dict(score=str(value),decimal=float(value))
    return dict(airports=sorted(vertices),size=len(vertices),connected=nx.is_connected(internal),
                internalroutes=internal.number_of_edges(),internalpassengers=sum(d['passengers'] for u,v,d in internal.edges(data=True)),
                boundary=cutdetails(graph,cuts),boundarysize=len(cuts),boundarypassengers=sum(graph.edges[e]['passengers'] for e in cuts),
                boundarydepartures=sum(graph.edges[e]['departures'] for e in cuts),components=groupdetails(graph,cuts),scores=values)

def analyze(graph,catalog):
    optimized,efficient,statistics = {},{},{}
    for mode in ('uu','mixed','pp'):
        optimized[mode],statistics[mode] = optimize(catalog,mode)
        efficient[mode],iterations = efficiency(catalog,mode)
        statistics[mode]['efficiencyiterations'] = iterations
    orders,records = rankings(graph),[]
    for pname,aname,mode,den in (('uniform','airports','uu',len(graph)),('inbound','airports','mixed',catalog['w']),
                               ('uniform','inbound','mixed',len(graph)),('inbound','inbound','pp',catalog['w'])):
        p,a = objectives(graph,pname,aname)
        for b in range(1,catalog['budget']+1):
            state = optimized[mode][b]
            cuts = state[4]
            value = Fraction(numerator(catalog,mode,state),den)
            assert score(graph,cuts,p,a)==value
            groups = groupdetails(graph,cuts)
            row = dict(prior=pname,target=aname,budget=b,exact=str(value),exactdecimal=float(value),optimalcut=cuts,
                       usedbudget=len(cuts),components=groups,componentsizes=[c['size'] for c in groups],
                       componentinbound=[c['inbound'] for c in groups],removedroutes=cutdetails(graph,cuts),
                       disruptedpassengers=state[3],disrupteddepartures=sum(graph.edges[e]['departures'] for e in cuts))
            for method,order in orders.items():
                result = score(graph,order[:b],p,a)
                assert result<=value
                row[method+'score'],row[method+'decimal'] = str(result),float(result)
                row[method+'ratio'] = str(result/value) if value else ''
                row[method+'ratiodecimal'] = float(result/value) if value else ''
                row[method+'cuts'] = order[:b]
            best = efficient[mode]
            costvalue = Fraction(numerator(catalog,mode,best),den)
            assert score(graph,best[4],p,a)==costvalue
            row['efficiency'] = dict(cardinalitycap=catalog['budget'],score=str(costvalue),
                                     protectionperpassenger=str(costvalue/best[3]),cuts=best[4],
                                     passengers=best[3],departures=sum(graph.edges[e]['departures'] for e in best[4]),
                                     components=groupdetails(graph,best[4]),iterations=statistics[mode]['efficiencyiterations'])
            records.append(row)
    return records,statistics,orders

def makereport(graph,audit,catalog,records,statistics,path,digest):
    nine = cohort(graph,washington)
    related = sorted({tuple(c) for branch in catalog['branches'] for opt in branch['options'] for c in opt['groups']
                      if len(c)==13 and set(washington)<=set(c) and len([e for e in edges(graph) if (e[0] in c)!=(e[1] in c)])==6})
    lines = ['BTS T-100 DOMESTIC SEGMENT 2025: EXACT STATIC AIRPORT APPLICATION',
             'Input: '+str(path),'Archive SHA256: '+digest,
             'Data dictionary: https://www.bts.gov/sites/bts.dot.gov/files/docs/explore-topics-and-geography/topics/airlines-and-airports/230176/reference-file-db28-segment-data-product.pdf',
             f'NetworkX {nx.__version__}; graph n={len(graph)}, m={graph.number_of_edges()}',json.dumps(audit,sort_keys=True),
             'Retain YEAR=2025, CLASS=F, PASSENGERS>0; both endpoints in the 48 contiguous states or DC; exclude self-loops.',
             'Sum all carriers, months, aircraft types, and both directions into each undirected airport pair; retain the giant component.',
             'Inbound/outbound masses count only retained contiguous-US scheduled passenger segments, not international, Alaska/Hawaii, or all-system airport throughput.',
             'Passenger totals are passenger-segments, not unique travelers. Departures are performed departures on the retained positive-passenger rows.',
             'Seats and performed departures exclude zero-passenger rows by the requested filter. They are not total scheduled capacity on all operations.',
             'Static unknown-source objective: a(V)-sum p(C)*a(C). No sequential spread/removal timing or production semantics are changed.',
             'Inbound target weights are exposure/proxy weights at airports, not a count of uniquely protected travelers.',
             'No minimum-traffic threshold is added. Even extremely low-volume CLASS=F routes and their airports remain in the requested network.',
             'Budgets are AT MOST b undirected relationships. Cutting a route suppresses both aggregated directions.',
             'Each row gives one deterministic cardinality-optimal representative, not every co-optimal cut. Its disruption cost need not be minimum among tied protection optima.',
             'BC and BCw are fixed original-graph ordered-pair values; BCw distance=1/passengers. All ties use canonical airport-code order.',
             'BC/passenger divides ordinary unweighted BC by passenger volume; it is a baseline, not the exact cost-efficiency solver.',
             '',proof,'CONTRACTION AND EXHAUSTIVE SEARCH',json.dumps(catalog['stats'],sort_keys=True),json.dumps(statistics,sort_keys=True),
             '', 'WASHINGTON CANDIDATE: NINE AIRPORTS',json.dumps(nine,sort_keys=True),
             '', 'RELATED THIRTEEN-AIRPORT BOUNDARY']
    for c in related:
        data = cohort(graph,c)
        data['ratiotoexact'] = {r['prior']+'/'+r['target']:str(Fraction(data['scores'][r['prior']+'/'+r['target']]['score'])/Fraction(r['exact']))
                                for r in records if r['budget']==data['boundarysize']}
        lines.append(json.dumps(data,sort_keys=True))
    if not related: lines.append('No qualifying related thirteen-airport group found.')
    lines += ['', 'EXACT RESULTS AND FIXED-RANKING BASELINES']
    for row in records:
        lines.append(f"{row['prior']}/{row['target']} b={row['budget']}: exact={row['exact']} ({row['exactdecimal']:.12f}), cuts={row['optimalcut']}, disrupted passengers={row['disruptedpassengers']}, departures={row['disrupteddepartures']}")
        lines.append('  Components: '+json.dumps(row['components'],sort_keys=True))
        lines.append('  Removed routes: '+json.dumps(row['removedroutes'],sort_keys=True))
        for method in ('bc','bcw','weak','strong','bcperpassenger'):
            lines.append(f"  {method}: score={row[method+'score']} ({row[method+'decimal']:.12f}), ratio={row[method+'ratio'] or 'NA'}, cuts={row[method+'cuts']}")
    lines += ['', 'SECONDARY EXACT PROTECTION PER DISRUPTED PASSENGER',
              'Global ratio optimum among nonempty cuts with at most six routes, irrespective of the individual table-row budget. Not a passenger-budget or unlimited-cardinality result.']
    for row in records:
        if row['budget']==catalog['budget']: lines.append(row['prior']+'/'+row['target']+': '+json.dumps(row['efficiency'],sort_keys=True))
    lines += ['', 'AIRPORT ATTRIBUTES: code, city, state, retained annual inbound/outbound passenger-segments']
    lines.extend(json.dumps(graph.nodes[v],sort_keys=True) for v in sorted(graph))
    lines += ['', 'REPRODUCE',
              'PYTHONDONTWRITEBYTECODE=1 python code/air.py',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py air',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py',
              '', 'Strongest structural finding',
              'The connected nine-airport Washington group has 30 internal routes carrying 27,447 passenger-segments but only three boundary routes carrying 183 (37 performed departures). It is cardinality-optimal at b=3 for uniform/airports, inbound/airports, and uniform/inbound.',
              'At b=5 these three objectives prefer the connected 12-airport group obtained by adding ESD, FRD and RNT to the nine-airport group. At b=6 they add a separate singleton, rather than BFI to form the connected thirteen-airport group.',
              'For uniform/airports, the b=6 optimum has component sizes 430,12,1 and score 11204/443; the connected thirteen-airport cut gives 11180/443, lower by 24/443. The latter disrupts only 3,713 passenger-segments, versus 28,044 for the canonical cardinality optimum.',
              '', 'Effect of source prior',
              'With airport-count targets, moving from uniform to inbound-proportional sources favors isolating PGV rather than the canonical equal-count choice AXX at b=1, and PGV rather than AXX alongside the Washington group at b=4 and b=6.',
              'Inbound/airports and uniform/inbound have identical optimal partitions because their objective numerators are identical, although their score units and normalizations differ. When both source and target use inbound traffic, the six-route optimum instead isolates COD, GCC, PGV, PUW and SHR as five singletons.',
              '', 'Cost versus structural importance',
              'All four primary fixed rankings (BC, inverse-passenger BCw, weak, strong) leave the graph connected at every budget 1..6, giving zero protection for all four objectives. BC/passenger-cost isolates IL4, but does not complete further useful cuts through b=6.',
              'The exact protection-per-passenger optimum within six routes isolates IL4 via IL4--ORD (two annual passenger-segments, two performed departures) for the first three objectives. This exposes sensitivity of a ratio objective to extremely low recorded traffic.',
              'For inbound/inbound, the exact efficiency optimum is instead the nine-airport Washington boundary: roughly 55,068 inbound-weight units protected per 183 disrupted passenger-segments, about 300.9 units per disrupted segment. This is a weighted static graph quantity, not a count of distinct people.',
              'Cardinality-optimal cuts are evaluated using observed passenger-segments and performed departures; these costs do not enter the primary objective. The separate efficiency solver has a six-route cap, not an unlimited passenger-budget guarantee.',
              '', 'Candidate paper finding',
              'In the specified 2025 contiguous-US scheduled-service network, an exact three-route cut separates a connected nine-airport Washington group while disrupting 183 of 797,071,268 retained annual passenger-segments. It is optimal for three of four source/target objectives; the four conventional fixed rankings produce no separation through six removals. Source/target weighting and passenger disruption yield different optimal interventions. These results concern static route-network connectivity, not measured epidemic outcomes.']
    return '\n'.join(lines)+'\n'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=root/'data/raw/t100_domestic_segment_2025.zip')
    args = parser.parse_args()
    start = perf_counter()
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if digest!=sourcehash: raise ValueError('Input archive differs from the regression source')
    graph,audit = readgraph(args.input)
    assert (len(graph),graph.number_of_edges())==(443,7160)
    assert audit['months']==list(range(1,13)) and audit['years']=={'2025':458793}
    print('PASS BTS filters: 443 airports, 7160 undirected routes, all 12 months of 2025',flush=True)
    catalog = partitions(graph)
    assert catalog['stats']['atoms']==187 and catalog['stats']['atomsizes']=={1:185,8:1,250:1}
    print('PASS contraction:',json.dumps({k:v for k,v in catalog['stats'].items() if k!='core'},sort_keys=True),flush=True)
    candidate = cohort(graph,washington)
    assert candidate['connected'] and (candidate['internalroutes'],candidate['internalpassengers'],candidate['boundarysize'],candidate['boundarypassengers'])==(30,27447,3,183)
    print('PASS Washington nine-airport group: connected, internal routes=30, internal passengers=27447, boundary routes=3, boundary passengers=183',flush=True)
    records,statistics,orders = analyze(graph,catalog)
    for row in records:
        print(f"{row['prior']}/{row['target']} b={row['budget']}: {row['exactdecimal']:.9f}; cuts={row['optimalcut']}; components={row['componentsizes']}; disrupted passengers={row['disruptedpassengers']}",flush=True)
    with (root/'data/results/airresults.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        for row in records:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
    savedreports.save('air',makereport(graph,audit,catalog,records,statistics,args.input,digest))
    print('Exact DP statistics:',json.dumps(statistics,sort_keys=True))
    print(f'Completed all 24 cardinality optima and secondary efficiency optimization in {perf_counter()-start:.3f} seconds')
    print('Wrote data/results/airresults.csv and data/records.json [air]')

if __name__=='__main__':
    main()
