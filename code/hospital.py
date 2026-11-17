# Hospital optimization and cohort analysis
import records as savedreports
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo
import networkx as nx
from itertools import combinations
from math import comb
import sys

root = Path(__file__).resolve().parent.parent

sourceurl = 'https://sociopatterns.org/assets/data/hospital_lyon_contacts.dat.gz'

sourcepage = 'https://sociopatterns.org/datasets/hospital-ward-dynamic-contact-network/'

sourcehash = '12124ddc438300bcaea0022d4937e0e566679e1fd8f11f16a62214fbebc2b012'

origin = datetime(2010,12,6,13,tzinfo=ZoneInfo('Europe/Paris'))

roleset = {'NUR','MED','ADM','PAT'}

proof = '''EXACTNESS CERTIFICATE
Give every original edge capacity 1, irrespective of contact duration. A Gomory-Hu
tree represents each pairwise minimum edge cut by the minimum tree-edge weight on
its path. Delete tree edges of weight <= B. Each remaining tree component is an
atom: any two of its vertices require more than B original edges to separate.
Consequently every component of G-F, for |F|<=B, is a union of entire atoms.

Given any feasible F, restore every removed edge whose endpoints lie in the same
component of G-F. This preserves the component partition and hence the objective,
and never increases the number of removals. The remaining cuts are exactly all
original edges crossing that partition. No edge internal to an atom is cut. Between
any two atoms either all original edges are cut, or none need be cut: a surviving
edge joins those atoms in one component. Thus group these parallel original edges
into bundles, whose cardinality is their removal cost, and exhaustively enumerate
bundle subsets of total cost <= B, including the empty subset.

Every feasible component partition has a representative in this enumeration. For
each enumerated subset, quotient components give the original components: atoms
remain connected in the surviving original graph by their >B cut connectivity,
even if an atom's induced subgraph is disconnected. Remove redundant internal
bundles, retain the unique boundary cut of each partition, and score that partition.
This also supplies every optimum for b=0,...,B by restricting to boundary cost<=b.
All objective comparisons use exact Fraction arithmetic. Ties prefer fewer cuts,
then lexicographically smallest canonical edge tuples. This does not claim to list
all optimal removal sets with redundant edges.

There are n-c max-flow computations for c original components. Enumeration visits
each bundle subset with total cost<=B, rather than every original edge subset.
This is an exhaustive reduction, not a polynomial-time guarantee for arbitrary
graphs: when atoms are singletons and all bundles have size 1, the reduction may
give no benefit. No approximation, timeout incumbent, or heuristic pruning is used.
Gomory-Hu reference: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.gomory_hu_tree.html
'''

def edges(graph):
    return tuple(sorted((min(u,v),max(u,v)) for u,v in graph.edges))

def readcontacts(path):
    rows,roles,seen = [],{},set()
    with gzip.open(path,'rt') as stream:
        for number,line in enumerate(stream,1):
            fields = line.split()
            if len(fields)!=5: raise ValueError(f'Invalid contact row {number}')
            t,u,v = map(int,fields[:3])
            if t<20 or t%20 or u==v: raise ValueError(f'Invalid contact bin {number}')
            urole,vrole = fields[3:]
            for node,role in ((u,urole),(v,vrole)):
                if role not in roleset or roles.get(node,role)!=role:
                    raise ValueError(f'Invalid or inconsistent role at row {number}')
                roles[node] = role
            key = t,min(u,v),max(u,v)
            if key in seen: raise ValueError(f'Duplicate contact bin {number}')
            seen.add(key)
            rows.append(key)
    if not rows: raise ValueError('Contact file is empty')
    return tuple(sorted(rows)),roles

def makegraph(rows,roles,period='full',threshold=20):
    if period not in ('full','tuesday','wednesday','night'):
        raise ValueError('Unknown observation period')
    if not isinstance(threshold,int) or isinstance(threshold,bool) or threshold<0:
        raise ValueError('Threshold must be a nonnegative integer number of seconds')
    counts,active = Counter(),set()
    for t,u,v in rows:
        time = origin+timedelta(seconds=t)
        if period=='tuesday' and time.date().isoformat()!='2010-12-07': continue
        if period=='wednesday' and time.date().isoformat()!='2010-12-08': continue
        if period=='night' and not (time.hour>=20 or time.hour<7): continue
        counts[u,v] += 1
        active.update((u,v))
    graph = nx.Graph(period=period,threshold=threshold)
    graph.add_nodes_from((v,dict(role=roles[v])) for v in sorted(active))
    graph.add_edges_from((u,v,dict(duration=20*count)) for (u,v),count in sorted(counts.items())
                         if 20*count>=threshold)
    return graph

def selectedgraphs(rows,roles):
    return {name:makegraph(rows,roles,period,threshold) for name,period,threshold in
            (('full60','full',60),('tue80','tuesday',80),('wed80','wednesday',80),('night','night',20))}

def prior(graph,name):
    if name not in ('uniform','HCW','PAT'): raise ValueError('Unknown source prior')
    if any(graph.nodes[v].get('role') not in roleset for v in graph):
        raise ValueError('Every person must have a valid role')
    eligible = {v for v in graph if name=='uniform' or
                (name=='HCW' and graph.nodes[v]['role'] in ('NUR','MED','ADM')) or
                (name=='PAT' and graph.nodes[v]['role']=='PAT')}
    if not eligible: raise ValueError('Source population is empty')
    return {v:Fraction(int(v in eligible),len(eligible)) for v in graph}

def targets(graph,name):
    if name not in ('all','PAT'): raise ValueError('Unknown target objective')
    return {v:int(name=='all' or graph.nodes[v]['role']=='PAT') for v in graph}

def objective(graph,p,a):
    if set(graph)!=set(p) or set(graph)!=set(a): raise ValueError('Objective vertices do not match graph')
    try:
        p,a = {v:Fraction(p[v]) for v in graph},{v:Fraction(a[v]) for v in graph}
    except (TypeError,ValueError,OverflowError,ZeroDivisionError):
        raise ValueError('Probabilities and target weights must be finite numeric values') from None
    if sum(p.values())!=1 or any(v<0 for v in p.values()) or any(v<0 for v in a.values()):
        raise ValueError('Require a probability distribution and nonnegative target weights')
    return p,a

def partitionvalue(groups,p,a):
    return sum(a.values())-sum(sum(p[v] for v in group)*sum(a[v] for v in group) for group in groups)

def components(graph,cuts):
    removed = tuple((min(u,v),max(u,v)) for u,v in cuts)
    if len(set(removed))!=len(removed) or any(not graph.has_edge(*e) for e in removed):
        raise ValueError('Cuts must be distinct existing edges')
    remaining = graph.copy()
    remaining.remove_edges_from(removed)
    return tuple(sorted(tuple(sorted(c)) for c in nx.connected_components(remaining)))

def score(graph,cuts,p,a):
    p,a = objective(graph,p,a)
    return partitionvalue(components(graph,cuts),p,a)

def partitions(graph,b):
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph) or not graph:
        raise ValueError('Require a nonempty simple undirected graph')
    if not isinstance(b,int) or isinstance(b,bool) or b<0:
        raise ValueError('Budget must be a nonnegative integer')
    start = perf_counter()
    original = edges(graph)
    capacity = nx.Graph()
    capacity.add_nodes_from(sorted(graph))
    capacity.add_edges_from((u,v,dict(capacity=1)) for u,v in original)
    forest = nx.Graph()
    forest.add_nodes_from(capacity)
    for group in sorted(nx.connected_components(capacity),key=min):
        if len(group)<2: continue
        tree = nx.gomory_hu_tree(capacity.subgraph(group).copy(),capacity='capacity')
        forest.add_edges_from((u,v) for u,v,d in tree.edges(data=True) if d['weight']>b)
    atoms = tuple(sorted(tuple(sorted(c)) for c in nx.connected_components(forest)))
    owner = {v:i for i,atom in enumerate(atoms) for v in atom}
    links = defaultdict(list)
    for u,v in original:
        if owner[u]!=owner[v]: links[tuple(sorted((owner[u],owner[v])))].append((u,v))
    bundles = tuple((pair,tuple(cuts)) for pair,cuts in sorted(links.items()))
    quotient = nx.Graph()
    quotient.add_nodes_from(range(len(atoms)))
    quotient.add_edges_from(pair for pair,cuts in bundles)
    found,visited = {},0
    def visit(index,spent):
        nonlocal visited
        visited += 1
        groups = tuple(sorted(tuple(sorted(v for i in group for v in atoms[i]))
                              for group in nx.connected_components(quotient)))
        if groups not in found:
            labels = {v:i for i,group in enumerate(groups) for v in group}
            cuts = tuple(e for e in original if labels[e[0]]!=labels[e[1]])
            assert len(cuts)<=spent<=b
            found[groups] = cuts
        for j in range(index,len(bundles)):
            pair,cuts = bundles[j]
            if spent+len(cuts)>b: continue
            quotient.remove_edge(*pair)
            visit(j+1,spent+len(cuts))
            quotient.add_edge(*pair)
    visit(0,0)
    candidates = sorted(((cuts,groups) for groups,cuts in found.items()),key=lambda row:(len(row[0]),row[0]))
    stats = dict(budget=b,atoms=len(atoms),atomsizes=sorted(map(len,atoms)),
                 crossingedges=sum(len(cuts) for pair,cuts in bundles),bundles=len(bundles),
                 bundlesubsets=visited,partitions=len(candidates),seconds=perf_counter()-start)
    return dict(budget=b,nodes=tuple(sorted(graph)),edges=original,states=candidates),stats

def solve(graph,p,a,b,candidates=None):
    if not isinstance(b,int) or isinstance(b,bool) or b<0:
        raise ValueError('Budget must be a nonnegative integer')
    p,a = objective(graph,p,a)
    if candidates is None: candidates,stats = partitions(graph,b)
    if candidates['budget']<b or candidates['nodes']!=tuple(sorted(graph)) or candidates['edges']!=edges(graph):
        raise ValueError('Partition catalog does not cover this graph and budget')
    available = [(partitionvalue(groups,p,a),cuts,groups) for cuts,groups in candidates['states'] if len(cuts)<=b]
    if not available: raise ValueError('No feasible partition supplied')
    value,cuts,groups = min(available,key=lambda row:(-row[0],len(row[1]),row[1]))
    return value,cuts

def rankings(graph):
    ordered = edges(graph)
    for e in ordered:
        value = graph.edges[e].get('duration')
        if isinstance(value,bool) or not isinstance(value,(int,float,Fraction)) or not math.isfinite(value) or value<=0:
            raise ValueError('Every edge needs positive finite duration')
    bc = {tuple(sorted(e)):2*v for e,v in nx.edge_betweenness_centrality(graph,normalized=False).items()}
    weighted = graph.copy()
    for e in ordered: weighted.edges[e]['distance'] = 1/graph.edges[e]['duration']
    bcw = {tuple(sorted(e)):2*v for e,v in nx.edge_betweenness_centrality(weighted,normalized=False,weight='distance').items()}
    return dict(bc=tuple(sorted(ordered,key=lambda e:(-bc[e],e))),
                bcw=tuple(sorted(ordered,key=lambda e:(-bcw[e],e))),
                weak=tuple(sorted(ordered,key=lambda e:(graph.edges[e]['duration'],e))),
                strong=tuple(sorted(ordered,key=lambda e:(-graph.edges[e]['duration'],e))))

def details(graph,cuts):
    return [dict(edge=e,roles=[graph.nodes[v]['role'] for v in e],duration=graph.edges[e]['duration']) for e in cuts]

def checkselected(graphs):
    expected = {'full60':(75,875,2),'tue80':(49,300,1),'wed80':(49,291,1),'night':(53,190,1)}
    for name,graph in graphs.items():
        actual = len(graph),graph.number_of_edges(),nx.edge_connectivity(graph)
        if actual!=expected[name]: raise ValueError(f'{name}: expected {expected[name]}, found {actual}')

def checkvalues(graphs,catalog):
    expected = {'full60':('148/75','119/46','1','29/46'),
                'tue80':('190/49','104/29','43/10','48/29'),
                'wed80':('96/49','76/29','67/20','1'),
                'night':('206/53','51/13','154/27','2')}
    for name,graph in graphs.items():
        for (pname,aname),value in zip((('uniform','all'),('HCW','all'),('PAT','all'),('HCW','PAT')),expected[name]):
            actual,cuts = solve(graph,prior(graph,pname),targets(graph,aname),2,catalog[name])
            if actual!=Fraction(value): raise ValueError(f'{name} {pname}/{aname}: expected {value}, found {actual}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=root/'data/raw/hospital_lyon_contacts.dat.gz')
    parser.add_argument('--budget',type=int,default=4)
    parser.add_argument('--output',type=Path,help='Separate directory for new results and report')
    args = parser.parse_args()
    destination = savedreports.prepare(args.output,('hospitalresults.csv',),('hospital',))
    if args.budget<2: parser.error('The application regression requires maximum budget at least 2')
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if digest!=sourcehash: raise ValueError('Raw hospital file does not match the documented SHA256')
    rows,roles = readcontacts(args.input)
    graphs = selectedgraphs(rows,roles)
    checkselected(graphs)
    catalog,statistics = {},{}
    for name,graph in graphs.items():
        catalog[name],statistics[name] = partitions(graph,args.budget)
        for cuts,groups in catalog[name]['states']:
            assert components(graph,cuts)==groups
        print(name,'n',len(graph),'m',graph.number_of_edges(),'solver',json.dumps(statistics[name]),flush=True)
    checkvalues(graphs,catalog)
    print('PASS all 16 supplied budget-2 values (exact rational equality)',flush=True)
    records = []
    lines = ['SOCIOPATTERNS HOSPITAL: EXACT STATIC APPLICATION',sourcepage,
             'Raw URL: '+sourceurl,'SHA256: '+digest,
             f'Raw bins: {len(rows)}; active people: {len(roles)}; roles: {dict(sorted(Counter(roles.values()).items()))}',
             f'Observed interval endpoints: {origin+timedelta(seconds=rows[0][0])} through {origin+timedelta(seconds=rows[-1][0])}',
             'Each row is one 20-second interval [t-20,t]; times are relative to 2010-12-06 13:00 Europe/Paris.',
             'The full aggregate uses all recorded bins (roughly four days spanning five calendar dates).',
             'Day/night selection uses the recorded interval ENDPOINT. Night: endpoint hour>=20 or <7.',
             'Endpoint selection reproduces night m=190; selecting by interval start instead gives m=189.',
             'Vertices are all people active in the selected period BEFORE duration thresholding; isolates are retained.',
             'Retain an edge iff duration>=threshold; duration=20 times its number of selected bins.',
             'HCW means NUR, MED or ADM. Priors are uniform on the selected source population within each graph.',
             'Target weights are 1 for all people, or 1 only for PAT. Zero target weights are allowed.',
             'STATIC model: choose one fixed removal set of cardinality AT MOST b before infection from the unknown source.',
             'No mandatory spread, sequential intervention, or recomputed centrality is involved in this application.',
             'Score = a(V)-sum_C p(C)a(C). Values and comparisons use exact rational arithmetic.',
             'All six combinations of three source priors and two target objectives are included.',
             'BC uses ordinary unweighted paths; BCw uses length 1/duration. Both use unnormalized ordered-pair values (2 times NetworkX).',
             'Heuristics remove their initial top b canonical edges. Ties use canonical numeric endpoint order.',
             'OPT ties prefer fewer cuts, then lexicographic cuts. Ratios are NA when OPT=0.',
             'Counts, optimal partitions and cuts were replayed on the original person-level graphs.',
             'All 16 supplied budget-2 scores reproduced exactly.',
             'Dataset: Vanhems et al., PLoS ONE 8(9):e73970 (2013); acknowledge SocioPatterns.',
             'License: Creative Commons Attribution-NonCommercial-ShareAlike; see official dataset page.',
             '',proof,'GRAPH AND SOLVER AUDIT']
    for name,graph in graphs.items():
        audit = dict(name=name,n=len(graph),m=graph.number_of_edges(),connectivity=nx.edge_connectivity(graph),
                     rolecounts=dict(sorted(Counter(graph.nodes[v]['role'] for v in graph).items())),**statistics[name])
        lines.append(json.dumps(audit,sort_keys=True))
        orders = rankings(graph)
        for pname in ('uniform','HCW','PAT'):
            p = prior(graph,pname)
            for aname in ('all','PAT'):
                a = targets(graph,aname)
                for b in range(1,args.budget+1):
                    value,cuts = solve(graph,p,a,b,catalog[name])
                    groups = components(graph,cuts)
                    assert score(graph,cuts,p,a)==value
                    record = dict(graph=name,prior=pname,target=aname,budget=b,n=len(graph),m=graph.number_of_edges(),
                                  sources=sum(v>0 for v in p.values()),targettotal=sum(a.values()),
                                  exact=str(value),exactdecimal=float(value),optimalcut=cuts,
                                  usedbudget=len(cuts),componentsizes=sorted(map(len,groups)),components=groups,
                                  removededges=details(graph,cuts),removedduration=sum(graph.edges[e]['duration'] for e in cuts))
                    for method,order in orders.items():
                        chosen = order[:b]
                        result = score(graph,chosen,p,a)
                        assert result<=value
                        record[method+'score'] = str(result)
                        record[method+'decimal'] = float(result)
                        record[method+'ratio'] = str(result/value) if value else ''
                        record[method+'ratiodecimal'] = float(result/value) if value else ''
                        record[method+'cut'] = chosen
                        record[method+'removededges'] = details(graph,chosen)
                        record[method+'removedduration'] = sum(graph.edges[e]['duration'] for e in chosen)
                    records.append(record)
    with (destination/'hospitalresults.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        for row in records:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,tuple,dict)) else v for k,v in row.items()})
    lines += ['', 'RESULTS: graph prior/target b | exact | optimal cut | component sizes | heuristic scores and ratios']
    for row in records:
        heuristic = '; '.join(f"{method}={row[method+'score']} ({row[method+'ratio'] or 'NA'})" for method in ('bc','bcw','weak','strong'))
        lines.append(f"{row['graph']} {row['prior']}/{row['target']} b={row['budget']} | {row['exact']} ({row['exactdecimal']:.12f}) | "
                     f"{row['optimalcut']} | {row['componentsizes']} | {heuristic}")
    lines += ['', 'REMOVED EDGE KEY: endpoints | roles | cumulative contact seconds']
    seen = set()
    for row in records:
        removed = row['removededges']+sum((row[method+'removededges'] for method in ('bc','bcw','weak','strong')),[])
        for item in removed:
            key = row['graph'],tuple(item['edge'])
            if key in seen: continue
            seen.add(key)
            lines.append(f"{row['graph']} {item['edge']} | {item['roles']} | {item['duration']}")
    lines += ['', 'LIMITATIONS',
             'This static aggregate model does not preserve chronological transmission or estimate clinical infection risk.',
             'Duration is tie strength for ranking, not removal cost. Each removed edge consumes exactly one unit of budget.',
             'Thresholding changes the baseline before intervention. This is not a public-health recommendation.',
             'Exactness applies to the stated graph, prior, weights and cardinality budget; no sequential semantics were changed.',
             '', 'REPRODUCE',
             'PYTHONDONTWRITEBYTECODE=1 python code/tests.py hospital',
             'PYTHONDONTWRITEBYTECODE=1 python code/hospital.py']
    savedreports.save('hospital','\n'.join(lines)+'\n',directory=destination)
    print('graph | prior/target | exact b=1,2,3,4 (up to requested budget)',flush=True)
    for name in graphs:
        for pname in ('uniform','HCW','PAT'):
            for aname in ('all','PAT'):
                values = [r['exact'] for r in records if (r['graph'],r['prior'],r['target'])==(name,pname,aname)]
                print(name,pname+'/'+aname,' | '.join(values),flush=True)
    print('Wrote',len(records),'rows to',destination/'hospitalresults.csv','and',destination/'hospital.txt',flush=True)

roleorder = ('PAT','NUR','MED','ADM')

methods = ('bc','bcw','weak','strong')

countproof = '''CO-OPTIMALITY AND COUNTING CONVENTIONS
Budget means at most b, including unused slots. A meaningful cut is the complete
boundary D(P) of a final component partition P. The existing contracted catalog
enumerates every attainable partition through b=4, but not every redundant edge
removal that leaves the same partition. We list every co-optimal partition and its
unique boundary cut, and separately count ALL co-optimal original edge subsets.

For each attainable partition P and exact cardinality k, put q=|D(P)|. There are
binom(m-q,k-q) size-k edge sets containing D(P). Every such set yields exactly one
attainable refinement R of P. Therefore, working from finer to coarser partitions,
N_k(P)=binom(m-q,k-q)-sum_{R strictly refining P} N_k(R), with zero when k<q.
Summing N_k(P) over optimal P and k<=b gives the exact number of optimal edge sets,
including redundant removals. Counts over ALL partitions sum to binom(m,k), which
is checked for every graph and k. Integer arithmetic is used throughout.

Each reported edge-set family is a lossless compact enumeration: take its boundary
D(P), add any subset R of E(G)\\D(P) of size <= b-|D(P)|, and exclude any R containing
a listed forbidden refinement set. A forbidden set is D(Q)\\D(P) for a feasible
strict refinement Q; supersets of another forbidden set are omitted. These families
are disjoint because their final partitions differ. We explicitly list all edge
sets only when there are <=100 and at most 10000 supersets need checking. Otherwise
the exact count and compact family replace a potentially enormous literal list.

Structurally unique means one labelled component partition, ignoring redundant
removals. Role-profile uniqueness is reported separately and ignores person IDs.
The giant is the unique largest component in these data. A full partition is its
listed non-giant components plus their complement in the graph's vertex roster.
The CSV also lists all giant IDs explicitly. A component is safe iff the source is
outside it: safe probability=1-p(C), expected target contribution=a(C)(1-p(C)).
No component is unconditionally 'safe' unless its source probability is zero.
The claim about multi-patient groups refers to NON-GIANT components; a giant with
many patients can also be protected when the source falls in an isolated HCW.
All ranks are 1-based positions in the existing fixed canonical ranking; equal
scores/durations use numeric lexicographic edge order. Duration ranks are given
both weakest-first and strongest-first. Bridge status is on the uncut baseline.
'''

def refines(fine,coarse):
    labels = {v:i for i,c in enumerate(coarse) for v in c}
    return all(len({labels[v] for v in c})==1 for c in fine)

def countsets(graph,catalog,b):
    if catalog['budget']<b: raise ValueError('Catalog does not cover the budget')
    counts = {}
    m = graph.number_of_edges()
    states = sorted(catalog['states'],key=lambda row:(-len(row[1]),row[1]))
    for cut,groups in states:
        refinements = [other for other in counts if refines(other,groups)]
        q = len(cut)
        counts[groups] = tuple(comb(m-q,k-q)-sum(counts[other][k] for other in refinements)
                               if k>=q else 0 for k in range(b+1))
        assert all(n>=0 for n in counts[groups])
    for k in range(b+1): assert sum(n[k] for n in counts.values())==comb(m,k)
    return counts

def family(graph,cut,groups,catalog,b,count):
    boundary = set(cut)
    possible = {tuple(sorted(set(other)-boundary)) for other,parts in catalog['states']
                if parts!=groups and len(other)<=b and refines(parts,groups)}
    forbidden = []
    for extra in sorted(possible,key=lambda e:(len(e),e)):
        if not any(set(small)<=set(extra) for small in forbidden): forbidden.append(extra)
    internal = tuple(e for e in edges(graph) if e not in boundary)
    spare = b-len(cut)
    explicit = None
    if count<=100 and sum(comb(len(internal),k) for k in range(spare+1))<=10000:
        explicit = []
        for size in range(spare+1):
            for extra in combinations(internal,size):
                if any(set(bad)<=set(extra) for bad in forbidden): continue
                removed = tuple(sorted(cut+extra))
                assert components(graph,removed)==groups
                explicit.append(removed)
        assert len(explicit)==count
    return dict(boundary=cut,extrasfrom='E(graph) minus boundary',maximumextras=spare,
                forbiddenextras=forbidden,edgesetcount=count,explicitsets=explicit)

def describe(graph,groups,p,a):
    largest = max(map(len,groups))
    assert sum(len(c)==largest for c in groups)==1
    result = []
    for group in groups:
        counts = Counter(graph.nodes[v]['role'] for v in group)
        mass = sum(p[v] for v in group)
        weight = sum(a[v] for v in group)
        result.append(dict(ids=group,size=len(group),roles={r:counts[r] for r in roleorder},
                           giant=len(group)==largest,singleton=len(group)==1,
                           patientcategory='multiple' if counts['PAT']>=2 else 'one' if counts['PAT'] else 'none',
                           sourceprobability=str(mass),safeprobability=str(1-mass),
                           targetweight=str(weight),expectedcontribution=str(weight*(1-mass))))
    return result

def profile(description):
    return tuple(sorted((c['giant'],c['size'],tuple(c['roles'][r] for r in roleorder)) for c in description))

def small(description):
    return [c for c in description if not c['giant']]

def compact(description):
    parts = small(description)
    if not parts: return 'no separated non-giant component'
    return '; '.join(f"{c['ids']} size={c['size']} roles(PAT,NUR,MED,ADM)="
                     f"{tuple(c['roles'][r] for r in roleorder)} singleton={c['singleton']} patients={c['patientcategory']}"
                     for c in parts)

def transition(previous,current):
    old = {tuple(map(tuple,p['groups'])) for p in previous['cooptimal']} if previous else set()
    new = {tuple(map(tuple,p['groups'])) for p in current['cooptimal']}
    value = Fraction(current['exact'])
    earlier = Fraction(previous['exact']) if previous else Fraction(0)
    retained,added = len(old&new),len(new-old)
    description = current['canonicalcomponents']
    before = small(previous['canonicalcomponents']) if previous else []
    after = small(description)
    oldids = {tuple(c['ids']) for c in before}
    newids = {tuple(c['ids']) for c in after}
    if value==earlier:
        reason = 'same optimum; previous representative remains feasible and optimal' if previous else 'zero optimum; no useful removal'
    elif any(c['size']>1 and tuple(c['ids']) not in oldids for c in after):
        reason = 'qualitatively new multi-person non-giant component becomes feasible'
    elif oldids<=newids:
        reason = f'{len(newids-oldids)} additional singleton component(s) in the canonical solution'
    else:
        reason = 'switch of singleton identities/roles; the optimal partition need not refine the previous optimum'
    reason += f"; canonical boundary uses {current['usedbudget']}/{current['budget']} cuts"
    if previous: reason += f'; {retained} old optimal partitions retained, {added} new co-optimal partitions'
    return dict(increase=str(value-earlier),reason=reason,retained=retained,new=added,
                canonicalunchanged=bool(previous and previous['canonicalpartition']==current['canonicalpartition']))

def edgetable(graph):
    orders = rankings(graph)
    positions = {method:{e:i+1 for i,e in enumerate(order)} for method,order in orders.items()}
    bridges = {tuple(sorted(e)) for e in nx.bridges(graph)}
    return {e:dict(edge=e,roles=[graph.nodes[v]['role'] for v in e],duration=graph.edges[e]['duration'],
                   bcrank=positions['bc'][e],bcwrank=positions['bcw'][e],
                   weakrank=positions['weak'][e],strongrank=positions['strong'][e],bridge=e in bridges)
            for e in edges(graph)}

def diagnosis(graph,row,options,ranks,p,a,uniformopt):
    result = {}
    exact = Fraction(row['exact'])
    for method in methods:
        cut = tuple(sorted(tuple(e) for e in json.loads(row[method+'cut'])))
        groups = components(graph,cut)
        value = partitionvalue(groups,p,a)
        assert value==Fraction(row[method+'score'])
        labels = {v:i for i,c in enumerate(groups) for v in c}
        boundary = {e for e in edges(graph) if labels[e[0]]!=labels[e[1]]}
        useful = len(boundary)
        witness = min(options,key=lambda opt:(len(set(opt['boundarycut'])-set(cut)),len(opt['boundarycut']),opt['boundarycut']))
        missing = sorted(set(witness['boundarycut'])-set(cut))
        reasons = []
        if value==exact:
            reasons.append('exactly optimal' if exact else 'trivial zero-OPT tie')
        else:
            reasons.append('wrong individual edge selection for this objective')
            if len(groups)==1: reasons.append('all selected cuts leave the graph connected')
            if useful<len(cut): reasons.append(f'{len(cut)-useful} selected edges are redundant for the achieved partition')
            coordinated = []
            for c in small(witness['components']):
                vertices = set(c['ids'])
                border = [e for e in edges(graph) if (e[0] in vertices)!=(e[1] in vertices)]
                if len(border)>1 and not set(border)<=set(cut): coordinated.append(c['ids'])
            if coordinated: reasons.append(f'incomplete coordinated boundary cut around {coordinated}')
            up = {v:Fraction(1,len(graph)) for v in graph}
            ua = {v:1 for v in graph}
            if partitionvalue(groups,up,ua)==uniformopt:
                reasons.append('this partition is optimal for uniform/all; source prior and/or patient target changes which side is valuable')
            elif not coordinated:
                reasons.append('ranking misses individually available separations or chooses less valuable roles')
        result[method] = dict(score=str(value),gap=str(exact-value),ratio=str(value/exact) if exact else None,
                              cut=cut,boundaryedges=useful,redundantedges=len(cut)-useful,
                              components=describe(graph,groups,p,a),reason='; '.join(reasons),
                              witness=witness['id'],missing=missing,
                              missingranks=[ranks[e][method+'rank'] for e in missing])
    return result

def analyze():
    raw = root/'data/raw/hospital_lyon_contacts.dat.gz'
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==sourcehash
    rows,roles = readcontacts(raw)
    graphs = selectedgraphs(rows,roles)
    checkselected(graphs)
    source = root/'data/results/hospitalresults.csv'
    saved = list(csv.DictReader(source.open()))
    assert len(saved)==96 and len({(r['graph'],r['prior'],r['target'],r['budget']) for r in saved})==96
    bykey = {(r['graph'],r['prior'],r['target'],int(r['budget'])):r for r in saved}
    records,catalogs,ranktables = [],{},{}
    for name,graph in graphs.items():
        catalog,stats = partitions(graph,4)
        counts = countsets(graph,catalog,4)
        ids = {groups:f'{name}:P{i+1:03d}' for i,(cut,groups) in enumerate(catalog['states'])}
        for cut,groups in catalog['states']: assert components(graph,cut)==groups
        catalogs[name] = dict(catalog=catalog,stats=stats,counts=counts,ids=ids)
        ranktables[name] = edgetable(graph)
        for pname in ('uniform','HCW','PAT'):
            p = prior(graph,pname)
            for aname in ('all','PAT'):
                a = targets(graph,aname)
                previous = None
                for b in range(1,5):
                    row = bykey[name,pname,aname,b]
                    value = Fraction(row['exact'])
                    options = []
                    feasible = [(cut,groups) for cut,groups in catalog['states'] if len(cut)<=b]
                    assert max(partitionvalue(groups,p,a) for cut,groups in feasible)==value
                    for cut,groups in feasible:
                        if partitionvalue(groups,p,a)!=value: continue
                        description = describe(graph,groups,p,a)
                        number = sum(counts[groups][:b+1])
                        options.append(dict(id=ids[groups],boundarycut=cut,boundarysize=len(cut),groups=groups,
                                            components=description,edgesetcount=number,
                                            edgesetsbycardinality=counts[groups][:b+1],
                                            family=family(graph,cut,groups,catalog,b,number)))
                    canonical = tuple(tuple(e) for e in json.loads(row['optimalcut']))
                    chosen = next(opt for opt in options if opt['boundarycut']==canonical)
                    assert chosen['groups']==tuple(tuple(c) for c in json.loads(row['components']))
                    profilecount = len({profile(opt['components']) for opt in options})
                    result = dict(graph=name,prior=pname,target=aname,budget=b,exact=str(value),exactdecimal=float(value),
                                  canonicalpartition=chosen['id'],canonicalcut=canonical,usedbudget=len(canonical),
                                  canonicalcomponents=chosen['components'],optimalpartitions=len(options),
                                  optimalboundarycuts=len(options),optimaledgesets=sum(opt['edgesetcount'] for opt in options),
                                  edgesetsbycardinality=[sum(opt['edgesetsbycardinality'][k] for opt in options) for k in range(b+1)],
                                  structurallyunique=len(options)==1,roleprofiles=profilecount,
                                  multipersonpartitions=sum(any(c['size']>1 for c in small(opt['components'])) for opt in options),
                                  multipatientpartitions=sum(any(c['roles']['PAT']>=2 for c in small(opt['components'])) for opt in options),
                                  cooptimal=options)
                    result['transition'] = transition(previous,result)
                    uniformopt = Fraction(bykey[name,'uniform','all',b]['exact'])
                    result['heuristics'] = diagnosis(graph,row,options,ranktables[name],p,a,uniformopt)
                    result['optimaledgedetails'] = [ranktables[name][e] for e in sorted({e for opt in options for e in opt['boundarycut']})]
                    records.append(result)
                    previous = result
    return records,graphs,catalogs,ranktables,hashlib.sha256(source.read_bytes()).hexdigest()

def makereport(records,graphs,catalogs,ranks,digest):
    lines = ['HOSPITAL EXACT SOLUTIONS: STRUCTURAL INTERPRETATION',
             'Input: data/results/hospitalresults.csv; SHA256 '+digest,
             'Scope: the existing four graphs, six prior/target combinations and budgets 1-4 only.',
             'All 96 saved exact scores and canonical partitions reproduced; no new graph experiment.',
             'Original endpoint-based day/night selection and duration thresholds are unchanged.',
             '',countproof,'VERTEX ROSTERS AND SMALL SEARCH SPACES']
    for name,graph in graphs.items():
        lines.append(f"{name}: n={len(graph)}, m={graph.number_of_edges()}, catalog partitions={catalogs[name]['stats']['partitions']}")
        for role in roleorder: lines.append(f"  {role}: {sorted(v for v in graph if graph.nodes[v]['role']==role)}")
    lines += ['', 'BUDGET-BY-BUDGET INTERPRETATION']
    for row in records:
        lines += ['',f"{row['graph']} {row['prior']}/{row['target']} b={row['budget']} OPT={row['exact']} ({row['exactdecimal']:.9f})",
                  f"  Optimal partitions/boundary cuts={row['optimalpartitions']}; all original edge sets={row['optimaledgesets']}; counts by exact size 0..b={row['edgesetsbycardinality']}",
                  f"  Structurally unique={row['structurallyunique']}; distinct size/role profiles={row['roleprofiles']}; canonical={row['canonicalpartition']}",
                  '  Budget transition: '+row['transition']['reason']+'; score increase='+row['transition']['increase']]
        for opt in row['cooptimal']:
            giant = next(c for c in opt['components'] if c['giant'])
            lines.append(f"  {opt['id']}: boundary={opt['boundarycut']}; edge sets={opt['edgesetcount']}; non-giants: {compact(opt['components'])}")
            lines.append(f"    Giant=complement, size={giant['size']}, roles={giant['roles']}; safe probability={giant['safeprobability']}, expected target contribution={giant['expectedcontribution']}")
            for c in small(opt['components']):
                lines.append(f"    Component {c['ids']}: safe probability={c['safeprobability']}; expected target contribution={c['expectedcontribution']}")
            fam = opt['family']
            if fam['explicitsets'] is not None:
                lines.append('    All literal co-optimal edge sets for this partition: '+str(fam['explicitsets']))
            else:
                lines.append(f"    Compact complete family: boundary plus <= {fam['maximumextras']} other original edges, excluding supersets of {fam['forbiddenextras']}; literal expansion omitted")
        for method,result in row['heuristics'].items():
            lines.append(f"  {method}: score={result['score']}, gap={result['gap']}, ratio={result['ratio'] or 'NA'}; {result['reason']}")
            if Fraction(result['gap']):
                lines.append(f"    Achieved non-giants: {compact(result['components'])}; witness {result['witness']} missing {result['missing']} at {method} ranks {result['missingranks']}")
    lines += ['', 'EVERY OPTIMAL MULTI-PERSON NON-GIANT CASE']
    for row in records:
        for opt in row['cooptimal']:
            groups = [c for c in small(opt['components']) if c['size']>1]
            if groups: lines.append(f"{row['graph']} {row['prior']}/{row['target']} b={row['budget']}: {groups}; cut={opt['boundarycut']}")
    lines += ['No optimal non-giant component contains two or more patients.',
              'Multiple isolated singleton patients are distinct components, not a single multi-patient safe group.',
              '', 'KEY BUDGET MECHANISMS',
              'full60: b=2 isolates administrator1525 via its two incident edges; b=3 cannot improve this positive optimum.',
              'At b=4, uniform/all, HCW/all and HCW/PAT uniquely separate the connected ADM pair {1525,1535}.',
              'The internal edge1525--1535 is retained. Their four-edge joint boundary costs less than isolating both separately.',
              'Uniform/PAT, PAT/all and PAT/PAT instead uniquely isolate patient1332 with four cuts. Thus the jump is not always group protection.',
              'For HCW/PAT, the ADM pair contains no targets: its source mass2/46 protects all29 patients in the giant when the source is in that pair, giving29/23.',
              'wed80: b=3 can combine the one-edge patient1373 isolation with a two-edge isolation of MED1168 or ADM1209.',
              'These are two separate singletons. PAT/PAT remains flat through b=3 because the additional HCW singleton carries neither source mass nor target weight.',
              'At b=4, uniform/all still saves190/49 in expectation but has six tied partitions; role-sensitive objectives select different singleton combinations.',
              'tue80: b=2 isolates PAT1373 and NUR1238 using two bridges. The uniform/all value plateaus through b=4.',
              'At b=4 there are seven uniform/all optimal partitions, not just the old representative. Other objectives improve by swapping singleton identities.',
              'HCW/all selects ADM1209 plus NUR1238; patient-oriented optima select PAT1373 plus PAT1323 or PAT1332.',
              'night: six one-bridge leaves are ADM1232 and PAT1320,1332,1377,1547,1701.',
              'Uniform/all selects any b of these six, so OPT=b*(105-b)/53 and co-optimal partition counts are6,15,20,15 for b=1..4.',
              'Patient-oriented solutions select patient leaves; HCW/PAT instead selects ADM1232 at b=1 (27/26), has15 tied pairs at b=2, then patient-only singleton sets at b=3,4.',
              'HCW/all is an exception to simple leaf accumulation: at b=3 it replaces its patient singleton with NUR1164, whose isolation requires two edges, keeping ADM1232.',
              '', 'LARGEST HEURISTIC GAPS BY GRAPH / PRIOR / TARGET',
              'For each method, maximize absolute expected-target deficit over b=1..4. Separately report the minimum ratio over positive-OPT budgets.']
    for name in graphs:
        for pname in ('uniform','HCW','PAT'):
            for aname in ('all','PAT'):
                selected = [r for r in records if (r['graph'],r['prior'],r['target'])==(name,pname,aname)]
                lines.append(f'{name} {pname}/{aname}')
                for method in methods:
                    gap = max(Fraction(r['heuristics'][method]['gap']) for r in selected)
                    worst = [r for r in selected if Fraction(r['heuristics'][method]['gap'])==gap]
                    positive = [r for r in selected if Fraction(r['exact'])>0]
                    ratio = min(Fraction(r['heuristics'][method]['ratio']) for r in positive) if positive else None
                    lines.append(f"  {method}: max gap={gap} ({float(gap):.9f}) at budgets {[r['budget'] for r in worst]}; minimum ratio={ratio}")
                    if gap:
                        for row in worst:
                            data = row['heuristics'][method]
                            lines.append(f"    b={row['budget']}: {data['reason']}; witness={data['witness']} missing={data['missing']} ranks={data['missingranks']}")
    lines += ['', 'RANKS OF EVERY EDGE IN ANY CO-OPTIMAL BOUNDARY CUT',
              'graph | edge | endpoint roles | seconds | BC rank | BCw rank | weak rank | strong rank | original bridge']
    for name in graphs:
        selected = sorted({e for r in records if r['graph']==name for opt in r['cooptimal'] for e in opt['boundarycut']})
        for e in selected:
            d = ranks[name][e]
            lines.append(f"{name} | {e} | {d['roles']} | {d['duration']} | {d['bcrank']} | {d['bcwrank']} | {d['weakrank']} | {d['strongrank']} | {d['bridge']}")
    lines += ['', 'REPRODUCE',
              'PYTHONDONTWRITEBYTECODE=1 python code/hospital.py --interpret',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py hospitalinterpret',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py hospital',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py',
              '', 'Candidate paper findings',
              '- In these four selected hospital graphs and budgets1-4, exactly three graph/prior/target/budget cases have an optimal non-giant group larger than one: full60 at b=4 with uniform/all, HCW/all or HCW/PAT; the group is two administrators.',
              '- No co-optimal non-giant component contains two patients. Patient separations in these optima are individual singleton components, not multi-patient groups.',
              '- More budget can change the set of optimal partitions without changing the optimum value: Tuesday uniform/all has one optimal partition at b=2,3 and seven at b=4.',
              '- Source priors and target populations change optimal identities. Isolating HCWs can protect patients in the large remaining component when the source lies among those isolated HCWs.',
              '- The night uniform/all optimum over b=1..4 is explained by independent leaf separations, with exact score b*(105-b)/53; this is nearly linear but has decreasing increments.',
              '- Fixed edge rankings can fail to complete a coordinated small boundary cut. In full60 at b=4 all four heuristics leave the graph connected despite a positive exact optimum for every tested objective.',
              '- These are properties of the specified static thresholded aggregates, not estimates of clinical intervention effectiveness.']
    return '\n'.join(lines)+'\n'

def interpret():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,help='Separate directory for new results and report')
    args = parser.parse_args()
    destination = savedreports.prepare(args.output,('hospitalinterpret.csv',),('hospitalinterpret',))
    records,graphs,catalogs,ranks,digest = analyze()
    csvpath = destination/'hospitalinterpret.csv'
    with csvpath.open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in record.items()})
    text = makereport(records,graphs,catalogs,ranks,digest)
    savedreports.save('hospitalinterpret',text,directory=destination)
    print('Verified all 96 saved cases; enumerated every optimal partition and boundary cut')
    print('Exact original-edge-set counts include redundant cuts; large families are represented compactly')
    print('Optimal multi-person non-giant cases:',sum(r['multipersonpartitions']>0 for r in records))
    print('Optimal multi-patient non-giant cases:',sum(r['multipatientpartitions']>0 for r in records))
    for name in graphs:
        rows = [r for r in records if r['graph']==name and r['prior']=='uniform' and r['target']=='all']
        print(name,'uniform/all: (b, optimal partitions, all optimal edge sets)',
              [(r['budget'],r['optimalpartitions'],r['optimaledgesets']) for r in rows])
    print('Wrote',csvpath,'and',destination/'hospitalinterpret.txt')

resolutions = (.75,1.,1.25,1.5)

seeds = tuple(range(20))

objectives = (('uniform','all'),('HCW','all'),('PAT','all'),('HCW','PAT'))

def graphsfrom(rows,roles):
    return {name:makegraph(rows,roles,period,20) for name,period in
            (('full','full'),('tuesday','tuesday'),('wednesday','wednesday'),('night','night'))}

def cohortmetrics(graph,cohort):
    cohort = frozenset(cohort)
    if not cohort or not cohort<=set(graph): raise ValueError('Require a nonempty set of graph vertices')
    internal = graph.subgraph(cohort)
    boundary = tuple(e for e in edges(graph) if (e[0] in cohort)!=(e[1] in cohort))
    inside = sum(d['duration'] for u,v,d in internal.edges(data=True))
    outside = sum(graph.edges[e]['duration'] for e in boundary)
    volume = 2*inside+outside
    denominator = min(volume,2*sum(d['duration'] for u,v,d in graph.edges(data=True))-volume)
    conductance = Fraction(outside,denominator) if denominator else None
    counts = Counter(graph.nodes[v]['role'] for v in cohort)
    connected = nx.is_connected(internal)
    candidate = len(cohort)<len(graph) and len(cohort)>=2 and counts['PAT']>=2 and connected and len(boundary)*10<=graph.number_of_edges()
    return dict(ids=tuple(sorted(cohort)),size=len(cohort),roles={r:counts[r] for r in roleorder},
                patients=tuple(sorted(v for v in cohort if graph.nodes[v]['role']=='PAT')),
                internaledges=internal.number_of_edges(),internalseconds=inside,boundary=boundary,
                boundaryedges=len(boundary),boundaryseconds=outside,
                boundaryfraction=str(Fraction(len(boundary),graph.number_of_edges())) if graph.number_of_edges() else None,
                conductance=str(conductance) if conductance is not None else None,
                conductancedecimal=float(conductance) if conductance is not None else None,
                connected=connected,candidate=candidate)

def discover(graph):
    runs,origins = [],defaultdict(lambda:defaultdict(set))
    for resolution in resolutions:
        for seed in seeds:
            communities = tuple(sorted(tuple(sorted(c)) for c in nx.community.louvain_communities(
                graph,weight='duration',resolution=resolution,seed=seed,backend='networkx')))
            assert set().union(*map(set,communities))==set(graph)
            assert sum(map(len,communities))==len(graph)
            runs.append(dict(resolution=resolution,seed=seed,communities=communities,
                             modularity=nx.community.modularity(graph,communities,weight='duration',resolution=resolution)))
            for community in communities:
                origins[community]['louvain'].add((resolution,seed))
                induced = graph.subgraph(community)
                if not nx.is_connected(induced):
                    for group in nx.connected_components(induced):
                        origins[tuple(sorted(group))]['connected-community-part'].add((resolution,seed))
                patients = {v for v in community if graph.nodes[v]['role']=='PAT'}
                for group in nx.connected_components(graph.subgraph(patients)):
                    if len(group)>=2:
                        origins[tuple(sorted(group))]['patient-component'].add((resolution,seed))
                for u,v in graph.subgraph(patients).edges:
                    origins[tuple(sorted((u,v)))]['patient-pair'].add((resolution,seed))
    return runs,origins

def stability(graph,cohort,runs):
    cohort = set(cohort)
    patients = {v for v in cohort if graph.nodes[v]['role']=='PAT'}
    exact,larger,samepatients,together = [],[],[],[]
    variants = Counter()
    for run in runs:
        key = run['resolution'],run['seed']
        for group in run['communities']:
            members = set(group)
            if cohort==members: exact.append(key)
            elif cohort<members: larger.append(key)
            if patients and patients<=members: together.append(key)
            if patients and {v for v in group if graph.nodes[v]['role']=='PAT'}==patients:
                samepatients.append(key)
                variants[tuple(sorted(members-patients))] += 1
    byresolution = []
    for resolution in resolutions:
        selected = sum(r['resolution']==resolution for r in runs)
        if not selected: continue
        byresolution.append(dict(resolution=resolution,runs=selected,
                                 exact=sum(r==resolution for r,s in exact),
                                 largersubset=sum(r==resolution for r,s in larger),
                                 exactpatients=sum(r==resolution for r,s in samepatients) if patients else None,
                                 patientstogether=sum(r==resolution for r,s in together) if patients else None))
    return dict(runs=len(runs),exact=len(exact),largersubset=len(larger),
                exactpatients=len(samepatients) if patients else None,
                patientstogether=len(together) if patients else None,
                exactappearances=exact,subsetappearances=larger,
                patientappearances=samepatients,patienttogetherappearances=together,
                staffvariants=[dict(ids=staff,runs=count) for staff,count in sorted(variants.items())],
                byresolution=byresolution)

def rawpair(rows,roles,night):
    cohort = {1460,1784}
    assert all(roles[v]=='PAT' for v in cohort)
    internal,boundary = defaultdict(list),defaultdict(list)
    for t,u,v in rows:
        clock = (13*3600+t)%86400
        if not (clock>=20*3600 or clock<7*3600): continue
        if u in cohort and v in cohort: internal[u,v].append(t)
        elif (u in cohort)!=(v in cohort): boundary[u,v].append(t)
    assert set(internal)=={(1460,1784)} and len(internal[1460,1784])==9
    expected = {(1202,1784):20,(1460,1485):140,(1460,1658):20,(1485,1784):20}
    assert {e:20*len(times) for e,times in boundary.items()}==expected
    data = cohortmetrics(night,cohort)
    assert data['internalseconds']==180 and data['boundaryedges']==4 and data['boundaryseconds']==200
    for e,value in expected.items(): assert night.edges[e]['duration']==value
    return dict(ids=sorted(cohort),roles=['PAT','PAT'],internalseconds=180,boundaryedges=4,boundaryseconds=200,
                internalbins=[dict(edge=e,count=len(times),timestamps=times) for e,times in sorted(internal.items())],
                boundarybins=[dict(edge=e,count=len(times),timestamps=times) for e,times in sorted(boundary.items())])

def estimate(graph,b):
    capacity = nx.Graph()
    capacity.add_nodes_from(sorted(graph))
    capacity.add_edges_from((u,v,dict(capacity=1)) for u,v in edges(graph))
    tree = nx.gomory_hu_tree(capacity,capacity='capacity')
    tree.remove_edges_from([(u,v) for u,v,d in tree.edges(data=True) if d['weight']<=b])
    owner = {v:i for i,c in enumerate(nx.connected_components(tree)) for v in c}
    bundles = Counter(tuple(sorted((owner[u],owner[v]))) for u,v in graph.edges if owner[u]!=owner[v])
    counts = [1]+[0]*b
    for cost in bundles.values():
        for j in range(b,cost-1,-1): counts[j] += counts[j-cost]
    return sum(counts)

def benchmarks(name,graph,budgets,saved,oldgraphs):
    if not budgets: return {},dict(status='not needed: no candidate boundary has size <=6')
    b = max(budgets)
    if b>6: raise ValueError('Exact comparison is restricted to budget <=6')
    if name=='full' and b>4:
        return {},dict(status='not attempted: budget 5/6 expansion restricted to smaller graphs',budget=b)
    estimated = estimate(graph,b)
    if estimated>100000:
        return {},dict(status='not attempted: exact bundle-subset estimate exceeds 100000',budget=b,subsets=estimated)
    catalog,stats = partitions(graph,b)
    assert stats['bundlesubsets']==estimated
    oldname = dict(full='full60',tuesday='tue80',wednesday='wed80',night='night')[name]
    old = oldgraphs[oldname]
    samegraph = set(old)==set(graph) and edges(old)==edges(graph) and all(old.edges[e]['duration']==graph.edges[e]['duration'] for e in edges(graph))
    result = {}
    for budget in sorted(budgets):
        for pname,aname in objectives:
            p,a = prior(graph,pname),targets(graph,aname)
            value,cut = solve(graph,p,a,budget,catalog)
            origin = 'exact partition enumeration on this unthresholded baseline'
            if samegraph and budget<=4:
                previous = next(r for r in saved if (r['graph'],r['prior'],r['target'],int(r['budget']))==(oldname,pname,aname,budget))
                assert value==Fraction(previous['exact'])
                origin = 'matches existing hospitalresults.csv and rechecked exact catalog on identical graph'
            result[budget,pname,aname] = dict(value=str(value),decimal=float(value),cut=cut,
                                            components=components(graph,cut),origin=origin)
    return result,dict(status='exact',samepreviousgraph=samegraph,requestedbudgets=sorted(budgets),**stats)

def evaluate(graph,record,exact):
    cut = record['boundary']
    groups = components(graph,cut)
    assert tuple(record['ids']) in groups
    remaining = graph.copy()
    remaining.remove_edges_from(cut)
    reachable = {s:nx.node_connected_component(remaining,s) for s in graph}
    result = dict(componentcount=len(groups),exactlytwo=len(groups)==2,components=groups,
                  componentsizes=sorted(map(len,groups)),objectives={})
    for pname,aname in objectives:
        p,a = prior(graph,pname),targets(graph,aname)
        value = score(graph,cut,p,a)
        direct = sum(p[s]*sum(a[v] for v in graph if v not in reachable[s]) for s in graph)
        assert value==direct
        cohortmass = sum(p[v] for v in record['ids'])
        targetmass = sum(a[v] for v in record['ids'])
        item = dict(value=str(value),decimal=float(value),cohortsafeprobability=str(1-cohortmass),
                    cohortcontribution=str(targetmass*(1-cohortmass)))
        optimum = exact.get((len(cut),pname,aname))
        if optimum is None:
            item.update(exact=None,ratio=None,classification='not compared: budget >6' if len(cut)>6 else 'exact comparison unavailable')
        else:
            best = Fraction(optimum['value'])
            assert value<=best
            ratio = value/best if best else None
            category = 'globally optimal' if value==best else 'near-optimal' if ratio>=Fraction(9,10) else 'substantially worse'
            item.update(exact=optimum,ratio=str(ratio) if ratio is not None else None,
                        ratiodecimal=float(ratio) if ratio is not None else None,gap=str(best-value),classification=category)
        result['objectives'][pname+'/'+aname] = item
    return result

def analyzecohorts():
    tick = perf_counter()
    path = root/'data/raw/hospital_lyon_contacts.dat.gz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sourcehash
    rows,roles = readcontacts(path)
    graphs = graphsfrom(rows,roles)
    oldgraphs = selectedgraphs(rows,roles)
    saved = list(csv.DictReader((root/'data/results/hospitalresults.csv').open()))
    audit = rawpair(rows,roles,graphs['night'])
    records,runaudit,searches = [],{},{}
    for name,graph in graphs.items():
        before = nx.node_link_data(graph)
        runs,origins = discover(graph)
        if name=='night': origins[(1460,1784)]['explicit'].add(('requested',None))
        local = []
        ranks = edgetable(graph)
        for ids,methods in sorted(origins.items()):
            data = dict(graph=name,n=len(graph),m=graph.number_of_edges(),threshold=20,
                        sources=sorted(methods),**cohortmetrics(graph,ids))
            data['stability'] = stability(graph,ids,runs)
            data['generation'] = {method:sorted(values) for method,values in sorted(methods.items())}
            local.append(data)
        budgets = {r['boundaryedges'] for r in local if r['candidate'] and r['boundaryedges']<=6}
        exact,search = benchmarks(name,graph,budgets,saved,oldgraphs)
        searches[name] = search
        for record in local:
            record['evaluation'] = evaluate(graph,record,exact) if record['candidate'] else None
            record['boundarydetails'] = [ranks[e] for e in record['boundary']] if record['candidate'] else []
        candidates = [r for r in local if r['candidate']]
        shortlist = min(candidates,key=lambda r:(r['boundaryedges'],-r['stability']['patientstogether'],
                                                -r['stability']['exactpatients'],-r['stability']['exact'],r['boundaryseconds'],r['ids'])) if candidates else None
        for record in local: record['shortlisted'] = record is shortlist or (record['candidate'] and record['boundaryedges']<=6)
        records.extend(local)
        runaudit[name] = runs
        assert nx.node_link_data(graph)==before
        print(name,'n',len(graph),'m',graph.number_of_edges(),'Louvain runs',len(runs),
              'unique communities',sum('louvain' in r['sources'] for r in local),'candidates',len(candidates),
              'candidate budgets <=6',sorted(budgets),flush=True)
    return records,graphs,runaudit,searches,audit,perf_counter()-tick

def cohortreport(records,graphs,runs,searches,audit,seconds):
    lines = ['HOSPITAL COHORT BOUNDARIES: WEIGHTED COMMUNITY ANALYSIS',
             'Raw input: data/raw/hospital_lyon_contacts.dat.gz; SHA256 '+sourcehash,
             'This is a separate unthresholded analysis: retain every observed edge (minimum possible duration20s).',
             'It does NOT use full60/tue80/wed80. The night graph coincides with the earlier night baseline.',
             'Full uses all bins; day/night bin selection uses the recorded interval endpoint, unchanged from hospital.py.',
             'Night means20:00<=clock or clock<07:00. Times are relative to2010-12-06 13:00 Europe/Paris.',
             f'NetworkX {nx.__version__}; Python NetworkX backend; sorted numeric IDs and edges; weighted Louvain uses duration.',
             'Louvain resolutions0.75,1.0,1.25,1.5; seeds0..19; 80 runs per graph,320 total.',
             'Louvain is a modularity heuristic. Seeds/resolutions measure algorithmic sensitivity, not independent data replications or clinical validity.',
             'Reference: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html',
             '', 'COHORT GENERATION AND SCREEN',
             'Inventory includes every distinct Louvain community, with all resolution/seed appearances retained.',
             'Subcommunities are connected components of the patient-induced graph within a community, and adjacent patient pairs within a community.',
             'Connected pieces of any disconnected Louvain output are also included. The requested night pair is explicitly added.',
             'This is a defined candidate pool, not an exhaustive search over every connected vertex subset.',
             'A candidate is a proper connected subset, size>=2, PAT count>=2, with boundary cardinality<=floor(m/10).',
             'The10% relationship screen is an operational exploratory rule, not an optimized or clinical threshold.',
             'Cumulative seconds count pair-contact bins; duration is strength, not removal cost or elapsed closure time.',
             'Weighted conductance=boundary duration/min(weighted volume(C),weighted volume(V-C)); volume(C)=2*internal duration+boundary duration.',
             'Evaluation removes ALL original person-level edges crossing C. Source priors and target objectives are unchanged.',
             'Static score is the exact component identity; direct source averaging independently checks every candidate/objective.',
             'Scores count protected targets throughout the graph, not only within the cohort. A cohort is safe only when its source is outside it.',
             'Four objectives: uniform/all, HCW/all, PAT/all, HCW/PAT. HCW=NUR/MED/ADM.',
             'Exact comparisons use the SAME baseline and boundary-cardinality budget. Earlier thresholded optima are never substituted.',
             'Global optimum: ratio1; near-optimal: 0.9<=ratio<1; substantially worse: ratio<0.9. The0.9 cutoff is descriptive.',
             'No exact comparisons beyond b=6. Budgets5/6 on smaller graphs are allowed only if a preflight finds<=100000 bundle subsets.',
             'No candidate in this run requires an exact budget5 or6 extension.',
             '', 'STABILITY DEFINITIONS',
             'Exact: C equals a full Louvain community. Larger subset: C is strictly contained in a larger community; these are disjoint counts.',
             'Patients together: all patient IDs in C share one community, ignoring any staff in C and allowing additional patients.',
             'Exact patients: some full community has exactly the patient IDs of C, allowing arbitrary staff. Staff variants list these staff sets.',
             'For staff-only communities patient stability is NA. Per-resolution counts and exact run keys are recorded.',
             'A patient pair can be consistently co-assigned without having stable exact patient membership at every resolution.',
             '', 'GRAPH AUDIT']
    for name,graph in graphs.items():
        candidates = [r for r in records if r['graph']==name and r['candidate']]
        lines.append(f"{name}: n={len(graph)}, m={graph.number_of_edges()}, lambda={nx.edge_connectivity(graph)}, "
                     f"roles={dict(Counter(graph.nodes[v]['role'] for v in graph))}, boundary screen={graph.number_of_edges()//10}, candidates={len(candidates)}")
        lines.append('  Exact search: '+json.dumps(searches[name],sort_keys=True))
    lines += ['', 'RAW VERIFICATION OF REQUESTED NIGHT PAIR',json.dumps(audit,sort_keys=True),
              '', 'ALL LOUVAIN PARTITIONS BY RESOLUTION AND SEED']
    for name,values in runs.items():
        for run in values:
            lines.append(f"{name} resolution={run['resolution']} seed={run['seed']} modularity={run['modularity']:.9f} communities={run['communities']}")
    lines += ['', 'COMPLETE COMMUNITY / SUBCOMMUNITY INVENTORY',
              'Each unique set is shown once; repeated appearances are recorded, so no Louvain output is omitted.']
    for r in records:
        st = r['stability']
        lines.append(f"{r['graph']} {r['ids']} sources={r['sources']} candidate={r['candidate']} size={r['size']} roles={r['roles']} "
                     f"internal edges/seconds={r['internaledges']}/{r['internalseconds']} boundary edges/seconds={r['boundaryedges']}/{r['boundaryseconds']} "
                     f"conductance={r['conductance']} connected={r['connected']}")
        lines.append(f"  exact={st['exact']}/80; strict subset={st['largersubset']}/80; exact patients={st['exactpatients']}/80; patients together={st['patientstogether']}/80")
        lines.append('  Per resolution: '+json.dumps(st['byresolution']))
        if not r['candidate']: continue
        ev = r['evaluation']
        lines.append(f"  Boundary={r['boundary']}; resulting components={ev['components']}; exactly two={ev['exactlytwo']}")
        lines.append('  Staff variants with identical patient membership: '+json.dumps(st['staffvariants']))
        for name,data in ev['objectives'].items():
            lines.append(f"  {name}: expected={data['value']} ({data['decimal']:.9f}); cohort safe probability={data['cohortsafeprobability']}; "
                         f"cohort contribution={data['cohortcontribution']}; {data['classification']}")
            if data['exact']:
                lines.append(f"    Exact={data['exact']['value']}; ratio={data['ratio']}; gap={data['gap']}; exact cut={data['exact']['cut']}; components={data['exact']['components']}")
    lines += ['', 'BOUNDARY DETAILS FOR SHORTLISTED COHORTS',
              'Shortlist: all candidates with b<=6, plus the lowest-cardinality candidate in each graph; ties prefer patient stability, then exact stability and duration.',
              'These are interpretable examples, not a claim of one universally best cohort. All candidate boundary details are in the CSV.',
              'edge | endpoint roles | cumulative seconds | ordinary BC rank | inverse-duration BCw rank | individual bridge']
    for r in records:
        if not r['shortlisted']: continue
        lines.append(f"{r['graph']} cohort={r['ids']} cost={r['boundaryedges']} relationships/{r['boundaryseconds']} pair-contact seconds")
        for d in r['boundarydetails']:
            lines.append(f"  {d['edge']} | {d['roles']} | {d['duration']} | {d['bcrank']} | {d['bcwrank']} | {d['bridge']}")
    lines += ['', 'REPRODUCIBILITY AND LIMITATIONS',
              'PYTHONDONTWRITEBYTECODE=1 python code/hospital.py --cohorts',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py hospitalcohorts',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py hospital',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py hospitalinterpret',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py',
              f'Analysis elapsed seconds: {seconds:.6f}',
              'This hypothetical cohorting/scheduling interpretation suppresses all observed relationships across a chosen boundary; operational feasibility is not measured.',
              'Temporal ordering, transmission probabilities and clinical roles beyond the four recorded categories are not modeled.',
              'Same patient membership across algorithm seeds does not establish longitudinal or clinical cohort stability.',
              'No production function, prior result file or manuscript was changed.']
    pair = next(r for r in records if r['graph']=='night' and r['ids']==(1460,1784))
    st,ev = pair['stability'],pair['evaluation']
    lines += ['', 'Strongest candidate cohort',
              'Within the requested exact-comparison range b<=6, the only qualifying discovered cohort is night {1460,1784}, two patients.',
              f"Its internal contact is{pair['internalseconds']}s and its boundary costs{pair['boundaryedges']} relationships/{pair['boundaryseconds']}s; weighted conductance={pair['conductance']}.",
              f"It is a complete community in{st['exact']}/80 runs and a strict subset in{st['largersubset']}/80; the patients are together in{st['patientstogether']}/80 and form the exact patient set in{st['exactpatients']}/80.",
              'At resolution0.75 it is embedded in a larger community in all20 seeds; at1.0,1.25,1.5 it is exactly the pair in all60 seeds.',
              'The four crossing edges connect the patients to NUR1202, NUR1485 and ADM1658. Their internal patient-patient edge is retained.',
              'Under HCW-only sources both patients are protected with probability1. This is the strongest small-budget candidate in the defined pool, not a proven best cohort over all subsets.',
              '', 'Comparison to edge-by-edge optimum']
    for name,data in ev['objectives'].items():
        lines.append(f"{name}: cohort={data['value']} ({data['decimal']:.9f}); exact={data['exact']['value']}; ratio={data['ratio']} ({data['ratiodecimal']:.6%}); {data['classification']}.")
    lines += ['The cohort cut produces components of sizes2 and51. The unrestricted optima at b=4 instead use singleton separations; no claim of cohort optimality is made.',
              'Larger candidate boundaries are evaluated exactly as fixed interventions, but their global optima are not computed beyond the requested budget limit.',
              '', 'Candidate paper finding',
              'In the measured unthresholded night aggregate, a two-patient subcohort is recovered as an exact weighted-Louvain community at all20 seeds for each of resolutions1.0,1.25,1.5, while remaining co-assigned at0.75.',
              'Suppressing its four external relationships (200 cumulative pair-contact seconds) retains180 seconds of internal contact and guarantees protection of both cohort patients for an HCW-only source under the static model.',
              'This interpretable group boundary is not cardinality-optimal: at the same four-edge budget it achieves about50.5% of uniform-source/all-person OPT and50% of HCW-source/patient-target OPT.']
    return '\n'.join(lines)+'\n'

def cohorts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,help='Separate directory for new results and report')
    args = parser.parse_args()
    destination = savedreports.prepare(args.output,('hospitalcohorts.csv',),('hospitalcohorts',))
    records,graphs,runs,searches,audit,seconds = analyzecohorts()
    path = destination/'hospitalcohorts.csv'
    with path.open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        for r in records:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})
    savedreports.save('hospitalcohorts',cohortreport(records,graphs,runs,searches,audit,seconds),directory=destination)
    pair = next(r for r in records if r['graph']=='night' and r['ids']==(1460,1784))
    print('PASS night pair: 2 PAT; internal180s; boundary4 edges/200s; components',pair['evaluation']['componentsizes'])
    print('Pair stability:',json.dumps(pair['stability']['byresolution']))
    for name,d in pair['evaluation']['objectives'].items():
        print(name,'cohort',d['value'],'exact',d['exact']['value'],'ratio',d['ratio'])
    print('Wrote',len(records),'unique community/subcommunity rows;',sum(r['candidate'] for r in records),'candidates;',sum(len(r) for r in runs.values()),'Louvain runs')
    print('Seconds',round(seconds,3))

if __name__=='__main__':
    modes = {'--interpret':interpret, '--cohorts':cohorts}
    selected = [flag for flag in modes if flag in sys.argv]
    if len(selected)>1: raise SystemExit('Choose one analysis mode')
    action = modes[selected[0]] if selected else main
    if selected: sys.argv.remove(selected[0])
    if '--help' in sys.argv or '-h' in sys.argv:
        print('Additional analysis modes: --interpret, --cohorts')
        action()
    else: action()
