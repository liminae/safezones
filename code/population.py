# Static population-weighted commuting analysis
import records as savedreports
import argparse
import csv
import hashlib
import io
import json
import platform
import signal
import sys
import urllib.request
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from time import perf_counter
import networkx as nx

root = Path(__file__).resolve().parent.parent
folder = root/'data/results'
import commuting
from hospital import edges,components,score

populationurl = 'https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt'
populationhash = '4f754a3bda9714e5820d73bfac44ef1dab2b8c371f6c136305474cda7a752014'
documentation = 'https://www.census.gov/geographies/reference-files/time-series/geo/centers-population.html'
release = 'https://www.census.gov/newsroom/press-releases/2021/center-of-population-hartville-missouri.html'
apiurl = 'https://api.census.gov/data/2020/dec/pl?get=NAME,P1_001N&for=county:*&in=state:*'
connecticuturl = 'https://www.govinfo.gov/content/pkg/FR-2022-06-06/html/2022-12063.htm'
connecticuthash = 'f64fdd6e45f96232cc09d41c1463e849ce0210c94cdade40c5fef3c2ac185ccf'
connecticut = {'09110':('Capitol',976248),'09120':('Greater Bridgeport',325778),
               '09130':('Lower Connecticut River Valley',174225),'09140':('Naugatuck Valley',450376),
               '09150':('Northeastern Connecticut',95348),'09160':('Northwest Hills',112503),
               '09170':('South Central Connecticut',570487),'09180':('Southeastern Connecticut',280430),
               '09190':('Western Connecticut',620549)}
source = root/'data/raw/table1.xlsx'
regional = ('08003','08021','08023','08079','08105','08109')
regionalcuts = (('08015','08109'),('08023','35055'))
modes = dict(A='uu',B='mixed',C='pp')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def signature(graph):
    return hashlib.sha256(json.dumps([sorted(graph),edges(graph)],separators=(',',':')).encode()).hexdigest()

def snapshot():
    paths = []
    for directory in ('code','data','figures'):
        paths += [p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.csv','.txt','.json','.gz','.zip')]
    return {str(p.relative_to(root)):digest(p) for p in sorted(paths)}

def fetch():
    path = root/'data/raw/population.csv'
    if path.exists(): raise FileExistsError('Refusing to overwrite population.csv')
    with urllib.request.urlopen(populationurl,timeout=45) as response:
        data = response.read()
        if response.url!=populationurl: raise RuntimeError('Unexpected population download redirect')
    if hashlib.sha256(data).hexdigest()!=populationhash: raise ValueError('Official input changed; review its provenance before use')
    assert len(list(csv.DictReader(io.StringIO(data.decode('utf-8-sig')))))==3221
    with path.open('xb') as stream: stream.write(data)
    print('Saved unchanged official CSV:',path,flush=True)
    print('SHA256:',populationhash,flush=True)

def populations():
    path = root/'data/raw/population.csv'
    if not path.exists(): raise FileNotFoundError('Run with --fetch to retrieve the official Census input')
    if digest(path)!=populationhash: raise ValueError('Population input SHA256 mismatch')
    records = {}
    with path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            state,county = row['STATEFP'],row['COUNTYFP']
            assert len(state)==2 and state.isdigit() and len(county)==3 and county.isdigit()
            fips = state+county
            assert fips not in records
            value = int(row['POPULATION'])
            assert value>=0
            records[fips] = dict(population=value,name=row['COUNAME'],state=row['STNAME'])
    assert len(records)==3221
    assert sum(r['population'] for v,r in records.items() if v[:2]!='72')==331449281
    assert sum(w for name,w in connecticut.values())==sum(r['population'] for v,r in records.items() if v[:2]=='09')==3605944
    for fips,(name,value) in connecticut.items():
        assert fips not in records
        records[fips] = dict(population=value,name=name+' Planning Region',state='Connecticut')
    return records

def baseline():
    if digest(source)!=commuting.sourcehash: raise ValueError('Commuting workbook SHA256 mismatch')
    nodes,pairs,audit = commuting.readflows(commuting.workbookrows(source))
    graph = commuting.buildgraph(nodes,pairs,1,True)
    assert (len(nodes),len(graph),graph.number_of_edges())==(3109,3105,24338)
    assert nx.is_connected(graph) and commuting.boundary(graph,regional)==regionalcuts
    assert graph.subgraph(regional).number_of_edges()==8
    return graph,dict(audit,excluded=sorted(set(nodes)-set(graph)))

def join(graph,records):
    missing = sorted(set(graph)-set(records))
    print('Unmatched graph FIPS:',missing,flush=True)
    if missing: raise ValueError('Unmatched populations; graph was not modified: '+str(missing))
    weights = {v:records[v]['population'] for v in sorted(graph)}
    assert all(isinstance(w,int) and w>=0 for w in weights.values()) and sum(weights.values())>0
    assert all(graph.nodes[v]['state']==records[v]['state'] for v in graph)
    assert all(graph.nodes[v]['name']==records[v]['name'] for v in connecticut)
    canonical = 'fips,population\n'+''.join(f'{v},{weights[v]}\n' for v in sorted(weights))
    return weights,hashlib.sha256(canonical.encode()).hexdigest()

def priors(graph,weights,objective):
    if objective not in modes: raise ValueError('Unknown objective')
    total = sum(weights.values()) if objective=='C' else len(graph)
    p = {v:Fraction(weights[v] if objective=='C' else 1,total) for v in graph}
    a = {v:1 if objective=='A' else weights[v] for v in graph}
    assert sum(p.values())==1
    return p,a

def values(graph,cuts,weights):
    groups = sorted(components(graph,cuts),key=lambda c:(-len(c),c))
    sizes = [len(c) for c in groups]
    masses = [sum(weights[v] for v in c) for c in groups]
    n,W = len(graph),sum(weights.values())
    result = dict(A=Fraction(n*n-sum(s*s for s in sizes),n),
                  B=Fraction(n*W-sum(s*w for s,w in zip(sizes,masses)),n),
                  C=Fraction(W*W-sum(w*w for w in masses),W))
    return result,groups,masses

def marginals(graph,weights,objective,n,W):
    bridges = {tuple(sorted(e)) for e in nx.bridges(graph)}
    sizes,masses = dict.fromkeys(graph,1),dict(weights)
    forest = list(nx.dfs_edges(graph))
    for u,v in reversed(forest):
        sizes[u] += sizes[v]
        masses[u] += masses[v]
    totals = {}
    for group in nx.connected_components(graph):
        count,mass = len(group),sum(weights[v] for v in group)
        totals.update((v,(count,mass)) for v in group)
    gains = dict.fromkeys(edges(graph),0)
    for u,v in forest:
        edge = tuple(sorted((u,v)))
        if edge not in bridges: continue
        count,mass = totals[v]
        x,y = sizes[v],count-sizes[v]
        p,q = masses[v],mass-masses[v]
        gains[edge] = 2*x*y if objective=='A' else (x*q+y*p if objective=='B' else 2*p*q)
    denominator = W if objective=='C' else n
    return {e:Fraction(g,denominator) for e,g in gains.items()}

def greedy(graph,weights,objective,b=2):
    remaining,cuts,steps = graph.copy(),[],[]
    for j in range(b):
        gains = marginals(remaining,weights,objective,len(graph),sum(weights.values()))
        best = max(gains.values())
        tied = sorted(e for e,value in gains.items() if value==best)
        edge = tied[0]
        before = values(graph,cuts,weights)[0][objective]
        cuts.append(edge)
        after = values(graph,cuts,weights)[0][objective]
        assert after-before==best
        steps.append(dict(round=j+1,edge=edge,gain=str(best),score=str(after),ties=len(tied)))
        remaining.remove_edge(*edge)
    return cuts,steps

def adapted(graph,weights):
    result = graph.copy()
    nx.set_node_attributes(result,weights,'resident')
    return result

def expired(signum,frame):
    raise TimeoutError('Population exact search reached the time cap')

def optimize(graph,weights,cap):
    start = perf_counter()
    previous = signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,cap)
    try:
        data = commuting.catalog(adapted(graph,weights),2,limit=2000000)
        result,stats = commuting.exact(data,limit=2000000)
        info = {k:v for k,v in data['stats'].items() if k!='core'}
        return result,dict(info,**stats,wallseconds=perf_counter()-start,status='verified',error='')
    except (RuntimeError,TimeoutError) as error:
        return None,dict(status='unverified',error=str(error),wallseconds=perf_counter()-start)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)

def cached(graph,weights):
    path = root/'data/results/commutingresults.csv'
    rows = [r for r in csv.DictReader(path.open()) if r['prior']=='uniform' and r['target']=='counties' and int(r['budget'])<=2]
    assert len(rows)==2
    targets = {1:Fraction(2068,345),2:Fraction(4132,345)}
    anchor,policies = {},{}
    for row in rows:
        b = int(row['budget'])
        cuts = tuple(tuple(e) for e in json.loads(row['optimalcut']))
        value,groups,masses = values(graph,cuts,weights)
        assert value['A']==Fraction(row['exact'])==targets[b]
        assert list(map(len,groups))==json.loads(row['componentsizes'])
        anchor[b] = dict(value=value['A'],cuts=cuts)
        policies['initial',b] = dict(cuts=tuple(tuple(e) for e in json.loads(row['bccuts'])),ties=[],source='data/results/commutingresults.csv',status='cached',error='')
        print(f'PASS cached A b={b}: {value["A"]}; cut={cuts}; sizes={list(map(len,groups))}',flush=True)
    path = root/'data/results/baselines.csv'
    if path.exists():
        for row in csv.DictReader(path.open()):
            if row['case']!='commuting' or row['policy'] not in ('initial','recomputed') or int(row['budget'])>2: continue
            assert row['graphhash']==signature(graph)
            b = int(row['budget'])
            if row['status']!='complete':
                if row['policy']=='recomputed': policies['recomputed',b] = dict(cuts=None,ties=[],source=str(path.relative_to(root)),status='unavailable',error=row['error'])
                continue
            cuts = tuple(tuple(e) for e in json.loads(row['cuts']))
            assert values(graph,cuts,weights)[0]['A']==Fraction(row['score'])
            steps = json.loads(row['steps'])
            if row['policy']=='initial': assert cuts==policies['initial',b]['cuts']
            policies[row['policy'],b] = dict(cuts=cuts,ties=[step['ties'] for step in steps],source=str(path.relative_to(root)),status='cached',error='')
    return anchor,policies

def tests():
    fixtures = [nx.path_graph(4),nx.cycle_graph(5),nx.complete_graph(4),nx.Graph([(0,1),(1,2),(2,0),(2,3),(3,4)])]
    count = 0
    for graph in fixtures:
        weights = {v:(0,2,7,11,3)[v] for v in graph}
        nx.set_edge_attributes(graph,1,'flow')
        data = commuting.catalog(adapted(graph,weights),2)
        exact,stats = commuting.exact(data)
        best = {name:{b:Fraction(-1) for b in (0,1,2)} for name in modes}
        for b in (0,1,2):
            for cuts in combinations(edges(graph),b):
                remaining = graph.copy()
                remaining.remove_edges_from(cuts)
                result,groups,masses = values(graph,cuts,weights)
                for name in modes:
                    p,a = priors(graph,weights,name)
                    direct = sum(p[s]*sum(a[v] for v in set(graph)-nx.node_connected_component(remaining,s)) for s in graph)
                    assert direct==result[name]==score(graph,cuts,p,a)
                    for budget in range(b,3): best[name][budget] = max(best[name][budget],direct)
                    gains = marginals(remaining,weights,name,len(graph),sum(weights.values()))
                    for edge,gain in gains.items():
                        assert result[name]+gain==values(graph,cuts+(edge,),weights)[0][name]
                count += 1
        for name,mode in modes.items():
            denominator = sum(weights.values()) if name=='C' else len(graph)
            for b in (0,1,2):
                assert Fraction(exact[mode][b]['numerator'],denominator)==best[name][b]
                for cuts in exact[mode][b]['cuts']: assert values(graph,cuts,weights)[0][name]==best[name][b]
            cuts,steps = greedy(graph,weights,name)
            assert values(graph,cuts,weights)[0][name]<=best[name][2]
        assert values(graph,(),weights)[0]==dict.fromkeys(modes,Fraction(0))
    return f'PASS {count} small graph/cut partitions: direct source average, component formula, production scorer, exact optima and every greedy marginal (including zero populations)'

def county(graph,weights,v):
    return dict(fips=v,name=graph.nodes[v]['name'],state=graph.nodes[v]['state'],population=weights[v])

def record(graph,weights,objective,b,policy,cuts,reference,provenance,ties=(),status='evaluated',error=''):
    n,W = len(graph),sum(weights.values())
    value = reference['value'] if reference else None
    row = dict(objective=objective,prior='population' if objective=='C' else 'uniform',target='counties' if objective=='A' else 'population',
               budget=b,policy=policy,status=status,error=error,n=n,m=graph.number_of_edges(),W=W,cuts=cuts,
               feasible=cuts is not None and len(cuts)<=b,ties=list(ties),provenance=provenance,
               optimumstatus='verified' if reference else 'unverified',exact=str(value) if value is not None else '',
               exactdecimal=float(value) if value is not None else '',optimalboundaries=reference['allcuts'] if reference else [],
               score='',scoredecimal='',ratio='',ratiodecimal='',isoptimal='',expectedpopulation='',expectedpopulationdecimal='',fractionW='',fractionWdecimal='',
               scores={},componentcount='',componentsizes=[],componentpopulations=[],partition=[],nongiant=[],componentdetails=[],removed=[],disruption='')
    if cuts is None: return row
    result,groups,masses = values(graph,cuts,weights)
    p,a = priors(graph,weights,objective)
    actual = result[objective]
    assert actual==score(graph,cuts,p,a)
    expected = result['C' if objective=='C' else 'B']
    row.update(score=str(actual),scoredecimal=float(actual),scores={name:str(v) for name,v in result.items()},
               expectedpopulation=str(expected),expectedpopulationdecimal=float(expected),fractionW=str(expected/W),fractionWdecimal=float(expected/W),
               componentcount=len(groups),componentsizes=list(map(len,groups)),componentpopulations=masses,partition=groups,
               nongiant=[[county(graph,weights,v) for v in group] for group in groups[1:]],
               removed=[dict(edge=e,endpoints=[county(graph,weights,v) for v in e],flow=graph.edges[e]['flow']) for e in cuts],
               disruption=sum(graph.edges[e]['flow'] for e in cuts))
    for group,mass in zip(groups,masses):
        probability = sum(p[v] for v in group)
        row['componentdetails'].append(dict(size=len(group),population=mass,sourceprobability=str(probability),
                                            safeprobability=str(1-probability),expectedprotectedpopulation=str(mass*(1-probability))))
    assert sum(Fraction(c['expectedprotectedpopulation']) for c in row['componentdetails'])==expected
    if not row['feasible']:
        row['status'] = 'infeasible'
    elif value is not None:
        assert actual<=value
        row.update(ratio=str(actual/value) if value else '',ratiodecimal=float(actual/value) if value else '',isoptimal=actual==value)
    return row

def describe(row):
    return '; '.join(', '.join(c['fips']+' '+c['name']+' ('+c['state']+')' for c in group) for group in row['nongiant']) or 'none; connected graph'

def report(rows,graph,weights,populationrecords,usedhash,audit,statistics,checks,seconds,protected):
    n,W = len(graph),sum(weights.values())
    lines = ['POPULATION-WEIGHTED COUNTY COMMUTING: STATIC ANALYSIS',
             f'Graph: {n} counties, {graph.number_of_edges()} edges; kappa=1, original largest connected component; W={W}.',
             'All removals are static, before infection; unit edge-cardinality budgets 1 and 2. No sequential simulator is used.',
             f'Python {platform.python_version()}; NetworkX {nx.__version__}.',
             '', 'POPULATION SOURCE AND JOIN',
             'Preferred API request: '+apiurl,
             'The API redirected to https://api.census.gov/data/missing_key.html on retrieval. No API key was invented or used.',
             'Fallback: official Census Bureau 2020 Centers of Population county file, POPULATION column. This is a published 2020 Census count, not a population estimate.',
             'Raw input URL: '+populationurl,'Record layout: '+documentation,'Census redistricting provenance: '+release,
             'Census states this release uses 2020 redistricting data; county POPULATION is the 2020 Census tabulation. This file is not relabeled as an API P1_001N response.',
             'Vintage/reference date: 2020 Decennial Census, April 1, 2020; downloaded 2026-11-15. Raw official bytes retained as population.csv.',
             'Raw population.csv SHA256: '+populationhash,
             'Used 3105-record SHA256: '+usedhash,
             'Used-record hash serialization: UTF-8, header fips,population followed by sorted five-digit FIPS,integer population rows, LF line endings including final LF.',
             f'Official raw input records: 3221; augmented lookup records: {len(populationrecords)}; US raw total excluding Puerto Rico checked as 331449281. Matched graph vertices: {len(weights)}/{n}; final unmatched IDs: [].',
             'The raw 2020 geography initially left nine graph IDs unmatched: '+str(sorted(connecticut))+'. The existing commuting workbook uses the nine post-2022 Connecticut planning regions, not its eight legacy counties.',
             'Connecticut source: '+connecticuturl+' (Census Bureau notice, June 6, 2022; Table 1 population counts, Table 4 FIPS codes).',
             'Connecticut official notice HTML SHA256 at verification: '+connecticuthash,
             'Published 2020 region counts were independently extracted from Table 1 and checked against the constants below; these are Census-published sums of constituent-town counts, not estimates or proportional county allocations.',
             'Connecticut supplements: '+json.dumps(connecticut,sort_keys=True),
             'Both the nine-region and eight-legacy-county counts sum to 3605944. The eight old Connecticut IDs are absent from the evaluated graph and are not double-counted.',
             'Join uses exact five-digit FIPS: STATEFP+COUNTYFP for the raw file, and official Table 4 codes for the Connecticut supplements. No name-based graph join, estimates, imputation, or vertex deletion. All graph populations are nonnegative integers.',
             'State names independently agree. Original county labels are retained even where Census display names omit County/Parish suffixes.',
             'Four counties already absent from the original giant component: '+str(audit['excluded'])+'. No additional county is dropped.',
             'All source probabilities are normalized on these same 3105 vertices. Population weights are fixed for every intervention.',
             '', 'OBJECTIVES AND INTERPRETATION',
             'A: uniform county source, county targets; Q_A=(n*n-sum size(C)^2)/n, measured in expected counties.',
             'B: uniform county source, population targets; Q_B=(n*W-sum size(C)*population(C))/n, measured in expected people.',
             'C (primary): population-proportional county source, population targets; Q_C=(W*W-sum population(C)^2)/W, measured in expected people.',
             'For A interventions the separately reported expected population is Q_B, because A uses a uniform county source. For B and C expected population equals the objective.',
             'fractionW uses that expected population divided by W. Ratios to OPT always compare the same objective and budget.',
             'A detached/non-giant component is not guaranteed safe: it is protected only if the random source lies in another component.',
             'For each component, the CSV records size, census population, source probability and population*(1-source probability), including the giant component.',
             'Thus expected protected population can exceed the detached population: a source in a detached region protects the large remainder.',
             'Infeasible fixed cuts retain their actual partition score for transparency, are explicitly marked infeasible, and have no achieved/OPT ratio.',
             '', 'EXACTNESS AND VALIDATION',*checks,
             'Existing commuting.catalog and commuting.exact are reused. A temporary graph copy maps population to resident, which the existing catalog maps to its inbound mass key. Original commuter attributes and graph data are not changed.',
             'Unit-capacity Gomory-Hu contraction at budget 2 groups vertices whose pairwise edge min-cut exceeds 2; no feasible removal can split such a block.',
             'Every final partition uses whole blocks. Restoring removed edges inside a final component preserves all three objectives, so only minimal partition boundaries need be enumerated.',
             'The quotient is split into branches around its largest block. The existing local search enumerates all bundle subsets of cost <=2; the existing global search enumerates all combinations of branch partitions of total cost <=2.',
             'All comparisons use integer numerators uu, mixed, pp, divided exactly by n,n,W. No population-based pruning is introduced.',
             'All tied optimal minimal boundaries are retained. Representative ties prefer fewer removed edges, then canonical lexicographic FIPS edge tuples.',
             'Search statistics: '+json.dumps(statistics,sort_keys=True),
             'Incomplete searches would be marked unverified with blank OPT and ratios; no candidate cut is substituted for an optimum.',
             '', 'BASELINES',
             'Initial BC and recomputed BC interventions are read from unchanged caches and their partitions replayed on the reconstructed graph. No centralities or earlier sweeps are rerun.',
             'Cached BC uses unnormalized ordinary NetworkX values times 2 for ordered pairs and canonical edge ties. Stored BC tie counts are reused; unavailable counts remain null.',
             'Objective Greedy is run separately for A, B and C. It selects the exact maximum marginal gain, canonical lexicographic ties, including zero-gain choices if necessary.',
             'Only a current bridge has nonzero one-step gain. For a split into sides with sizes x,y and populations p,q, marginal gains are 2xy/n, (xq+yp)/n, and 2pq/W for A,B,C.',
             'Bridge status, DFS subtree sizes and population sums evaluate these exact gains on each current graph. Brute-force small tests independently check every edge gain.',
             '', 'RESULTS',
             'objective | budget | intervention | status | score | expected population | fraction of W | ratio | component sizes | component populations | ties']
    for row in rows:
        lines.append(' | '.join(str(row[k]) for k in ('objective','budget','policy','status','score','expectedpopulation','fractionW','ratio','componentsizes','componentpopulations','ties')))
        lines.append('  Cuts: '+json.dumps(row['removed'],sort_keys=True))
        lines.append('  Non-giant regions: '+describe(row))
        lines.append('  Component contributions: '+json.dumps(row['componentdetails'],sort_keys=True))
        if row['policy']=='exact': lines.append('  All co-optimal minimal boundaries: '+json.dumps(row['optimalboundaries']))
    lookup = {(r['objective'],r['budget'],r['policy']):r for r in rows}
    lines += ['', 'ANSWERS TO THE RESEARCH QUESTIONS']
    for objective in ('B','C'):
        r = lookup[objective,2,'San Luis Valley']
        lines.append(f'1. San Luis Valley under {objective}, b=2: score {r["score"]} ({r["scoredecimal"]:.9f}), optimal={r["isoptimal"]}, ratio={r["ratio"]}.')
    for b in (1,2):
        left,right = lookup['B',b,'exact'],lookup['C',b,'exact']
        if left['cuts'] is not None and right['cuts'] is not None:
            lines.append(f'2. Uniform to population source, b={b}: representative cut changes={left["cuts"]!=right["cuts"]}. B: {describe(left)}. C: {describe(right)}.')
    largest = sorted(graph,key=lambda v:(-weights[v],v))[:5]
    lines.append('3. Five largest counties by population: '+json.dumps([dict(county(graph,weights,v),degree=graph.degree(v)) for v in largest]))
    for b in (1,2):
        r = lookup['C',b,'exact']
        if r['cuts'] is not None:
            detached = {c['fips'] for group in r['nongiant'] for c in group}
            lines.append(f'   Primary optimum b={b}: detached population {sum(weights[v] for v in detached)} ({float(Fraction(sum(weights[v] for v in detached),W)):.9%} of W), '
                         f'contains any of the five largest counties={bool(detached&set(largest))}; regions={describe(r)}.')
    lines.append('   Population matters only among regions separable within budget. These results do not imply that the largest metropolitan counties are removable with one or two edges.')
    for objective in modes:
        for b in (1,2):
            lines.append(f'4. {objective}, b={b}: '+ '; '.join(f'{policy}: score={lookup[objective,b,policy]["score"]}, ratio={lookup[objective,b,policy]["ratio"]}, ties={lookup[objective,b,policy]["ties"]}' for policy in ('Initial BC','Recomputed BC','Objective Greedy')))
    lines += ['5. This is deterministic reachability protection in a static county graph, not an estimate of actual infections, hospitalizations, or travel risk.',
              'Uniform county sources give a small county the same source probability as a large one. Population-proportional sources correspond to selecting a resident uniformly and locating the source in that resident\'s county, an assumption rather than an estimated introduction-risk model.',
              'Each infected county is treated as exposing all its residents; no within-county dynamics, transmission probabilities, timing or heterogeneity is represented.',
              'An edge deletion suppresses an entire aggregated commuting relationship in both directions. Budgets count relationships, not commuters, closure feasibility or economic disruption.',
              'The graph uses 2016-2020 ACS estimates filtered by their reported MOEs; population is a 2020 Census count. Reliability filtering and giant-component restriction condition every result.',
              'These findings concern only the evaluated graph, population vintage and budgets. They are not causal public-health forecasts or a recommendation to close the named links.',
              '', 'PROVENANCE AND PRESERVATION','Workbook: '+str(source),'Workbook SHA256: '+digest(source),'Graph topology SHA256: '+signature(graph),
              'Input/code hashes: '+json.dumps({name:protected[name] for name in ('code/commuting.py','code/air.py','code/hospital.py','data/results/commutingresults.csv','data/results/baselines.csv')},sort_keys=True),
              'run.py SHA256: '+digest(__file__),f'All {len(protected)} protected existing files have unchanged SHA256 hashes.',
              f'Analysis runtime before serialization: {seconds:.6f}s.',
              'Commands:','PYTHONDONTWRITEBYTECODE=1 python code/population.py --fetch',
              'PYTHONDONTWRITEBYTECODE=1 python code/population.py --check',
              'PYTHONDONTWRITEBYTECODE=1 python code/population.py',
              'Script: code/population.py; outputs: data/results/population.csv, its entry in data/records.json and the reproducibility input population.csv are used.']
    return '\n'.join(lines)+'\n'

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fetch',action='store_true')
    group.add_argument('--check',action='store_true')
    parser.add_argument('--timecap',type=float,default=180)
    args = parser.parse_args()
    if args.timecap<=0: parser.error('Time cap must be positive')
    if args.fetch:
        fetch()
        return
    if not args.check and ((folder/'population.csv').exists() or savedreports.exists('population')): raise FileExistsError('Refusing to overwrite analysis outputs')
    start = perf_counter()
    protected = snapshot()
    checks = [tests()]
    print(checks[0],flush=True)
    if args.check:
        if (root/'data/raw/population.csv').exists(): print(f'PASS official population input: {len(populations())} records and checksum')
        assert snapshot()==protected
        return
    graph,audit = baseline()
    original = nx.node_link_data(graph)
    anchor,policies = cached(graph,dict.fromkeys(graph,1))
    populationrecords = populations()
    weights,usedhash = join(graph,populationrecords)
    print(f'PASS graph {len(graph)}/{graph.number_of_edges()}; population matched {len(weights)}; W={sum(weights.values())}',flush=True)
    for name in modes:
        p,a = priors(graph,weights,name)
        assert score(graph,(),p,a)==values(graph,(),weights)[0][name]==0
    checks += ['PASS original 3105-vertex/24338-edge graph and both cached A optimum cut/score replays',
               'PASS complete exact FIPS-population join; nonnegative counts; source distributions sum exactly to one; Q(empty)=0 for A,B,C']
    print('Enumerating exact budget <=2 partitions once for A, B and C',flush=True)
    solutions,statistics = optimize(graph,weights,args.timecap)
    references = {}
    if solutions is not None:
        for name,mode in modes.items():
            denominator = sum(weights.values()) if name=='C' else len(graph)
            for b in (1,2):
                entry = solutions[mode][b]
                value = Fraction(entry['numerator'],denominator)
                for cuts in entry['cuts']: assert values(graph,cuts,weights)[0][name]==value
                references[name,b] = dict(value=value,cuts=entry['cuts'][0],allcuts=entry['cuts'])
                if name=='A': assert value==anchor[b]['value'] and anchor[b]['cuts'] in entry['cuts']
                print(f'EXACT {name} b={b}: {value} ({float(value):.9f}), cuts={entry["cuts"][0]}, optimal boundaries={len(entry["cuts"])}',flush=True)
        checks.append('PASS all six exact optima verified by partition replay; both A values reproduced by the exhaustive solver')
    else: print('UNVERIFIED:',statistics,flush=True)
    rows = []
    for name in modes:
        greedycuts,steps = greedy(graph,weights,name)
        for b in (1,2):
            ref = references.get((name,b))
            methods = [('exact',ref['cuts'] if ref else None,'commuting.catalog/exact',[], 'verified' if ref else 'unverified',statistics['error']),
                       ('San Luis Valley',regionalcuts,'original measured two-edge boundary',[],'evaluated',''),
                       ('Big Bend',(commuting.bridge,),'original measured bridge',[],'evaluated',''),
                       ('Objective Greedy',tuple(greedycuts[:b]),json.dumps(steps[:b]),[step['ties'] for step in steps[:b]],'evaluated','')]
            for policy,title in (('initial','Initial BC'),('recomputed','Recomputed BC')):
                item = policies.get((policy,b),dict(cuts=None,source='cache unavailable',ties=[],status='unavailable',error='No completed cached intervention'))
                methods.append((title,item['cuts'],item['source'],item['ties'],item['status'],item['error']))
            for title,cuts,provenance,ties,status,error in methods:
                rows.append(record(graph,weights,name,b,title,cuts,ref,provenance,ties,status,error))
    assert len(rows)==36 and nx.node_link_data(graph)==original
    assert snapshot()==protected
    checks.append('PASS all feasible candidate scores <= same-objective verified OPT; independent partition/production scores agree; graph attributes and existing files unchanged')
    text = report(rows,graph,weights,populationrecords,usedhash,audit,statistics,checks,perf_counter()-start,protected)
    with (folder/'population.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(tuple,list,dict)) else v for k,v in row.items()})
    savedreports.save('population',text,exclusive=True)
    for line in checks: print(line,flush=True)
    for row in rows:
        print(f'{row["objective"]} b={row["budget"]} {row["policy"]}: {row["status"]}, score={row["score"]}, ratio={row["ratio"]}, populations={row["componentpopulations"]}',flush=True)
    print(f'Completed {len(rows)} comparison records in {perf_counter()-start:.3f}s; population CSV and consolidated report written',flush=True)

if __name__=='__main__':
    main()
