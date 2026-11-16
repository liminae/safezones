# Livestock contact and directed reachability analyses
import records as savedreports
import argparse
import csv
import hashlib
import json
import platform
import signal
import sys
import urllib.request
from collections import Counter
from datetime import datetime
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from time import perf_counter
import networkx as nx
import air
import baselines as baseline
from hospital import components, edges, score
from itertools import permutations
from unittest.mock import patch

root = Path(__file__).resolve().parent.parent

folder = root/'data/results'

raw = root/'data/raw/livestock'

article = 'https://doi.org/10.1371/journal.pone.0232681'

sources = {
    'pone.0232681.s007.CSV':('https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0232681.s007&type=supplementary','1f5e3df7223d142d9e2fee4523ec685bb4dfb2c0f6fc61db3490d7dbfa21c517'),
    'pone.0232681.s008.CSV':('https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0232681.s008&type=supplementary','cf48ca6322ce1ef3129f081093d20d0165e4b8492663c3ceda85e36c310e5be0')}

periods = ('2017','2013-2017')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def snapshot():
    return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and p not in {folder/'livestock.csv'}
            and not any(v in p.relative_to(root).parts for v in ('.git','.venv','__pycache__'))}

def fetch():
    raw.mkdir(parents=True,exist_ok=True)
    for name,(url,expected) in sources.items():
        path = raw/name
        if path.exists():
            assert digest(path)==expected,name
            continue
        with urllib.request.urlopen(url,timeout=45) as response: data = response.read()
        if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('Official source checksum changed: '+name)
        with path.open('xb') as stream: stream.write(data)
        print('Downloaded',name,len(data),'bytes')

def read():
    data = {}
    for name,(url,expected) in sources.items():
        assert digest(raw/name)==expected,name
        with (raw/name).open(newline='',encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            data[name] = (reader.fieldnames,list(reader))
    columns,rows = data['pone.0232681.s007.CSV']
    fields,records = data['pone.0232681.s008.CSV']
    assert columns==['from','to','weight','type','transp','dmonth','intl','fromc','toc']
    assert fields==['id','name','country','cid']
    assert len(rows)==42252 and len(records)==262
    nodes = {r['id']:dict(name=r['name'],country=r['country'],cid=r['cid']) for r in records}
    assert len(nodes)==len(records) and all(all(r.values()) for r in records)
    for r in rows:
        assert all(r.values()) and r['from']!=r['to']
        assert r['from'] in nodes and r['to'] in nodes
        assert nodes[r['from']]['cid']==r['fromc'] and nodes[r['to']]['cid']==r['toc']
        assert (r['intl']=='National')==(r['fromc']==r['toc'])
        stamp = datetime.strptime(r['dmonth'],'%m/%d/%Y')
        assert stamp.day==1 and 2013<=stamp.year<=2017
        r['year'],r['month'],r['scaled'] = stamp.year,stamp.month,Fraction(r['weight'])
        assert r['scaled']>=0
    assert {v for r in rows for v in (r['from'],r['to'])}==set(nodes)
    audit = dict(rows=len(rows),metadata=len(nodes),columns=columns,nodecolumns=fields,
                 byyear=dict(sorted(Counter(r['year'] for r in rows).items())),
                 zero=sum(r['scaled']==0 for r in rows),countries=dict(sorted({(v['cid'],v['country']) for v in nodes.values()})),
                 first=min((r['year'],r['month']) for r in rows),last=max((r['year'],r['month']) for r in rows),
                 types=dict(Counter(r['type'] for r in rows)),transport=dict(Counter(r['transp'] for r in rows)))
    assert audit['zero']==115 and audit['last']==(2017,8)
    return rows,nodes,audit

def build(rows,nodes,period):
    selected = [r for r in rows if period=='2013-2017' or r['year']==2017]
    active = sorted({v for r in selected for v in (r['from'],r['to'])})
    counts,weights = Counter(),Counter()
    for r in selected:
        edge = r['from'],r['to']
        counts[edge] += 1
        weights[edge] += r['scaled']
    directed = nx.DiGraph()
    directed.add_nodes_from((v,dict(nodes[v])) for v in active)
    directed.add_edges_from((u,v,dict(movements=counts[u,v],scaled=weights[u,v])) for u,v in sorted(counts))
    graph = nx.Graph()
    graph.add_nodes_from((v,dict(nodes[v])) for v in active)
    for u,v,d in directed.edges(data=True):
        if not graph.has_edge(u,v): graph.add_edge(u,v,movements=0,scaled=Fraction(0))
        graph.edges[u,v]['movements'] += d['movements']
        graph.edges[u,v]['scaled'] += d['scaled']
    ordered = nx.Graph()
    ordered.add_nodes_from((v,dict(graph.nodes[v])) for v in sorted(graph))
    ordered.add_edges_from((u,v,dict(graph.edges[u,v])) for u,v in edges(graph))
    assert sum(d['movements'] for u,v,d in directed.edges(data=True))==len(selected)
    assert sum(d['scaled'] for u,v,d in ordered.edges(data=True))==sum(r['scaled'] for r in selected)
    return directed,ordered

def groups(graph,cuts=()):
    return sorted(components(graph,cuts),key=lambda c:(-len(c),c))

def value(graph,cuts=()):
    sizes = list(map(len,groups(graph,cuts)))
    return Fraction(len(graph)**2-sum(n*n for n in sizes),len(graph))

def named(graph,vertices):
    return [dict(id=v,name=graph.nodes[v]['name'],country=graph.nodes[v]['country']) for v in sorted(vertices)]

def directionvalue(directed,cuts=()):
    remaining = directed.copy()
    remaining.remove_edges_from((u,v) for x,y in cuts for u,v in ((x,y),(y,x)))
    reachable = sum(len(nx.descendants(remaining,v)) for v in remaining)
    return Fraction(len(remaining)*(len(remaining)-1)-reachable,len(remaining))

def describe(directed,graph,nodes):
    parts = groups(graph)
    giant = graph.subgraph(parts[0]).copy()
    scc = sorted(map(len,nx.strongly_connected_components(directed)),reverse=True)
    directedpairs = sum(len(nx.descendants(directed,v)) for v in directed)
    undirectedpairs = sum(len(c)*(len(c)-1) for c in parts)
    hubs = sorted(giant,key=lambda v:(-giant.degree(v),v))[:5]
    rest = giant.copy()
    rest.remove_nodes_from(hubs)
    hubdata = [dict(**named(giant,[v])[0],degree=giant.degree(v),indegree=directed.in_degree(v),outdegree=directed.out_degree(v)) for v in hubs]
    return giant,dict(vertices=len(graph),arcs=directed.number_of_edges(),relationships=graph.number_of_edges(),
                      movements=sum(d['movements'] for u,v,d in directed.edges(data=True)),
                      scaled=str(sum(d['scaled'] for u,v,d in directed.edges(data=True))),
                      weakcomponents=len(parts),weaksizes=list(map(len,parts)),strongcomponents=len(scc),strongsizes=dict(sorted(Counter(scc).items())),
                      largeststrong=max(scc),giant=len(giant),giantedges=giant.number_of_edges(),excluded=sorted(set(graph)-set(giant)),
                      inactive=sorted(set(nodes)-set(graph)),edgeconnectivity=nx.edge_connectivity(giant),
                      bridges=len(list(nx.bridges(giant))),leaves=sum(giant.degree(v)==1 for v in giant),
                      countries=dict(sorted(Counter(graph.nodes[v]['country'] for v in graph).items())),
                      directedpairs=directedpairs,undirectedpairs=undirectedpairs,reachfraction=str(Fraction(directedpairs,undirectedpairs)),
                      directedexpected=str(Fraction(len(graph)*(len(graph)-1)-directedpairs,len(graph))),
                      undirectedexpected=str(Fraction(len(graph)*(len(graph)-1)-undirectedpairs,len(graph))),
                      hubs=hubdata,topfiveincident=sum(u in hubs or v in hubs for u,v in giant.edges),
                      afterhubs=list(map(len,groups(rest))))

def expired(signum,frame):
    raise TimeoutError('Exact pilot wall-time cap reached')

def solve(graph,b,cap,limit):
    start = perf_counter()
    old = signal.signal(signal.SIGALRM,expired)
    result = dict(status='UNVERIFIED',error='',seconds=0,stats={})
    try:
        signal.setitimer(signal.ITIMER_REAL,cap)
        adapted = graph.copy()
        for v in adapted: adapted.nodes[v]['inbound'] = 1
        for edge in edges(adapted): adapted.edges[edge]['passengers'] = 1
        catalog = air.partitions(adapted,b,limit=limit)
        result['stats']['contraction'] = catalog['stats']
        optimal,stats = air.optimize(catalog,'uu',limit=limit)
        result['stats']['optimization'] = stats
        state = optimal[b]
        optimum = Fraction(air.numerator(catalog,'uu',state),len(graph))
        assert optimum==value(graph,state[4]) and len(state[4])<=b
        options = [o for branch in catalog['branches'] for o in branch['options'] if len(o['groups'])==1 and o['cost']<=b]
        single = min(options,key=lambda o:(-2*o['n']*(len(graph)-o['n']),o['cost'],o['cuts'])) if options else None
        singlevalue = Fraction(2*single['n']*(len(graph)-single['n']),len(graph)) if single else None
        if single: assert len(groups(graph,single['cuts']))==2 and value(graph,single['cuts'])==singlevalue
        result.update(status='verified',value=optimum,cuts=state[4],single=singlevalue,singlecuts=single['cuts'] if single else (),
                      lower={j:Fraction(air.numerator(catalog,'uu',s),len(graph)) for j,s in optimal.items()})
    except (TimeoutError,RuntimeError) as error: result['error'] = str(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old)
    result['seconds'] = perf_counter()-start
    return result

def linkdata(graph,directed,cuts,ranks):
    bridges = set(tuple(sorted(e)) for e in nx.bridges(graph))
    result = []
    for edge in cuts:
        u,v = edge
        directions = [dict(source=x,target=y,movements=directed.edges[x,y]['movements'],scaled=str(directed.edges[x,y]['scaled']))
                      for x,y in ((u,v),(v,u)) if directed.has_edge(x,y)]
        result.append(dict(edge=edge,endpoints=named(graph,edge),international=graph.nodes[u]['country']!=graph.nodes[v]['country'],
                           movements=graph.edges[edge]['movements'],scaled=str(graph.edges[edge]['scaled']),
                           bcrank=ranks.get(edge),bridge=edge in bridges,directions=directions))
    return result

def record(period,graph,directed,b,method,cuts,status,seconds,reference,ranks,steps=(),error=''):
    cuts = tuple(map(tuple,cuts))
    assert len(cuts)<=b and len(set(cuts))==len(cuts) and all(graph.has_edge(*e) for e in cuts)
    partition = groups(graph,cuts)
    actual = value(graph,cuts)
    complete = status in ('verified','complete')
    p,a = {v:Fraction(1,len(graph)) for v in graph},{v:1 for v in graph}
    assert score(graph,cuts,p,a)==actual
    exact = reference.get('value')
    if exact is not None: assert actual<=exact
    regions = []
    for c in partition[1:]:
        boundary = tuple(e for e in edges(graph) if (e[0] in c)!=(e[1] in c))
        assert set(boundary)<=set(cuts)
        regions.append(dict(size=len(c),singleton=len(c)==1,markets=named(graph,c),connected=nx.is_connected(graph.subgraph(c)),
                            internaledges=graph.subgraph(c).number_of_edges(),internallinks=edges(graph.subgraph(c)),
                            countries=dict(Counter(graph.nodes[v]['country'] for v in c)),boundary=linkdata(graph,directed,boundary,ranks),
                            boundaryscaled=str(sum(graph.edges[e]['scaled'] for e in boundary))))
    restricted = directed.subgraph(graph).copy()
    directedbefore,directedafter = directionvalue(restricted),directionvalue(restricted,cuts)
    return dict(period=period,n=len(graph),m=graph.number_of_edges(),budget=b,method=method,status=status,error=error,
                cuts=cuts,removed=linkdata(graph,directed,cuts,ranks),score=str(actual) if complete else '',scoredecimal=float(actual) if complete else '',
                ratio=str(actual/exact) if complete and exact else '',ratiodecimal=float(actual/exact) if complete and exact else '',
                exactstatus=reference['status'],exact=str(exact) if exact is not None else '',
                directedbefore=str(directedbefore),directedafter=str(directedafter),directedgain=str(directedafter-directedbefore),
                components=list(map(len,partition)),partition=partition,regions=regions,partialscore='' if complete else str(actual),
                scaledboundary=str(sum(graph.edges[e]['scaled'] for e in cuts)),movements=sum(graph.edges[e]['movements'] for e in cuts),
                seconds=seconds,stepseconds=[s['seconds'] for s in steps],gains=[s['gain'] for s in steps],
                greedyties=[s['ties'] for s in steps] if method=='greedy' else [],
                zerogainties=[s['ties'] for s in steps if Fraction(s['gain'])==0] if method=='greedy' else [],
                steps=steps,searchstats=reference['stats'] if method=='exact' else {},
                bestsingle=str(reference['single']) if method=='exact' and reference.get('single') is not None else '',
                singlecuts=reference.get('singlecuts',()) if method=='exact' else (),
                graphhash=baseline.signature(graph))

def regressions():
    cases,checks = baseline.regressions()
    fixtures = [nx.path_graph(6),nx.cycle_graph(5),nx.complete_graph(5),nx.barbell_graph(3,1)]
    fixtures += [case[1] for case in cases]
    count = 0
    for graph in fixtures:
        B = 2 if len(graph)>10 else 3
        exact = solve(graph,B,10,200000)
        assert exact['status']=='verified',exact
        for b in range(B+1):
            direct = baseline.exhaustive(graph,b)
            assert exact['lower'][b]==direct['value'],(b,exact,direct)
            count += 1
    checks.append(f'PASS existing contraction/DP versus independent exhaustive source averaging: {count} small graph/budget cases')
    return checks

def replay(rows,graphs):
    checked = 0
    for row in rows:
        graph,directed = graphs[row['period']]
        b = int(row['budget'])
        cuts = tuple(map(tuple,json.loads(row['cuts'])))
        assert len(cuts)<=b and baseline.signature(graph)==row['graphhash']
        assert list(map(len,groups(graph,cuts)))==json.loads(row['components'])
        assert list(map(list,groups(graph,cuts)))==json.loads(row['partition'])
        if row['score']:
            assert value(graph,cuts)==Fraction(row['score'])
            if row['exact']: assert Fraction(row['score'])<=Fraction(row['exact'])
            if row['ratio']: assert Fraction(row['ratio'])==Fraction(row['score'])/Fraction(row['exact'])
        assert sum(graph.edges[e]['scaled'] for e in cuts)==Fraction(row['scaledboundary'])
        assert directionvalue(directed,cuts)==Fraction(row['directedafter'])
        assert Fraction(row['directedafter'])-Fraction(row['directedbefore'])==Fraction(row['directedgain'])
        for region in json.loads(row['regions']):
            vertices = [v['id'] for v in region['markets']]
            assert region['markets']==named(graph,vertices) and len(vertices)==region['size']
            assert nx.is_connected(graph.subgraph(vertices))
            expected = {e for e in edges(graph) if (e[0] in vertices)!=(e[1] in vertices)}
            assert expected=={tuple(d['edge']) for d in region['boundary']}
        if row['method']=='greedy':
            remaining = graph.copy()
            for step in json.loads(row['steps']):
                gains = baseline.marginals(remaining)
                best = max(gains.values())
                ties = sorted(e for e in gains if gains[e]==best)
                assert tuple(step['edge'])==ties[0] and step['ties']==len(ties)
                assert Fraction(step['gain'])==Fraction(best,len(graph))
                remaining.remove_edge(*ties[0])
        checked += 1
    return checked

def savedchecks(rows,graphs):
    bykey = {(r['period'],int(r['budget']),r['method']):r for r in rows}
    for period,sizes in [('2017',[[179,4],[177,4,2],[173,10]]),('2013-2017',[[257,5],[255,5,2],[251,11]])]:
        graph,directed = graphs[period]
        for b,parts in enumerate(sizes,1):
            row = bykey[period,b,'exact']
            if row['status']!='verified': continue
            assert json.loads(row['components'])==parts
            expected = Fraction(len(graph)**2-sum(n*n for n in parts),len(graph))
            assert Fraction(row['score'])==expected
        for method in ('initial','recomputed'):
            for b in (1,2,3):
                row = bykey[period,b,method]
                if row['status']=='complete': assert Fraction(row['score'])==0 and json.loads(row['components'])==[len(graph)]
        row = bykey[period,3,'exact']
        if row['status']=='verified':
            assert json.loads(row['cuts'])==[['n002','n215'],['n013','n215'],['n117','n215']]
            for e in json.loads(row['cuts']): assert nx.is_connected(nx.restricted_view(graph,[],[tuple(e)]))
        for b in (1,2):
            row = bykey[period,b,'greedy']
            if row['status']=='complete' and row['exact']: assert Fraction(row['score'])==Fraction(row['exact'])
    return 'PASS period-specific saved optima, joint three-link regional boundary, zero BC values and optimal greedy prefixes'

def report(records,audit,diagnostics,nodes,checks,elapsed,cap,limit,ranktimes):
    lines = ['WEST AFRICAN LIVESTOCK TRADE: STATIC FEASIBILITY PILOT','',
             'SOURCE AND RECONSTRUCTION',article,
             'Valerio et al. (2020), PLOS ONE 15(5): e0232681. Official S2 edge and S3 node supplements retrieved 2026-11-16.',
             'Data collection covers January 2013 through August 2017; the 2017 network is January-August, not a full calendar year.',
             'Each CSV movement is already a preprocessed origin-survey or survey-destination leg. Do not split again or deduplicate identical rows.',
             'Edge schema: from,to,weight,type,transp,dmonth,intl,fromc,toc. Node schema: id,name,country,cid.',
             'dmonth parses as month/1/year; records are month-labelled, not exact temporal paths. All four livestock types and all transport types are included.',
             'weight is a scaled livestock quantity, not an exact animal count, money, or intervention cost. Decimal strings are summed as exact Fractions.',
             '115 zero-weight records are retained as observed movements. Edge existence uses record presence, not a positive-weight filter.',
             'All market IDs and country codes match exactly; no self-loops, missing fields, or unmatched endpoints. Raw files preserve directions and original weight strings.',
             'Annual vertices are observed endpoints in that period. S3 is an all-period dictionary, not 262 active annual vertices. No inactive metadata-only vertices are added.',
             'Directed arcs aggregate each ordered market pair; undirected relationships aggregate both directions and retain a link if either direction occurs.',
             'The 42,252 records are processed movements, not distinct relationships. No new regional sampling or connectivity threshold is imposed.',
             'Audit: '+json.dumps(audit,sort_keys=True),
             'Inputs:']
    for name,(url,expected) in sources.items(): lines += [str((raw/name).relative_to(root)),url,'SHA256 '+expected]
    lines += ['',f'Python {platform.python_version()}; NetworkX {nx.__version__}',
              'Source hashes: '+json.dumps({p:digest(root/p) for p in ['code/air.py','code/hospital.py','code/baselines.py','code/livestock.py']},sort_keys=True),
              '', 'GRAPH AND DIRECTION DIAGNOSTICS']
    for period,d in diagnostics.items():
        lines += [period+': '+json.dumps(d,sort_keys=True),
                  f"Directed reachability: {d['directedpairs']}/{d['undirectedpairs']} undirected-reachable ordered distinct pairs ({100*float(Fraction(d['reachfraction'])):.3f}%).",
                  f"No-cut expected unreachable markets: directed={d['directedexpected']}; undirected={d['undirectedexpected']}. These are different objectives; the pilot optimizes only undirected static protection.",
                  f"Top five hubs touch {d['topfiveincident']}/{d['giantedges']} giant edges ({100*d['topfiveincident']/d['giantedges']:.2f}%). Their joint deletion leaves components {d['afterhubs']}; this is a diagnostic, not a tested intervention policy."]
    lines += ['', 'MODEL AND EXACTNESS',
              'Select the largest undirected component once; keep that entire vertex set after all removals. All active markets belong to the giant in both periods, so no active vertices are excluded.',
              'Uniform source over giant vertices; unit target per market. Q(F)=(n^2-sum(component sizes squared))/n. No cuts gives Q=0. Static cuts precede transmission; no sequential simulator is used.',
              'A detached component is protected when the source lies elsewhere; it is not guaranteed safe for every source. Q is expected protected markets, not detached-region size.',
              'Reuse air.partitions and air.optimize(mode=uu). An adapter sets inbound=1 and passengers=1 solely for the existing solver fields; neither represents livestock population or scaled shipment volume.',
              'Completeness: capacity-one Gomory-Hu contraction merges vertices inseparable by <=b edges. Every feasible final component is a union of atoms.',
              'Restoring removals internal to final components preserves Q and lowers cost. Thus every relevant cut removes whole inter-atom bundles.',
              'The solver exhaustively enumerates local bundle subsets within budget in every branch around the largest atom, then combines branch partitions by its exact DP. All root components merge; detached sizes and their squared-size sum determine Q.',
              'For uniform/unit targets, DP dominance preserves every optimum because equal detached total size has the same final root size and a smaller squared-size sum is never worse. This is not a search over single Gomory-Hu edges.',
              'Best single-region comparison: a two-component global partition has one detached component in one quotient branch. Enumerating completed local options with exactly one detached component gives the exact best two-component value under the same budget.',
              f'Feasibility gate: reconstruct both graphs and solve budget 1 first, capped at min(20,{cap}) seconds each. Continue only if both connected giants have at least 3 vertices and both pilot optima are verified.',
              f'Budgets 2 and 3 each have a {cap}-second wall cap and {limit} local-enumeration/DP-combination limits. No failed search supplies an OPT or approximation ratio.',
              'BC policy code is imported from code/baselines.py. Ordered-pair BC=2*NetworkX normalized=False; canonical string ID pairs break equal computed scores lexicographically.',
              'Initial BC ranks once; recomputed BC ranks after each previous deletion. Each BC step has a 20-second cap and each graph a 60-second cumulative recomputed cap.',
              'Per-step BC/greedy selection times and cumulative selection times are recorded; replay and diagnostic time is included only in total pilot runtime. The first BC computation is shared by initial and recomputed policies.',
              'Objective Greedy uses exact integer marginal numerators: a bridge splitting a current component into x and y gives 2xy/n; all non-bridges give zero. Zero-gain ties are recorded and resolved lexicographically.',
              'BC rank audit seconds (separate original-graph calculation): '+json.dumps(ranktimes,sort_keys=True),
              '', 'RESULTS', 'period | budget | method | status | Q | BC or greedy / OPT | component sizes | cuts | seconds | greedy ties / zero-gain ties']
    for r in records:
        lines.append(f"{r['period']} | {r['budget']} | {r['method']} | {r['status']} | {r['score']} ({r['scoredecimal']}) | {r['ratio']} | {r['components']} | {r['cuts']} | {r['seconds']:.6f} | {r['greedyties']} / {r['zerogainties']}")
    lines += ['', 'EXACT CUTS AND REGIONAL INTERPRETATION']
    for r in records:
        if r['method']!='exact': continue
        lines += [f"{r['period']}, b={r['budget']}: {r['status']}; Q={r['score']}; components={r['components']}; scaled boundary total={r['scaledboundary']}"]
        if r['status']!='verified':
            lines.append('UNVERIFIED: '+r['error'])
            continue
        lines += ['Complete removed links: '+json.dumps(r['removed'],sort_keys=True),
                  'Non-giant components: '+json.dumps(r['regions'],sort_keys=True),
                  f"Best two-component cut: Q={r['bestsingle']}; cut={r['singlecuts']}",
                  f"Direction-only diagnostic for this same relationship cut (not directed optimization): Qdirected(empty)={r['directedbefore']}, Qdirected(cut)={r['directedafter']}, incremental gain={r['directedgain']}.",
                  'Search completion: '+json.dumps(r['searchstats'],sort_keys=True)]
        if r['bestsingle'] and Fraction(r['score'])>Fraction(r['bestsingle']):
            lines.append('Multiple detached components strictly outperform every feasible two-component cut at this budget.')
        else: lines.append('No strict advantage over the best two-component cut is established at this budget.')
    lines += ['', 'LIMITATIONS',
              'The undirected projection introduces reverse traversals not supported by recorded shipments and has much greater reachability. Directed no-cut isolation is not an intervention effect.',
              'Static aggregation also allows paths whose shipment ordering is impossible in time; combining 2013-2017 is not an observed simultaneous network.',
              'Survey coverage is selected, concentrated on Burkina Faso-linked corridors, changes across years, and omits incomplete or unmapped movements. The article does not claim a complete West African trade census.',
              'Market names and countries identify recorded locations, not verified geographic compactness, common administration, or feasible closure plans. The node supplement supplies no coordinates.',
              'Scaled boundary sums are descriptive only. No exact animal counts, disease probabilities, costs, welfare losses, or actual containment benefits are inferred.',
              'Only one deterministic exact representative is reported per budget. No claim of unique optimum or exhaustive co-optimal enumeration is made.',
              '', 'FEASIBILITY RECOMMENDATION',
              'Interpret the computed comparisons below as a test of a possible additional application, not confirmation of the existing paper narrative.']
    lines += findings(records,diagnostics)
    lines += ['', 'MARKET DICTIONARY (official IDs; names and countries preserved)']
    lines.extend(v+' | '+d['name']+' | '+d['country']+' | '+d['cid'] for v,d in sorted(nodes.items()))
    lines += ['', 'VALIDATION']+checks+[f'Pilot runtime: {elapsed:.6f} seconds (downloads excluded).',
              'Reproduce: PYTHONDONTWRITEBYTECODE=1 python code/livestock.py',
              'Validate: PYTHONDONTWRITEBYTECODE=1 python code/livestock.py --check',
              'Fetch missing official inputs: PYTHONDONTWRITEBYTECODE=1 python code/livestock.py --fetch',
              'Generation refuses to overwrite data/results/livestock.csv or the livestock report entry in data/records.json. Use a separate reproduction copy and remove only those copied outputs to rerun.',
              'No existing application, source, figure, manuscript, or saved result was modified. No other application was rerun.']
    return '\n'.join(lines)+'\n'

def findings(records,diagnostics):
    lines = []
    for period in periods:
        selected = [r for r in records if r['period']==period]
        for b in (1,2,3):
            rows = {r['method']:r for r in selected if r['budget']==b}
            exact = rows['exact']
            if exact['status']!='verified':
                lines.append(f'{period} b={b}: exact optimum UNVERIFIED.')
                continue
            detail = '; '.join(f'{method} Q={rows[method]["score"]}, ratio={rows[method]["ratio"]}' for method in ('initial','recomputed','greedy'))
            lines.append(f'{period} b={b}: OPT={exact["score"]}; detached sizes={exact["components"][1:]}; {detail}.')
    lookup = {(r['period'],r['budget'],r['method']):r for r in records}
    if not all(r['status'] in ('verified','complete') for r in records):
        return lines+['Recommendation: EXPAND PILOT. Incomplete computations prevent a complete six-case comparison; no missing optimum or ratio has been imputed.']
    lines += [
        '1. Coordinated-cut advantage: yes in both observed periods at b=3. The three removed relationships are Fada N\'Gourma--Kouremale, Bobo-Dioulasso--Kouremale and Ouagadougou--Kouremale. Each edge alone is non-separating. Initial and recomputed BC achieve zero through b=3.',
        '2. Mechanism: a multi-link gateway feeds a downstream branching market region. All three boundary links terminate at Kouremale, and the recorded movement direction on each is Burkina Faso to Mali. This resembles the existing coordinated-boundary mechanism, not a demonstrated new universal graph mechanism.',
        'The additional contrast is between fragmentation and regional isolation: b=2 optimally separates two smaller branch regions, whereas b=3 replaces those cuts with the entire three-link gateway. Objective Greedy commits to immediate bridge gains and cannot replace its earlier choices.',
        '3. Interpretation: the separated connected region spans identified markets in Mali and Guinea. Names/countries are supplied, but geographic compactness and operational feasibility are not established by the CSVs.',
        '4. Period comparison: the representative b=3 boundary is identical across both periods. The full-period region adds Diankana (n185, Guinea) to the 10-market 2017 region. The same two bridge cuts are selected at b=2. These periods overlap and differ in survey coverage; this is consistency under aggregation, not an independent replication.',
        '5. Direction: only about one tenth of distinct ordered pairs are reachable in the directed networks. Undirected projection therefore makes a major interpretive change. The recorded b=3 boundaries are one-way inputs; directed no-cut protection is already large. Direction-aware intervention analysis is needed before making disease-control claims.',
        '6. Sparsity: both evaluated networks are connected, so baseline fragmentation is not the explanation. There are many bridges and leaves, but the b=1 optimum already detaches 4 or 5 markets, b=2 detaches two multi-market components, and b=3 detaches 10 or 11. The exact results are not merely singleton peeling.',
        'Greedy zero-gain ties: none in these measured runs. Its third choice has many tied POSITIVE gains: '+', '.join(p+'='+str(lookup[p,3,'greedy']['greedyties'][-1]) for p in periods)+'.',
        'BC zero values are verified connected post-cut graphs, not failed computations. High-BC selections do not complete a separating boundary within the tested budget.',
        'Recommendation: EXPAND PILOT. The undirected result is reproducible and interpretable, but it largely repeats a coordinated-cut mechanism while direction and incomplete survey coverage are major limitations. Next assess directed and temporally feasible containment before deciding on a fifth application.']
    return lines

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--fetch',action='store_true')
    mode.add_argument('--pilot',action='store_true')
    parser.add_argument('--timecap',type=float,default=45)
    parser.add_argument('--limit',type=int,default=200000)
    args = parser.parse_args()
    if args.timecap<=0 or args.limit<1: parser.error('Positive time and search limits required')
    if args.fetch:
        fetch()
        return
    if not args.check and not args.pilot and ((folder/'livestock.csv').exists() or savedreports.exists('livestock')):
        raise FileExistsError('Refusing to overwrite saved pilot results; use --check')
    start = perf_counter()
    protected = snapshot()
    rows,nodes,audit = read()
    graphs,diagnostics = {},{}
    for period in periods:
        directed,full = build(rows,nodes,period)
        again,other = build(list(reversed(rows)),nodes,period)
        assert nx.utils.graphs_equal(directed,again) and nx.utils.graphs_equal(full,other)
        graph,diagnostics[period] = describe(directed,full,nodes)
        assert value(graph)==0 and nx.is_connected(graph)
        graphs[period] = graph,directed
        print(f'{period}: active markets={len(full)}, arcs={directed.number_of_edges()}, undirected edges={full.number_of_edges()}, giant={len(graph)}, bridges={diagnostics[period]["bridges"]}',flush=True)
    assert (len(graphs['2017'][0]),graphs['2017'][0].number_of_edges())==(183,285)
    assert (len(graphs['2013-2017'][0]),graphs['2013-2017'][0].number_of_edges())==(262,578)
    checks = regressions()
    checks += ['PASS official CSV hashes, exact schemas, all 42252 movement records and 262 market-country joins',
               'PASS deterministic reversed-input reconstruction, summed movement counts and scaled weights, connected giants and zero no-cut objective']
    for line in checks: print(line,flush=True)
    if args.check:
        if (folder/'livestock.csv').exists():
            with (folder/'livestock.csv').open(newline='') as stream: saved = list(csv.DictReader(stream))
            assert len(saved)==24 and len({(r['period'],r['budget'],r['method']) for r in saved})==24
            print(f'PASS {replay(saved,graphs)} saved cut/partition/score/region replays and greedy tie sequences',flush=True)
            print(savedchecks(saved,graphs),flush=True)
        assert protected==snapshot()
        print(f'PASS existing project files unchanged; checks completed in {perf_counter()-start:.3f}s')
        return
    exact = {}
    for period,(graph,directed) in graphs.items():
        exact[period,1] = solve(graph,1,min(20,args.timecap),args.limit)
        print(f'Pilot {period} b=1: {exact[period,1]["status"]}, {exact[period,1].get("value")}, {exact[period,1]["seconds"]:.4f}s',flush=True)
    if args.pilot:
        assert protected==snapshot()
        return
    feasible = all(exact[p,1]['status']=='verified' and len(graphs[p][0])>=3 for p in periods)
    records,ranktimes = [],{}
    for period,(graph,directed) in graphs.items():
        for b in (2,3):
            exact[period,b] = solve(graph,b,args.timecap,args.limit) if feasible else dict(status='UNVERIFIED',error='Feasibility gate did not pass',seconds=0,stats={})
            for j in range(1,b):
                if exact[period,b]['status']==exact[period,j]['status']=='verified': assert exact[period,b]['lower'][j]==exact[period,j]['value']
            print(f'{period} b={b}: {exact[period,b]["status"]}, {exact[period,b].get("value")}, {exact[period,b]["seconds"]:.4f}s',flush=True)
        ranking,rankseconds,rankerror = baseline.limited(lambda:baseline.bcorder(graph),20)
        ranks = {e:i+1 for i,e in enumerate(ranking[0])} if ranking else {}
        ranktimes[period] = dict(seconds=rankseconds,error=rankerror)
        traces = baseline.policies(graph,3,20,60,period) if feasible else {}
        for b in (1,2,3):
            ref = exact[period,b]
            records.append(record(period,graph,directed,b,'exact',ref.get('cuts',()),ref['status'],ref['seconds'],ref,ranks,error=ref['error']))
            for method in ('initial','recomputed','greedy'):
                trace = traces.get(method,dict(steps=[],error='Feasibility gate did not pass'))
                steps = trace['steps'][:b]
                complete = len(steps)==b
                records.append(record(period,graph,directed,b,method,[s['edge'] for s in steps],'complete' if complete else 'incomplete',
                                      sum(s['seconds'] for s in steps),ref,ranks,steps,'' if complete else trace['error']))
    assert len(records)==24
    checks += ['PASS all completed policy scores <= verified OPT; exact DP scores independently replayed via component squares and the general prior/target identity',
               'PASS earlier-budget exact values agree across completed higher-budget catalogs',
               'PASS existing project and source data fingerprints unchanged']
    assert protected==snapshot()
    text = report(records,audit,diagnostics,nodes,checks,perf_counter()-start,args.timecap,args.limit,ranktimes)
    with (folder/'livestock.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        for r in records: writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
    savedreports.save('livestock',text,exclusive=True)
    for line in findings(records,diagnostics): print(line,flush=True)
    print(f'Wrote 24 records; runtime {perf_counter()-start:.3f}s',flush=True)

destination = folder/'directed.csv'

names = dict(kouremale='Coordinated Kouremale',initial='Initial undirected BC',recomputed='Recomputed undirected BC',
             greedy='Undirected objective greedy',directedgreedy='Directed objective greedy',directedbc='Recomputed directed BC')

def directedsnapshot():
    excluded = {root/'code/livestock.py',destination}
    return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and p not in excluded
            and not any(v in p.relative_to(root).parts for v in ('.git','.venv','__pycache__'))}

def relationships(graph):
    return tuple(sorted({tuple(sorted(e)) for e in graph.edges}))

def remove(graph,edge):
    u,v = edge
    if not graph.has_edge(u,v) and not graph.has_edge(v,u): raise ValueError('Relationship is absent')
    graph.remove_edges_from(((u,v),(v,u)))

def remaining(graph,cuts):
    result = graph.copy()
    for edge in cuts: remove(result,edge)
    assert set(result)==set(graph)
    return result

def reachable(graph):
    adjacency = {v:tuple(graph.successors(v)) for v in graph}
    total = 0
    for source in graph:
        seen,queue = {source},[source]
        for u in queue:
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        total += len(seen)-1
    return total

def objective(graph,pairs=None):
    if pairs is None: pairs = reachable(graph)
    n = len(graph)
    return Fraction(n*(n-1)-pairs,n)

def reference(graph):
    vertices = sorted(graph)
    table = [[u==v or graph.has_edge(u,v) for v in vertices] for u in vertices]
    for k in range(len(vertices)):
        for i in range(len(vertices)):
            for j in range(len(vertices)): table[i][j] = table[i][j] or (table[i][k] and table[k][j])
    return sum(map(sum,table))-len(vertices)

def greedychoice(graph):
    original = reachable(graph)
    best,tied = -1,[]
    for edge in relationships(graph):
        gain = original-reachable(remaining(graph,(edge,)))
        if gain>best: best,tied = gain,[edge]
        elif gain==best: tied.append(edge)
    return tied[0],dict(metric='eliminated ordered pairs',value=best,ties=len(tied),maximizers=tied,
                        candidates=len(relationships(graph)))

def bcchoice(graph):
    arcs = nx.edge_betweenness_centrality(graph,normalized=False,weight=None)
    values = {e:arcs.get(e,0)+arcs.get((e[1],e[0]),0) for e in relationships(graph)}
    best = max(values.values())
    tied = sorted(e for e in values if values[e]==best)
    return tied[0],dict(metric='sum of directed arc BC',value=best,ties=len(tied),maximizers=tied,
                        candidates=len(values))

def directedexpired(signum,frame):
    raise TimeoutError('Directed policy time cap reached')

def limited(action,cap):
    start = perf_counter()
    if cap<=0: return None,0.0,'Cumulative policy time cap reached'
    previous = signal.signal(signal.SIGALRM,directedexpired)
    try:
        signal.setitimer(signal.ITIMER_REAL,cap)
        return action(),perf_counter()-start,''
    except TimeoutError as error:
        return None,perf_counter()-start,str(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)

def policy(graph,method,b=3,stepcap=30,totalcap=60):
    if method not in ('directedgreedy','directedbc'): raise ValueError('Unknown policy')
    work = graph.copy()
    steps,cuts,attempts,spent = [],[],[],0.0
    for j in range(min(b,len(relationships(graph)))):
        choice,seconds,error = limited(lambda:(greedychoice(work) if method=='directedgreedy' else bcchoice(work)),min(stepcap,totalcap-spent))
        spent += seconds
        attempts.append(dict(round=j+1,seconds=seconds,status='incomplete' if error else 'complete',error=error))
        if error: return dict(status='incomplete',cuts=cuts,steps=steps,attempts=attempts,seconds=spent,error=error)
        edge,detail = choice
        before = reachable(work)
        remove(work,edge)
        after = reachable(work)
        assert after<=before
        if method=='directedgreedy': assert before-after==detail['value']
        steps.append(dict(round=j+1,edge=edge,pairsbefore=before,pairsafter=after,pairslost=before-after,
                          gain=str(Fraction(before-after,len(graph))),seconds=seconds,**detail))
        cuts.append(edge)
    return dict(status='complete',cuts=cuts,steps=steps,attempts=attempts,seconds=spent,error='')

def details(graph,cuts):
    result = []
    for u,v in cuts:
        arcs = [dict(source=x,target=y,movements=graph.edges[x,y]['movements'],scaled=str(graph.edges[x,y]['scaled']))
                for x,y in ((u,v),(v,u)) if graph.has_edge(x,y)]
        result.append(dict(pair=(u,v),markets=named(graph,(u,v)),international=graph.nodes[u]['country']!=graph.nodes[v]['country'],
                           arcs=arcs,scaled=str(sum(Fraction(d['scaled']) for d in arcs))))
    return result

def provenance():
    paths = [root/'code/livestock.py',folder/'livestock.csv',root/'code/livestock.py']+[raw/name for name in sources]
    hashes = {str(p.relative_to(root)):digest(p) for p in paths}
    hashes['report:livestock'] = hashlib.sha256(savedreports.text('livestock').encode()).hexdigest()
    return hashes

def directedrecord(period,graph,method,result,base,hashes,stepcap,totalcap):
    start = perf_counter()
    cuts = tuple(map(tuple,result['cuts']))
    assert len(cuts)<=3 and len(set(cuts))==len(cuts) and all(tuple(sorted(e))==e for e in cuts)
    after = remaining(graph,cuts)
    pairs = reachable(after)
    beforevalue,aftervalue = objective(graph,base),objective(graph,pairs)
    assert aftervalue==directionvalue(graph,cuts)
    assert len(after)==len(graph)
    gain = aftervalue-beforevalue
    lost = base-pairs
    assert gain==Fraction(lost,len(graph)) and gain>=0
    complete = result['status']=='complete'
    percent = Fraction(100*lost,base) if base else None
    return dict(period=period,method=method,label=names[method],budget=3,status=result['status'],error=result['error'],
                n=len(graph),arcs=graph.number_of_edges(),relationships=len(relationships(graph)),cuts=cuts,removed=details(graph,cuts),
                baselinepairs=base,baselinewithself=base+len(graph),remainingpairs=pairs,remainingwithself=pairs+len(graph),
                eliminated=lost,percent=str(percent) if percent is not None else '',percentdecimal=float(percent) if percent is not None else '',
                qbefore=str(beforevalue),qafter=str(aftervalue),gain=str(gain) if complete else '',gaindecimal=float(gain) if complete else '',
                partialgain='' if complete else str(gain),directedoptimality='not established',
                weakcomponents=sorted(map(len,nx.weakly_connected_components(after)),reverse=True),
                strongcomponents=sorted(map(len,nx.strongly_connected_components(after)),reverse=True),
                selectionseconds=result['seconds'],evaluationseconds=perf_counter()-start,steps=result['steps'],attempts=result['attempts'],
                stepcap=stepcap,totalcap=totalcap,ties=[s['ties'] for s in result['steps'] if 'ties' in s],
                python=platform.python_version(),networkx=nx.__version__,provenance=hashes)

def directedregressions():
    count = 0
    arcs = tuple(permutations(range(3),2))
    for mask in range(1<<len(arcs)):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(3))
        graph.add_edges_from(e for i,e in enumerate(arcs) if mask & (1<<i))
        before = graph.copy()
        assert reachable(graph)==reference(graph)
        assert objective(graph)==Fraction(sum(3-len(nx.descendants(graph,s))-1 for s in graph),3)
        if relationships(graph):
            result = policy(graph,'directedgreedy',min(2,len(relationships(graph))))
            assert result['status']=='complete'
            work = graph.copy()
            for step in result['steps']:
                values = {e:reference(work)-reference(remaining(work,[e])) for e in relationships(work)}
                largest = max(values.values())
                tied = sorted(e for e in values if values[e]==largest)
                assert step['edge']==tied[0] and step['value']==largest and step['ties']==len(tied)
                remove(work,step['edge'])
                count += 1
        assert nx.utils.graphs_equal(graph,before)
    graph = nx.DiGraph([(0,1),(1,0)])
    edge,detail = bcchoice(graph)
    assert edge==(0,1) and detail['value']==2
    after = remaining(graph,[edge])
    assert after.number_of_edges()==0 and len(after)==2 and objective(after)-objective(graph)==1
    graph = nx.DiGraph([(0,1),(1,2)])
    edge,detail = bcchoice(graph)
    assert edge==(0,1) and detail['value']==2 and detail['ties']==2
    assert objective(graph)==1 and objective(remaining(graph,[edge]))-objective(graph)==Fraction(2,3)
    graph = nx.path_graph(4).to_directed()
    assert bcchoice(graph)[0]==(1,2) and bcchoice(graph)[1]['value']==8
    assert reachable(graph)==12 and objective(graph)==0
    triangle = nx.complete_graph(3).to_directed()
    edge,detail = greedychoice(triangle)
    assert edge==(0,1) and detail['value']==0 and detail['ties']==3
    answer,seconds,error = limited(lambda:signal.raise_signal(signal.SIGALRM),1)
    assert answer is None and error
    assert limited(lambda:None,0)[2]
    original = bcchoice
    def interruption(work):
        if work.number_of_edges()<graph.number_of_edges(): raise TimeoutError('Regression interruption')
        return original(work)
    with patch.dict(policy.__globals__,bcchoice=interruption):
        result = policy(graph,'directedbc',3)
    assert result['status']=='incomplete' and len(result['cuts'])==1 and len(result['attempts'])==2
    return [f'PASS all 64 three-vertex digraphs: independent transitive-closure counts and {count} greedy choice/tie checks',
            'PASS self reachability, reciprocal-arc pair removal, directed ordered-pair BC without factor two, lexicographic ties and zero-gain ties',
            'PASS immutable inputs, per-step/cumulative caps, and honest incomplete-prefix handling']

def load():
    rows,nodes,audit = read()
    with (folder/'livestock.csv').open(newline='') as stream:
        saved = {(r['period'],r['method']):r for r in csv.DictReader(stream) if r['budget']=='3'}
    text = savedreports.text('livestock')
    assert '36/61' in text and '117/131' in text
    graphs = {}
    for period in periods:
        directed,graph = build(rows,nodes,period)
        assert nx.is_connected(graph) and set(directed)==set(graph)
        expected = (183,298,285,3221) if period=='2017' else (262,607,578,6825)
        assert (len(directed),directed.number_of_edges(),len(relationships(directed)),reachable(directed))==expected
        graphs[period] = directed
        for method in ('exact','initial','recomputed','greedy'):
            row = saved[period,method]
            cuts = tuple(map(tuple,json.loads(row['cuts'])))
            assert row['status'] in ('verified','complete') and len(cuts)==3
            assert baseline.signature(graph)==row['graphhash']
            assert objective(directed)==Fraction(row['directedbefore'])
            assert objective(remaining(directed,cuts))==Fraction(row['directedafter'])
            assert objective(remaining(directed,cuts))-objective(directed)==Fraction(row['directedgain'])
        gain = Fraction(saved[period,'exact']['directedgain'])
        assert gain==(Fraction(36,61) if period=='2017' else Fraction(117,131))
    return graphs,saved

def directedreplay(records,graphs):
    assert len(records)==12 and len({(r['period'],r['method']) for r in records})==12
    for row in records:
        graph = graphs[row['period']]
        cuts = tuple(map(tuple,json.loads(row['cuts'])))
        base,after = reachable(graph),reachable(remaining(graph,cuts))
        assert int(row['baselinepairs'])==base and int(row['remainingpairs'])==after
        assert int(row['eliminated'])==base-after
        assert Fraction(row['qbefore'])==objective(graph,base) and Fraction(row['qafter'])==objective(graph,after)
        gain = Fraction(base-after,len(graph))
        assert Fraction(row['gain'] or row['partialgain'])==gain
        assert Fraction(row['percent'])==Fraction(100*(base-after),base)
        assert int(row['baselinewithself'])==base+len(graph)
        assert row['directedoptimality']=='not established'
        assert json.loads(row['removed'])==json.loads(json.dumps(details(graph,cuts)))
        work = graph.copy()
        for step in json.loads(row['steps']):
            old = reachable(work)
            remove(work,tuple(step['edge']))
            new = reachable(work)
            assert (step['pairsbefore'],step['pairsafter'],step['pairslost'])==(old,new,old-new)
            assert Fraction(step['gain'])==Fraction(old-new,len(graph))
        for path,digest in json.loads(row['provenance']).items(): savedreports.verify(path,digest)
    return 'PASS all 12 saved directed gains, pair-loss percentages, market metadata, step traces and source hashes'

def summary(records,graphs):
    print('DIRECTED LIVESTOCK ASSESSMENT: b=3 unordered relationships; each removes all available arcs in both directions.')
    print('Reach includes its source. Pair counts below exclude self pairs: Q_D=n-1-R/n; Gain_D=(R0-R)/n.')
    print('Directed BC is unnormalized and already ordered-pair: sum reciprocal arc scores, without multiplying by 2.')
    superior = []
    for period,graph in graphs.items():
        base = reachable(graph)
        print(f'\n{period}: n={len(graph)}, arcs={graph.number_of_edges()}, relationships={len(relationships(graph))}, baseline reachable pairs={base} ({base+len(graph)} including self), Q_D(empty)={objective(graph)}')
        print('Method | gain | eliminated pairs / baseline | percent | selection seconds')
        for row in records:
            if row['period']!=period: continue
            gain = f"{float(Fraction(row['gain'])):.6f} ({row['gain']})" if row['gain'] else 'INCOMPLETE, prefix '+row['partialgain']
            seconds = 'cached' if row['selectionseconds']=='' else f"{row['selectionseconds']:.4f}"
            print(f"{row['label']} | {gain} | {row['eliminated']}/{base} | {float(Fraction(row['percent'])):.3f}% | {seconds}")
            print('  '+ '; '.join(' -- '.join(f"{v['name']} [{v['id']}, {v['country']}]" for v in edge['markets']) for edge in row['removed']))
            if row['ties']: print('  Tied maximizers per round:',row['ties'])
            if row['error']: print('  INCOMPLETE:',row['error'])
        selected = [r for r in records if r['period']==period and r['status']=='complete']
        best = max(selected,key=lambda r:Fraction(r['gain']))
        regional = next(r for r in selected if r['method']=='kouremale')
        superior.append(Fraction(best['gain'])>Fraction(regional['gain']))
        print(f"Best tested: {best['label']}; gain is {float(Fraction(best['gain'])/Fraction(regional['gain'])):.2f} times Kouremale. This is not a proof of directed optimality.")
    if all(superior): print('\nAssessment: the coordinated regional cut is inferior to a tested alternative in both periods. A complete undirected separating boundary is not necessary to reduce directed reachability.')
    else: print('\nAssessment: a superior tested alternative was found in '+str(sum(superior))+' of the two periods; absence of a better tested cut does not prove optimality.')
    for method in ('directedgreedy','directedbc'):
        selected = [r for r in records if r['method']==method]
        common = set(selected[0]['cuts']) & set(selected[1]['cuts'])
        print(f"{names[method]} shares {len(common)} selected relationships across periods: {sorted(common)}.")
    print('The 2017 and pooled comparisons share observations and differ in active markets and survey coverage. Pooling adds possible paths without respecting shipment chronology; it is not an independent temporal replication.')
    print('Missing/unrecorded movements may change reachability and selected links. All 115 zero scaled-weight rows remain observed contacts. Scaled weights are neither exact animal counts nor costs and do not enter these unweighted policies.')
    print('Uniform market sources do not represent infection prevalence, livestock volume, or risk. Static directed reachability is not temporally feasible transmission or an epidemiological prediction.')
    if all(r['status']=='complete' for r in records):
        print('Recommendation: ADD AS SHORT COMPARISON. Its distinct contribution is the sensitivity to retaining movement direction, not another main undirected coordinated-cut success. No directed global optimum has been established.')
    else: print('Recommendation: DO NOT ADD at present. Incomplete capped policies prevent a final comparison; prefix results are explicitly marked.')

def directed():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--stepcap',type=float,default=30)
    parser.add_argument('--totalcap',type=float,default=60)
    args = parser.parse_args()
    if args.stepcap<=0 or args.totalcap<=0: parser.error('Caps must be positive')
    if not args.check and destination.exists(): raise FileExistsError('Refusing to overwrite directed.csv; use --check')
    start = perf_counter()
    protected = directedsnapshot()
    for message in directedregressions(): print(message,flush=True)
    graphs,saved = load()
    print('PASS original graphs, all eight cached b=3 directed gains, and Kouremale gains 36/61 and 117/131',flush=True)
    if args.check:
        if destination.exists():
            with destination.open(newline='') as stream: records = list(csv.DictReader(stream))
            print(directedreplay(records,graphs),flush=True)
        assert protected==directedsnapshot()
        print(f'PASS existing files unchanged; check runtime {perf_counter()-start:.3f}s')
        return
    records,hashes = [],provenance()
    for period,graph in graphs.items():
        base = reachable(graph)
        for method in ('kouremale','initial','recomputed','greedy'):
            original = saved[period,'exact' if method=='kouremale' else method]
            result = dict(cuts=json.loads(original['cuts']),status='complete',error='',seconds='',steps=[],attempts=[])
            records.append(directedrecord(period,graph,method,result,base,hashes,args.stepcap,args.totalcap))
        for method in ('directedgreedy','directedbc'):
            result = policy(graph,method,3,args.stepcap,args.totalcap)
            print(f"{period} {method}: {result['status']}, {len(result['cuts'])} relationships, selection {result['seconds']:.3f}s",flush=True)
            records.append(directedrecord(period,graph,method,result,base,hashes,args.stepcap,args.totalcap))
    assert protected==directedsnapshot()
    with destination.open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        for row in records: writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
    summary(records,graphs)
    print(f'Created only directed.py and directed.csv; existing pilot unchanged; total runtime {perf_counter()-start:.3f}s')

if __name__=='__main__':
    modes = {'--directed':directed}
    selected = [flag for flag in modes if flag in sys.argv]
    if len(selected)>1: raise SystemExit('Choose one analysis mode')
    action = modes[selected[0]] if selected else main
    if selected: sys.argv.remove(selected[0])
    if '--help' in sys.argv or '-h' in sys.argv:
        print('Additional analysis modes: --directed')
        if action in (main,directed,): action()
    else: action()
