# County optimization and regional interpretation
import records as savedreports
import argparse
import csv
import hashlib
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from time import perf_counter
import networkx as nx
from air import partitions
from hospital import edges, components, score

root = Path(__file__).resolve().parent.parent

sourcehash = '1dbf52bede8af185ec487e193903e19332c9633cbb3da689afcc382fef7317ee'

sourceurl = 'https://www.census.gov/data/tables/2020/demo/metro-micro/commuting-flows-2020.html'

adjacencyurl = 'https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2010.txt'

states = set('01 04 05 06 08 09 10 11 12 13 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56'.split())

bigbend = ('48043','48243','48377')

bridge = ('48043','48371')

objectives = (('uniform','counties','uu'),('commuters','counties','mixed'),('uniform','commuters','mixed'),('commuters','commuters','pp'))

def workbookrows(path):
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with zipfile.ZipFile(path) as archive:
        strings = [''.join(t.text or '' for t in item.iter(ns+'t')) for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        with archive.open('xl/worksheets/sheet1.xml') as stream:
            for event,row in ET.iterparse(stream,events=('end',)):
                if row.tag!=ns+'row': continue
                values = {}
                for cell in row:
                    item = cell.find(ns+'v')
                    value = item.text if item is not None else ''
                    if cell.get('t')=='s': value = strings[int(value)]
                    values[''.join(c for c in cell.get('r') if c.isalpha())] = value
                row.clear()
                if values.get('A','').isdigit() and values.get('B','').isdigit(): yield values

def readflows(rows):
    nodes,pairs,seen = {},defaultdict(list),set()
    audit = Counter()
    def county(state,code,name,label):
        if not state.isdigit() or not code.isdigit(): return None
        state = f'{int(state):02}'
        if state not in states: return None
        key = state+f'{int(code):03}'
        if key not in nodes: nodes[key] = dict(name=name,state=label,resident=0,inbound=0,within=0)
        if (nodes[key]['name'],nodes[key]['state'])!=(name,label): raise ValueError('Inconsistent county metadata')
        return key
    for row in rows:
        flow,moe = int(row['I']),int(row['J'])
        if min(flow,moe)<0: raise ValueError('Negative estimate or MOE')
        key = tuple(row.get(c,'') for c in ('A','B','E','F','G','H'))
        if key in seen: raise ValueError('Duplicate directed flow')
        seen.add(key)
        audit['rows'] += 1
        u = county(row['A'],row['B'],row['D'],row['C'])
        v = county(row.get('E',''),row.get('F',''),row.get('H',''),row.get('G',''))
        if u: nodes[u]['resident'] += flow
        if v: nodes[v]['inbound'] += flow
        if not u or not v:
            audit['outsidecontiguous'] += 1
            continue
        if u==v:
            nodes[u]['within'] += flow
            audit['withincounty'] += 1
        else:
            pairs[tuple(sorted((u,v)))].append((u,v,flow,moe))
            audit['intercountydirections'] += 1
    if not nodes: raise ValueError('No contiguous-US counties')
    audit['counties'],audit['pairs'] = len(nodes),len(pairs)
    return nodes,{e:tuple(sorted(v)) for e,v in sorted(pairs.items())},dict(audit)

def buildgraph(nodes,pairs,k=1,giant=True):
    k = Fraction(str(k))
    if k<0: raise ValueError('Negative reliability threshold')
    graph = nx.Graph()
    graph.add_nodes_from((v,dict(nodes[v])) for v in sorted(nodes))
    for edge,rows in sorted(pairs.items()):
        if any(flow>k*moe for u,v,flow,moe in rows):
            graph.add_edge(*edge,flow=sum(flow for u,v,flow,moe in rows),directions=tuple(sorted(rows)))
    if giant and graph:
        chosen = min(nx.connected_components(graph),key=lambda c:(-len(c),tuple(sorted(c))))
        graph = graph.subgraph(chosen).copy()
    return graph

def weights(graph,prior,target):
    if prior not in ('uniform','commuters') or target not in ('counties','commuters'): raise ValueError('Unknown objective')
    p = {v:1 if prior=='uniform' else graph.nodes[v]['resident'] for v in graph}
    if any(v<0 for v in p.values()) or sum(p.values())<=0: raise ValueError('Invalid source weights')
    total = sum(p.values())
    return {v:Fraction(w,total) for v,w in p.items()},{v:1 if target=='counties' else graph.nodes[v]['resident'] for v in graph}

def catalog(graph,b=4,limit=2000000):
    adapted = graph.copy()
    for v in adapted: adapted.nodes[v]['inbound'] = adapted.nodes[v]['resident']
    for e in edges(adapted): adapted.edges[e]['passengers'] = adapted.edges[e]['flow']
    return partitions(adapted,b,limit)

def exact(data,limit=2000000):
    B,N,W = data['budget'],data['n'],data['w']
    choices = [(i,o) for i,branch in enumerate(data['branches']) for o in branch['options'] if o['cost']]
    best = {mode:{b:(-1,set()) for b in range(B+1)} for mode in ('uu','mixed','pp')}
    visits = 0
    def visit(start,spent,n,w,uu,mixed,pp,cuts):
        nonlocal visits
        visits += 1
        if visits>limit: raise RuntimeError(f'Exhaustive partition search exceeded {limit} states')
        values = dict(uu=N*N-(N-n)**2-uu,mixed=N*W-(N-n)*(W-w)-mixed,pp=W*W-(W-w)**2-pp)
        canonical = tuple(sorted(cuts))
        for mode,value in values.items():
            old,solutions = best[mode][spent]
            if value>old: best[mode][spent] = value,{canonical}
            elif value==old: solutions.add(canonical)
        for j in range(start,len(choices)):
            branch,option = choices[j]
            cost = spent+option['cost']
            if cost>B: continue
            following = j+1
            while following<len(choices) and choices[following][0]==branch: following += 1
            visit(following,cost,n+option['n'],w+option['w'],uu+option['uu'],mixed+option['mixed'],pp+option['pp'],cuts+option['cuts'])
    visit(0,0,0,0,0,0,0,())
    result = {}
    for mode in best:
        result[mode] = {}
        for b in range(B+1):
            value = max(best[mode][j][0] for j in range(b+1))
            cuts = sorted({cut for j in range(b+1) if best[mode][j][0]==value for cut in best[mode][j][1]},key=lambda c:(len(c),c))
            result[mode][b] = dict(numerator=value,cuts=cuts)
    return result,dict(globalpartitions=visits)

def rankings(graph):
    order = edges(graph)
    if any(graph.edges[e]['flow']<=0 for e in order): raise ValueError('Nonpositive retained flow')
    bc = {tuple(sorted(e)):2*v for e,v in nx.edge_betweenness_centrality(graph,normalized=False).items()}
    weighted = graph.copy()
    for e in order: weighted.edges[e]['distance'] = 1/graph.edges[e]['flow']
    bcw = {tuple(sorted(e)):2*v for e,v in nx.edge_betweenness_centrality(weighted,normalized=False,weight='distance').items()}
    orders = dict(bc=tuple(sorted(order,key=lambda e:(-bc[e],e))),bcw=tuple(sorted(order,key=lambda e:(-bcw[e],e))),
                  weak=tuple(sorted(order,key=lambda e:(graph.edges[e]['flow'],e))),strong=tuple(sorted(order,key=lambda e:(-graph.edges[e]['flow'],e))))
    return orders,bc,bcw

def adjacency(path):
    graph = nx.Graph()
    with Path(path).open(encoding='latin1',newline='') as stream:
        current = None
        for row in csv.reader(stream,delimiter='\t'):
            if len(row)!=4: raise ValueError('Expected the 2010 Census adjacency layout')
            if row[1]: current = row[1].zfill(5)
            neighbor = row[3].zfill(5)
            graph.add_node(current)
            if current!=neighbor: graph.add_edge(current,neighbor)
    return graph

def boundary(graph,vertices):
    selected = set(vertices)
    return tuple(e for e in edges(graph) if (e[0] in selected)!=(e[1] in selected))

def region(graph,vertices,geography=None):
    vertices = tuple(sorted(vertices))
    cuts = boundary(graph,vertices)
    known = geography is not None and set(vertices)<=set(geography)
    return dict(fips=vertices,counties=[graph.nodes[v]['name'] for v in vertices],states=sorted({graph.nodes[v]['state'] for v in vertices}),
                size=len(vertices),commuters=sum(graph.nodes[v]['resident'] for v in vertices),connected=nx.is_connected(graph.subgraph(vertices)),
                boundary=cuts,boundarysize=len(cuts),disruption=sum(graph.edges[e]['flow'] for e in cuts),
                adjacent2010=nx.is_connected(geography.subgraph(vertices)) if known else None)

def edgedetails(graph,cuts,orders=None,bc=None,bcw=None):
    bridges = set(tuple(sorted(e)) for e in nx.bridges(graph))
    ranks = {name:{e:i+1 for i,e in enumerate(order)} for name,order in (orders or {}).items()}
    return [dict(edge=e,counties=[graph.nodes[v]['name'] for v in e],states=[graph.nodes[v]['state'] for v in e],
                 flow=graph.edges[e]['flow'],directions=graph.edges[e]['directions'],bridge=e in bridges,
                 ranks={name:r[e] for name,r in ranks.items()},bc=bc[e] if bc else None,bcw=bcw[e] if bcw else None) for e in cuts]

def sensitivity(nodes,pairs):
    result = []
    for k in ('0.75','1.0','1.25','1.5','1.75'):
        graph = buildgraph(nodes,pairs,k,False)
        groups = sorted(components(graph,()),key=lambda c:(-len(c),c))
        cuts = boundary(graph,bigbend)
        bridges = set(tuple(sorted(e)) for e in nx.bridges(graph))
        connected = nx.is_connected(graph.subgraph(bigbend))
        item = dict(k=k,n=len(graph),m=graph.number_of_edges(),components=len(groups),giant=len(groups[0]),
                    sizes=dict(sorted(Counter(map(len,groups)).items())),bridges=len(bridges),
                    bigbendconnected=connected,bridgeseparated=connected and len(cuts)==1 and cuts[0] in bridges,
                    boundary=edgedetails(graph,cuts),scores={})
        for scope,H in (('allcounties',graph),('giant',graph.subgraph(groups[0]).copy())):
            if not set(bigbend)<=set(H):
                item['scores'][scope] = None
                continue
            item['scores'][scope] = {}
            for prior,target,mode in objectives:
                p,a = weights(H,prior,target)
                value = score(H,cuts,p,a)
                baseline = score(H,(),p,a)
                item['scores'][scope][prior+'/'+target] = dict(exact=str(value),decimal=float(value),baseline=str(baseline),increment=str(value-baseline))
        result.append(item)
    return result

def analyze(graph,solutions,orders,bc,bcw,geography=None):
    records,regions = [],{}
    for prior,target,mode in objectives:
        p,a = weights(graph,prior,target)
        denominator = len(graph) if prior=='uniform' else sum(graph.nodes[v]['resident'] for v in graph)
        for b,solution in solutions[mode].items():
            if not b: continue
            cuts = solution['cuts'][0]
            value = Fraction(solution['numerator'],denominator)
            assert score(graph,cuts,p,a)==value
            groups = sorted(components(graph,cuts),key=lambda c:(-len(c),c))
            summaries = [region(graph,c,geography) for c in groups[1:]]
            for alternate in solution['cuts']:
                assert score(graph,alternate,p,a)==value
                for c in sorted(components(graph,alternate),key=lambda c:(-len(c),c))[1:]:
                    if len(c)>1:
                        if c not in regions: regions[c] = dict(region(graph,c,geography),optimalin=[])
                        label = prior+'/'+target+f' b={b}'
                        if label not in regions[c]['optimalin']: regions[c]['optimalin'].append(label)
            bigvalue = score(graph,(bridge,),p,a)
            row = dict(prior=prior,target=target,budget=b,n=len(graph),m=graph.number_of_edges(),exact=str(value),decimal=float(value),
                       optimalcut=cuts,optimalpartitions=len(solution['cuts']),alloptimalboundaries=solution['cuts'],
                       componentsizes=list(map(len,groups)),componentcommuters=[sum(graph.nodes[v]['resident'] for v in c) for c in groups],
                       nongiant=summaries,disruption=sum(graph.edges[e]['flow'] for e in cuts),removed=edgedetails(graph,cuts,orders,bc,bcw),
                       bigbendexact=str(bigvalue),bigbendoptimalatone=(bridge,) in solutions[mode][1]['cuts'])
            for name,order in orders.items():
                result = score(graph,order[:b],p,a)
                assert result<=value
                row[name+'score'],row[name+'decimal'] = str(result),float(result)
                row[name+'ratio'] = str(result/value) if value else ''
                row[name+'ratiodecimal'] = float(result/value) if value else ''
                row[name+'cuts'] = order[:b]
                row[name+'components'] = sorted(map(len,components(graph,order[:b])),reverse=True)
            records.append(row)
    return records,list(regions.values())

def report(graph,audit,data,stats,records,regions,sweep,orders,bc,bcw,path,geopath):
    lines = ['COUNTY COMMUTING: EXACT STATIC APPLICATION',
             f'Workbook: {path}',f'Workbook SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}',
             'Source: '+sourceurl,'2016-2020 5-year ACS; workers age 16 and over. Estimates are workers, not annual trip counts.',
             f'Python {sys.version.split()[0]}; NetworkX {nx.__version__}',json.dumps(audit,sort_keys=True),
             'Main graph: k=1, giant component only, 3105 counties and 24338 undirected edges. The complete contiguous-US universe has 3109 counties; four isolated counties are excluded only from the main optimization.',
             'Direction-specific filter: retain uv iff at least one directed estimate exceeds k times THAT direction\'s MOE, strictly. Weight is the sum of BOTH reported directions, including any direction that fails the filter. Missing reverse flow contributes zero. Self-flows never create edges.',
             'County commuter mass = sum of all reported residence-origin worker estimates, including within-county workers and destinations outside the contiguous US. These masses are fixed before reliability filtering. Source weights are renormalized on the graph being scored. Targets use the same fixed masses.',
             'Contiguous US means the 48 states plus DC; county equivalents are included. County FIPS are zero-padded five-digit strings. All source-workplace records, including rows without a US workplace FIPS, contribute to the resident total.',
             'The objective is static: a(V)-sum p(C)*a(C), with at most b undirected relationships removed before connectivity is evaluated. This does not change the sequential production model. County-weighted protection is a graph exposure proxy, not an epidemiological estimate of protected people.',
             'BC and inverse-flow BCw are computed once on the original main graph, normalized=False and multiplied by 2 for ordered pairs. Scores are NetworkX floating-point outputs; rank positions break ties by canonical FIPS pairs. Weak/strong use increasing/decreasing total flow. No source information enters these rankings.',
             'EXHAUSTIVENESS',
             'The existing air application partition routine is reused read-only: on a temporary graph copy, resident mass is supplied under its inbound key and commuter flow under its passengers key. Those adapter keys never reinterpret or modify any existing air results.',
             'A unit-capacity Gomory-Hu tree contracts pairs whose minimum cut exceeds 4. Such vertices cannot be separated by any feasible cut, even if a contracted block is not induced-connected. Every feasible partition is therefore a partition of whole blocks.',
             'Restore every removed edge internal to a resulting component. The remaining cut is exactly its partition boundary; parallel original edges between two blocks must all be removed or need not be removed. Enumerate all local bundle subsets of cardinality at most 4.',
             'Remove the largest block from the quotient. The remaining connected branches interact only through this block. Every feasible global partition restricts to one local partition per branch. Conversely each combination of local options with total boundary cost <=4 gives a feasible partition. The new exhaustive recursion visits all such combinations, without dominance pruning, and compares integer objective numerators. It retains every tied optimal partition boundary.',
             'Returned co-optima are all distinct optimal component partitions and their minimal boundary edge sets, not redundant extra removals inside components. Each component partition has exactly one such boundary. Equal scores prefer fewer cuts then canonical FIPS order for the representative. A hard state limit raises an error rather than reporting an approximate optimum.',
             json.dumps({k:v for k,v in data['stats'].items() if k!='core'},sort_keys=True),json.dumps(stats,sort_keys=True),
             '', 'BIG BEND AND BC RANKING',json.dumps(region(graph,bigbend),sort_keys=True),
             'Internal flows: '+json.dumps([(e,graph.edges[e]['flow']) for e in edges(graph.subgraph(bigbend))]),
             'Boundary: '+json.dumps(edgedetails(graph,(bridge,),orders,bc,bcw),sort_keys=True),
             'Rank-1 BC edge: '+json.dumps(edgedetails(graph,orders['bc'][:1],orders,bc,bcw),sort_keys=True),
             'Rank-1 BC removal connected: '+str(len(components(graph,orders['bc'][:1]))==1),
             '', 'EXACT RESULTS AND FIXED RANKINGS']
    for row in records:
        lines.append(f"{row['prior']}/{row['target']} b={row['budget']}: {row['exact']} ({row['decimal']:.12f}); cuts={row['optimalcut']}; disruption={row['disruption']}; optimal partitions={row['optimalpartitions']}")
        lines.append('  Non-giant components: '+json.dumps(row['nongiant'],sort_keys=True))
        lines.append('  All optimal boundaries: '+json.dumps(row['alloptimalboundaries']))
        lines.append('  Removed edges: '+json.dumps(row['removed'],sort_keys=True))
        for name in orders:
            lines.append(f"  {name}: score={row[name+'score']} ({row[name+'decimal']:.12f}), ratio={row[name+'ratio']}; cuts={row[name+'cuts']}; component sizes={row[name+'components']}")
        if row['budget']==1: lines.append('  Brewster-Pecos exact b=1 optimum: '+str(row['bigbendoptimalatone']))
    lines += ['', 'MAIN FINDINGS',
              'Every objective/budget case has one optimal component partition. At b=1 all four objectives isolate Big Bend via Brewster-Pecos. Its ordinary BC rank is 179 and inverse-flow BCw rank is 1069. Rank-1 ordinary BC instead selects Cook County, IL--Douglas County, NE; this edge is not a bridge.',
              'At b=2 all four objectives isolate Alamosa, Conejos, Costilla, Mineral, Rio Grande and Saguache Counties, Colorado. Their 19095 resident-worker mass exceeds Big Bend\'s 8039. The coordinated two-edge boundary is Chaffee--Saguache (74) and Costilla--Taos (26), totaling 100 workers; neither edge alone is a bridge.',
              'For uniform/counties, commuters/counties and uniform/commuters, b=3 combines the Colorado group and Big Bend; b=4 additionally separates Edwards and Real Counties, Texas, via Real--Uvalde (203). The latter pair has resident-worker mass 2147.',
              'For commuters/commuters, b=3 instead isolates Josephine County, Oregon (32206 resident workers), and b=4 combines that singleton with Big Bend. The cross-state endpoints of its boundary do not make Josephine a multi-county region.',
              'All four fixed rankings leave the main graph connected for b=1,2,3,4, so all their protection scores and exact ratios are zero. The Colorado optimum requires a coordinated cut: its two edges have ordinary BC ranks 110 and 280, and weighted-BC ranks 708 and 6978. Source and target weighting changes which feasible partition is most valuable.',
              'Exact values (uniform/counties | commuters/counties | uniform/commuters | commuters/commuters)']
    for b in range(1,5):
        lines.append(f"b={b}: "+' | '.join(f"{r['decimal']:.9f}" for r in records if r['budget']==b))
    lines += ['', 'RELIABILITY SENSITIVITY',
              'For each k, n/m/components/bridges describe ALL 3109 counties, including isolates. Giant size is also reported. Fixed-cut scores are shown both on all counties and, if it contains all three candidate counties, on that k\'s giant component. Baseline and incremental protection distinguish pre-existing fragmentation from the new cut. These are exact scores of the stated Big Bend boundary, not optimized scores at other thresholds.',
              'At k=1.75 the three-county set is internally disconnected; its boundary still contains Brewster-Pecos, but that cut does not isolate a connected three-county cohort. A missing giant-scope score means at least one candidate county lies outside that giant.',
              'k | n | m | components | giant n | bridges | Big Bend connected | bridge-separated | boundary size']
    for row in sweep:
        lines.append(' | '.join(str(row[k]) for k in ('k','n','m','components','giant','bridges','bigbendconnected','bridgeseparated'))+' | '+str(len(row['boundary'])))
    lines.append('Big Bend fixed-cut scores on all counties (uniform/counties | commuters/counties | uniform/commuters | commuters/commuters), INCLUDING baseline fragmentation:')
    for row in sweep:
        lines.append(row['k']+': '+' | '.join(f"{row['scores']['allcounties'][p+'/'+a]['decimal']:.9f}" for p,a,mode in objectives))
    lines.append('Exact fractions, baseline/increment decomposition, giant-scope scores and every boundary edge:')
    lines.extend(json.dumps(row,sort_keys=True) for row in sweep)
    lines += ['', 'NON-SINGLETON REGIONS ACROSS EVERY CO-OPTIMAL PARTITION',
              'The giant remainder is excluded. Every non-singleton component of every optimum in all 16 objective/budget cases is included below. Disruption is the flow on that individual region boundary, not necessarily the full multi-component intervention.']
    if geopath:
        lines += ['Geographic check: connectedness in the 2010 Census county adjacency graph, which may include point/water adjacency. This is a historical adjacency check, not road connectivity, travel distance, or a current boundary survey.',
                  adjacencyurl,'Adjacency SHA256: '+hashlib.sha256(geopath.read_bytes()).hexdigest()]
    else: lines.append('No geographic adjacency file provided; same-state membership and commuting connectivity do not establish geographic contiguity.')
    for key,reverse,label in (('size',True,'By county count'),('commuters',True,'By resident-worker mass'),('disruption',False,'By lowest boundary disruption'),('adjacent2010',True,'By verified historical geographic coherence')):
        lines.append(label)
        lines.extend(json.dumps(r,sort_keys=True) for r in sorted(regions,key=lambda r:((-(r[key] or 0) if reverse else r[key]),r['fips'])))
    lines += ['', 'INTERPRETATION AND LIMITS',
              'All four objectives select Brewster-Pecos at b=1. The exact regional comparisons and all co-optima above determine the larger-budget results; a cardinality-optimal representative need not minimize disruption among ties.',
              'Reliability filtering changes the topology substantially. A retained bridge is a bridge of the filtered aggregate graph, not proof that all real travel or infection routes run through that county pair. The threshold is a user-specified estimate/MOE rule, not a multiple-testing procedure.',
              'The four pre-existing isolated counties create positive baseline protection if included; the main giant-component scores intentionally exclude that effect. Source priors and target masses are graph weights, with no stochastic or temporal mobility model.',
              '', 'REPRODUCE',f'PYTHONDONTWRITEBYTECODE=1 python code/commuting.py --input "{path}"'+(f' --adjacency "{geopath}"' if geopath else ''),
              'The adjacency reference is read-only and optional; download the linked Census file to a temporary path and pass --adjacency to reproduce the geographic check.',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py commuting',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py']
    return '\n'.join(lines)+'\n'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=root/'data/raw/table1.xlsx')
    parser.add_argument('--adjacency',type=Path)
    parser.add_argument('--output',type=Path,help='Separate directory for new results and report')
    args = parser.parse_args()
    destination = savedreports.prepare(args.output,('commutingresults.csv',),('commuting',))
    start = perf_counter()
    if hashlib.sha256(args.input.read_bytes()).hexdigest()!=sourcehash: raise ValueError('Workbook differs from the regression source')
    nodes,pairs,audit = readflows(workbookrows(args.input))
    graph = buildgraph(nodes,pairs)
    audit['excludedisolates'] = [dict(fips=v,**nodes[v]) for v in sorted(set(nodes)-set(graph))]
    assert (len(nodes),len(graph),graph.number_of_edges())==(3109,3105,24338)
    assert boundary(graph,bigbend)==(bridge,)
    assert [graph.edges[e]['flow'] for e in edges(graph.subgraph(bigbend))]==[281,221,223]
    assert graph.edges[bridge]['flow']==217
    print('PASS workbook: 3109 counties, main giant 3105 counties / 24338 edges; Big Bend flows 281, 221, 223 and boundary 217',flush=True)
    data = catalog(graph)
    solutions,stats = exact(data)
    print('Exact enumeration:',json.dumps({**{k:v for k,v in data['stats'].items() if k!='core'},**stats},sort_keys=True),flush=True)
    print('Computing fixed original-graph BC and inverse-flow BCw',flush=True)
    orders,bc,bcw = rankings(graph)
    print('Fixed rankings complete; scoring exact cuts, baselines and sensitivity graphs',flush=True)
    assert len(components(graph,orders['bc'][:1]))==1
    geography = adjacency(args.adjacency) if args.adjacency else None
    records,regions = analyze(graph,solutions,orders,bc,bcw,geography)
    sweep = sensitivity(nodes,pairs)
    assert all(r['bigbendoptimalatone'] for r in records)
    for r in records:
        print(f"{r['prior']}/{r['target']} b={r['budget']}: {r['decimal']:.12f}; cuts={r['optimalcut']}; optimal partitions={r['optimalpartitions']}",flush=True)
    print('Brewster-Pecos BC rank:',orders['bc'].index(bridge)+1,'of',graph.number_of_edges())
    print('Rank-1 BC edge:',orders['bc'][0],'; graph remains connected')
    for r in sweep: print('Sensitivity:',r['k'],r['n'],r['m'],r['components'],'components;',r['bridges'],'bridges; Big Bend connected',r['bigbendconnected'],'bridge-separated',r['bridgeseparated'])
    with (destination/'commutingresults.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        for row in records: writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
    savedreports.save('commuting',report(graph,audit,data,stats,records,regions,sweep,orders,bc,bcw,args.input,args.adjacency),directory=destination)
    print(f'Completed 16 exact optima and sensitivity sweep in {perf_counter()-start:.3f} seconds')
    print('Wrote',destination/'commutingresults.csv','and',destination/'commuting.txt')

colorado = ('08003','08021','08023','08079','08105','08109')

edwardsreal = ('48137','48385')

coloradocuts = (('08015','08109'),('08023','35055'))

labels = {colorado:'Colorado: San Luis Valley',bigbend:'Texas: Big Bend',edwardsreal:'Texas: Edwards-Real'}

regionurl = 'https://choosecolorado.com/regional-profile/san-luis-valley-region/'

adjacencypath = root/'data/raw/adjacency.txt'

def readresults(path,graph):
    with Path(path).open() as stream: rows = list(csv.DictReader(stream))
    cases,occurrences,saved = [],{},{}
    for row in rows:
        prior,target,b = row['prior'],row['target'],int(row['budget'])
        p,a = weights(graph,prior,target)
        value = Fraction(row['exact'])
        partitions = []
        for cuts in json.loads(row['alloptimalboundaries']):
            cuts = tuple(map(tuple,cuts))
            groups = sorted(components(graph,cuts),key=lambda c:(-len(c),c))
            assert score(graph,cuts,p,a)==value and len(cuts)<=b
            partitions.append(groups)
            for c in groups[1:]:
                if len(c)>1: occurrences.setdefault(c,set()).add((prior,target,b))
        assert len(partitions)==int(row['optimalpartitions'])
        for detail in json.loads(row['removed']):
            edge = tuple(detail['edge'])
            if edge in saved: assert saved[edge]==detail
            saved[edge] = detail
        cases.append(dict(prior=prior,target=target,budget=b,exact=str(value),partitions=partitions))
    assert len(cases)==16 and len({(c['prior'],c['target'],c['budget']) for c in cases})==16
    return cases,{c:sorted(v) for c,v in sorted(occurrences.items())},saved

def details(graph,selected,saved=None):
    bridges = set(tuple(sorted(e)) for e in nx.bridges(graph))
    result = []
    for edge in selected:
        rows = graph.edges[edge]['directions']
        savededge = (saved or {}).get(edge)
        if savededge: assert savededge['flow']==graph.edges[edge]['flow'] and savededge['bridge']==(edge in bridges)
        result.append(dict(edge=edge,counties=[graph.nodes[v]['name'] for v in edge],states=[graph.nodes[v]['state'] for v in edge],
                           flow=graph.edges[edge]['flow'],directions=rows,bridge=edge in bridges,
                           retentionlimit=str(max(Fraction(f,m) for u,v,f,m in rows)) if all(m for u,v,f,m in rows) else None,
                           ranks=savededge['ranks'] if savededge else None))
    return result

def structure(graph,vertices,geo,saved=None):
    vertices = tuple(sorted(vertices))
    internal = graph.subgraph(vertices)
    geographic = geo.subgraph(vertices)
    assert set(vertices)<=set(graph) and set(vertices)<=set(geo)
    cuts = boundary(graph,vertices)
    flow = sum(graph.edges[e]['flow'] for e in cuts)
    mass = sum(graph.nodes[v]['resident'] for v in vertices)
    return dict(fips=vertices,counties=[graph.nodes[v]['name'] for v in vertices],states=sorted({graph.nodes[v]['state'] for v in vertices}),
                count=len(vertices),commuters=mass,countyweights={v:graph.nodes[v]['resident'] for v in vertices},
                connected=nx.is_connected(internal),internalcomponents=components(internal,()),
                internaledges=details(graph,edges(internal)),internalflow=sum(d['flow'] for u,v,d in internal.edges(data=True)),
                boundaryedges=details(graph,cuts,saved),boundarysize=len(cuts),disruption=flow,
                geographicconnected=nx.is_connected(geographic),geographicadjacencies=edges(geographic),
                geographicdensity=nx.density(geographic),geographicdiameter=nx.diameter(geographic) if nx.is_connected(geographic) else None,
                disruptionpercounty=str(Fraction(flow,len(vertices))),disruptionpercommuter=str(Fraction(flow,mass)))

def evaluate(graph,data):
    cuts = tuple(tuple(d['edge']) for d in data['boundaryedges'])
    output = {}
    for prior,target,mode in objectives:
        p,a = weights(graph,prior,target)
        value,baseline = score(graph,cuts,p,a),score(graph,(),p,a)
        output[prior+'/'+target] = dict(exact=str(value),decimal=float(value),baseline=str(baseline),increment=str(value-baseline),
                                      disruptionperexpected=str(Fraction(data['disruption'],1)/value) if value else None)
    return output

def replayoriginal(graph,vertices,cuts):
    present = tuple(e for e in cuts if graph.has_edge(*e))
    groups = components(graph,present)
    anchor = next(c for c in groups if vertices[0] in c)
    return dict(cuts=present,persistent=tuple(vertices) in groups,anchorsize=len(anchor),
                added=tuple(sorted(set(anchor)-set(vertices))),lost=tuple(sorted(set(vertices)-set(anchor))),
                othercandidatecomponents=[c for c in groups if set(c)&set(vertices) and c!=anchor],
                alreadyisolated=[v for v in vertices if graph.degree(v)==0])

def sweep(nodes,pairs,geo):
    results = []
    for k in ('0.75','1.0','1.25','1.5','1.75'):
        graph = buildgraph(nodes,pairs,k,False)
        data = structure(graph,colorado,geo)
        results.append(dict(k=k,n=len(graph),m=graph.number_of_edges(),components=nx.number_connected_components(graph),
                            data=data,scores=evaluate(graph,data),replay=replayoriginal(graph,colorado,coloradocuts)))
    return results

def records(graph,regions,occurrences,sensitivity):
    rows = []
    snapshots = [('main','1',len(graph),graph.number_of_edges(),1,c,d,evaluate(graph,d),{}) for c,d in regions.items()]
    snapshots += [('sensitivity',r['k'],r['n'],r['m'],r['components'],colorado,r['data'],r['scores'],r['replay']) for r in sensitivity]
    for kind,k,n,m,count,c,data,values,replay in snapshots:
        for prior,target,mode in objectives:
            value = values[prior+'/'+target]
            rows.append(dict(record=kind,region=labels[c],k=k,scope='main giant' if kind=='main' else 'all contiguous counties',
                             n=n,m=m,components=count,prior=prior,target=target,**data,**value,
                             optimalcases=occurrences.get(c,[]) if kind=='main' else [],originalcutreplay=replay))
    return rows

def countylabel(graph,edge):
    return ' -- '.join(graph.nodes[v]['name']+' ['+v+']' for v in edge)

def regionreport(graph,regions,occurrences,cases,sensitivity,paths):
    C,B = regions[colorado],regions[bigbend]
    cs,bs = evaluate(graph,C),evaluate(graph,B)
    lines = ['COMMUTING OPTIMA: STRUCTURAL INTERPRETATION',
             'Read-only analysis of the completed k=1 results. No optimization at new reliability thresholds and no new centrality calculation are performed.',
             'Main universe: 3105 counties, 24338 reliable-flow edges, 152611850 resident-worker weight. County source/target masses include within-county workers and are fixed before thresholding.',
             'Flows are 2016-2020 ACS worker estimates, not annual journeys. Edge totals sum both directed estimates; retention requires at least one direction to satisfy flow > k*its own MOE.',
             'The source is unknown and protection is an expectation over the final partition. A separated region is safe only when the source lies outside it; a source inside it instead leaves the complementary component safe.',
             'Static score: a(V)-sum p(component)*a(component). Exact scores and cost ratios use Fraction arithmetic. Boundary disruption is the sum of retained undirected edge flows; within-county flows contribute to county target mass but not to internal intercounty edge totals.',
             'Input provenance:']
    lines.extend(str(path)+' SHA256 '+hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)
    lines += ['Workbook documentation: '+sourceurl,'Geographic adjacency reference: '+adjacencyurl,
              'Geographic tests use the 2010 Census adjacency file, including any point/water adjacency in that source; they do not establish road connectivity or geometric compactness.',
              'Regional interpretation: '+regionurl,
              '', 'COLORADO: THE SIX-COUNTY SAN LUIS VALLEY REGION',
              'Colorado OEDIT identifies exactly these six counties as its San Luis Valley region. The induced geographic adjacency graph is connected (9 edges, diameter 3); the reliable commuting graph is also connected.',
              'County | FIPS | resident-worker weight']
    lines.extend(f"{graph.nodes[v]['name']} | {v} | {graph.nodes[v]['resident']}" for v in colorado)
    lines += ['', 'All internal reliable-flow edges at k=1', 'Pair | undirected commuter total | directed estimate/MOE records']
    for edge in C['internaledges']:
        lines.append(countylabel(graph,edge['edge'])+' | '+str(edge['flow'])+' | '+json.dumps(edge['directions']))
    lines += [f"Internal total = {C['internalflow']} across {len(C['internaledges'])} edges; resident-worker mass = {C['commuters']}.",
              '', 'Exact boundary and original-graph ranking positions',
              'Pair | flow | BC rank | BCw rank | weak rank | strong rank | individually a bridge']
    for edge in C['boundaryedges']:
        lines.append(countylabel(graph,edge['edge'])+' | '+str(edge['flow'])+' | '+' | '.join(str(edge['ranks'][key]) for key in ('bc','bcw','weak','strong'))+' | '+str(edge['bridge']))
    lines += [f"Boundary total = {C['disruption']} across {C['boundarysize']} edges.",
              'Neither boundary edge is a bridge individually; either single removal leaves the main graph connected. Removing both produces components of sizes 6 and 3099. This is a coordinated two-edge cut, not two independent singleton isolations.',
              'Rank positions are copied and cross-checked from the existing exact-results CSV. BC/BCw use the original main graph and ordered-pair convention; BCw distance is inverse total flow. Ties use canonical FIPS order.',
              '', 'Exact objective values: Colorado versus Big Bend',
              'Source / target | Big Bend exact (decimal) | Colorado exact (decimal)']
    for prior,target,mode in objectives:
        key = prior+'/'+target
        lines.append(f"{key} | {bs[key]['exact']} ({bs[key]['decimal']:.12f}) | {cs[key]['exact']} ({cs[key]['decimal']:.12f})")
    lines += ['', 'Disruption per expected protected unit', 'Source / target | Big Bend | Colorado']
    for prior,target,mode in objectives:
        key = prior+'/'+target
        values = [Fraction(d[key]['disruptionperexpected']) for d in (bs,cs)]
        lines.append(key+' | '+' | '.join(f'{v} ({float(v):.12f})' for v in values))
    lines += ['', 'Comparison',
              'Geographic coherence: both groups are connected and each lies in one state. Big Bend has all 3 possible geographic adjacencies (density 1, diameter 1); Colorado has 9 of 15 (density 0.6, diameter 3). Big Bend is tighter in this adjacency sense. There is no area/distance evidence to rank geometric compactness. Colorado has a particularly clear regional interpretation because it matches the named six-county San Luis Valley region.',
              'Protected size conditional on a source outside the separated region: Colorado has 6 counties and 19095 resident workers, versus Big Bend\'s 3 counties and 8039 resident workers. Colorado also has the larger unconditional exact score under all four objectives.',
              'Colorado disrupts 100 estimated commuters rather than 217 despite using two relationships rather than one. It has lower disruption per expected protected unit for every objective. The optimal cardinality budgets differ, so this is a comparison of the selected b=1 and b=2 interventions, not a same-budget dominance claim.',
              'Conditional disruption per separated county: Big Bend '+B['disruptionpercounty']+'; Colorado '+C['disruptionpercounty']+'.',
              'Conditional disruption per separated resident-worker weight: Big Bend '+B['disruptionpercommuter']+'; Colorado '+C['disruptionpercommuter']+'.',
              'Edge-wise ranking failure: Big Bend is the simpler missed-bridge example (BC rank 179, BCw rank 1069). Colorado more directly illustrates coordination: both edges must be selected together, one has BCw rank 6978, and neither alone creates any protection. BC, BCw, weak and strong all score zero through b=4 in the existing results.',
              '', 'COLORADO RELIABILITY ROBUSTNESS',
              'Two distinct tests are reported: (1) keep the same six county IDs and inspect their induced graph and full boundary; (2) replay the ORIGINAL two-edge intervention, deleting only original edges that still exist, and inspect the component containing Alamosa. No community detection or optimum recomputation is used to define persistence.',
              'Sensitivity retains all 3109 contiguous-US counties, including isolates. Boundary-intervention scores in the CSV include exact baseline and increment, so protection from pre-existing fragmentation is identifiable.',
              'k | induced sizes | internal edges / flow | boundary edges / flow | exact six persist after original cut | Alamosa component | lost counties']
    for row in sensitivity:
        data,replay = row['data'],row['replay']
        lines.append(f"{row['k']} | {sorted(map(len,data['internalcomponents']),reverse=True)} | {len(data['internaledges'])} / {data['internalflow']} | {data['boundarysize']} / {data['disruption']} | {replay['persistent']} | {replay['anchorsize']} | {[graph.nodes[v]['name'] for v in replay['lost']]}")
    lines += ['At k=0.75 the six counties remain internally connected but have 21 boundary edges. The original two-edge cut leaves them within the 3107-county giant component: this is loss of separation, not discovery of a meaningful larger local region. Full-boundary scores at this threshold require 21 removals and are not feasible at budget 2 or 4.',
              'At k=1 the original two cuts isolate exactly six counties. At k=1.25 the same six persist and only Chaffee-Saguache remains on their boundary; Costilla-Taos has already failed the reliability filter.',
              'At k=1.5 and 1.75 Mineral County is already isolated before intervention. The connected Alamosa-containing region shrinks to Alamosa, Conejos, Costilla, Rio Grande and Saguache (18637 resident workers). Cutting Chaffee-Saguache isolates that five-county region; it does not newly protect Mineral.',
              'Exact thresholds explain the transitions: Costilla-Taos survives only for k<26/24=13/12; Mineral-Rio Grande only for k<max(46/35,43/42)=46/35; Chaffee-Saguache survives for k<74/38=37/19. All inequalities are strict.',
              '', 'Every Colorado boundary in the sweep']
    for row in sensitivity:
        lines.append('k='+row['k'])
        for edge in row['data']['boundaryedges']:
            lines.append('  '+countylabel(graph,edge['edge'])+' | '+str(edge['flow'])+' | '+json.dumps(edge['directions']))
    lines += ['', 'RECURRING NON-SINGLETON OPTIMAL REGIONS',
              'Every co-optimal partition in all 16 completed main-graph cases was replayed. Each case has a unique partition. The following are ALL non-giant, non-singleton components; giant remainders are accounted for separately below, rather than presented as regional discoveries.']
    for c,data in regions.items():
        lines += [labels[c],json.dumps(dict(fips=c,counties=data['counties'],size=data['count'],commuters=data['commuters'],
                                          boundarysize=data['boundarysize'],disruption=data['disruption'],geographicconnected=data['geographicconnected'],
                                          optimalcases=occurrences[c]),sort_keys=True)]
    lines += ['Colorado recurs in 10 of the 16 cases, Big Bend in 11, and Edwards-Real in 3; several cases contain multiple regions. All three are geographically connected in the Census reference.',
              'Josephine County, Oregon appears at commuters/commuters b=3 and b=4 but is a singleton, so it is not a fourth multi-county region.',
              '', 'Complete component accounting for every optimum',
              'The giant is the original 3105-county main vertex set minus the listed smaller components; this specifies its membership without repeating thousands of FIPS codes.',
              'Source / target | b | smaller components | giant size | giant resident-worker weight']
    for case in cases:
        for groups in case['partitions']:
            names = [labels.get(c,'singleton '+graph.nodes[c[0]]['name']) for c in groups[1:]]
            lines.append(f"{case['prior']}/{case['target']} | {case['budget']} | {names} | {len(groups[0])} | {sum(graph.nodes[v]['resident'] for v in groups[0])}")
    lines += ['', 'Interpretation limits',
              'These are structural facts about a filtered, undirected aggregate commuting graph. Weak estimates that fail the reliability rule are excluded, not known to be zero. Boundary suppression removes aggregated county relationships and is not a literal road closure or validated disease intervention.',
              'The six-county result is robust as a separated region from k=1 to 1.25, but loses two-edge separability at 0.75 and loses one connected member at 1.5. Scores at changed thresholds are fixed-boundary evaluations, not newly computed exact optima.',
              '', 'Reproduce',
              f'PYTHONDONTWRITEBYTECODE=1 python code/commuting.py --interpret --input "{paths[0]}" --adjacency "{paths[2]}"',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py commutinginterpret',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py commuting',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py']
    return '\n'.join(lines)+'\n'

def interpret():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=root/'data/raw/table1.xlsx')
    parser.add_argument('--adjacency',type=Path,default=adjacencypath)
    parser.add_argument('--output',type=Path,help='Separate directory for new results and report')
    args = parser.parse_args()
    destination = savedreports.prepare(args.output,('commutinginterpret.csv',),('commutinginterpret',))
    resultpath = root/'data/results/commutingresults.csv'
    if hashlib.sha256(args.input.read_bytes()).hexdigest()!=sourcehash: raise ValueError('Unexpected workbook')
    nodes,pairs,audit = readflows(workbookrows(args.input))
    graph,geo = buildgraph(nodes,pairs),adjacency(args.adjacency)
    cases,occurrences,saved = readresults(resultpath,graph)
    assert set(occurrences)==set(labels)
    regions = {c:structure(graph,c,geo,saved) for c in occurrences}
    sensitivity = sweep(nodes,pairs,geo)
    rows = records(graph,regions,occurrences,sensitivity)
    with (destination/'commutinginterpret.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
    savedreports.save('commutinginterpret',regionreport(graph,regions,occurrences,cases,sensitivity,[args.input,resultpath,args.adjacency]),directory=destination)
    for c,data in regions.items():
        print(f"{labels[c]}: {data['count']} counties, {data['commuters']} resident workers, internal flow {data['internalflow']}, boundary {data['boundarysize']} edges / {data['disruption']}, optimal in {len(occurrences[c])} cases")
    for row in sensitivity:
        print(f"k={row['k']}: induced sizes={sorted(map(len,row['data']['internalcomponents']),reverse=True)}, boundary={row['data']['boundarysize']} / {row['data']['disruption']}, original-cut Alamosa component={row['replay']['anchorsize']}")
    print(f'Validated {len(cases)} exact cases; wrote {len(rows)} region/objective records')
    print('Wrote',destination/'commutinginterpret.csv','and',destination/'commutinginterpret.txt')

if __name__=='__main__':
    modes = {'--interpret':interpret}
    selected = [flag for flag in modes if flag in sys.argv]
    if len(selected)>1: raise SystemExit('Choose one analysis mode')
    action = modes[selected[0]] if selected else main
    if selected: sys.argv.remove(selected[0])
    if '--help' in sys.argv or '-h' in sys.argv:
        print('Additional analysis modes: --interpret')
        if action in (main,interpret,): action()
    else: action()
