# Exact application threshold sweeps
import records as savedreports
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
from pathlib import Path
from time import perf_counter
import networkx as nx
import air
import commuting
from commuting import colorado
from hospital import edges,components,score

root = Path(__file__).resolve().parent.parent
inputs = {}
methods = ('bc','bcw','weak','strong')
airmodes = (('uniform','airports','uu'),('inbound','airports','mixed'),('uniform','inbound','mixed'),('inbound','inbound','pp'))
cacheversion = 'fixed-masses-v1'

def kappas():
    return sorted(set([Fraction(i,100) for i in range(75,151,5)]+[Fraction(i,1000) for i in range(980,1101,10)]+
                      [Fraction(i,10000) for i in range(13000,13201,25)]+[Fraction(95,96),Fraction(13,12),Fraction(46,35)]))

def giant(graph):
    vertices = min(nx.connected_components(graph),key=lambda c:(-len(c),tuple(sorted(c))))
    return graph.subgraph(vertices).copy()

def filterair(graph,tau):
    if not isinstance(tau,int) or isinstance(tau,bool) or tau<1: raise ValueError('Invalid passenger threshold')
    result = graph.copy()
    result.remove_edges_from(e for e in edges(graph) if graph.edges[e]['passengers']<tau)
    return result

def candidate(full,graph,vertices):
    selected = set(vertices)
    internal = full.subgraph(vertices)
    cuts = commuting.boundary(full,vertices)
    present = selected<=set(graph)
    connected = len(internal)==len(vertices) and nx.is_connected(internal)
    return dict(candidate=list(vertices),candidatepresent=present,candidateconnected=connected,
                candidateinternalcomponents=components(internal,()),candidateboundary=cuts,candidateboundarysize=len(cuts),
                candidatemissing=sorted(selected-set(graph)),candidatescope='full filtered graph',
                candidaterepresentable=present and connected)

def analyze(kind,graph,B,limit):
    start = perf_counter()
    modes = commuting.objectives if kind=='commuting' else airmodes
    mass = 'resident' if kind=='commuting' else 'inbound'
    cost = 'flow' if kind=='commuting' else 'passengers'
    weight = commuting.weights if kind=='commuting' else air.objectives
    solutions,stats,error = {},{},''
    try:
        catalog = commuting.catalog(graph,B,limit) if kind=='commuting' else air.partitions(graph,B,limit)
        stats['contraction'] = {k:v for k,v in catalog['stats'].items() if k!='core'}
        for mode in ('uu','mixed','pp'):
            solutions[mode],stats[mode] = air.optimize(catalog,mode,limit=limit)
    except RuntimeError as exc:
        error = str(exc)
    exactseconds = perf_counter()-start
    orders = commuting.rankings(graph)[0] if kind=='commuting' else air.rankings(graph)
    rows = []
    for prior,target,mode in modes:
        p,a = weight(graph,prior,target)
        denominator = len(graph) if prior=='uniform' else sum(graph.nodes[v][mass] for v in graph)
        for b in range(1,B+1):
            row = dict(prior=prior,target=target,budget=b,n=len(graph),m=graph.number_of_edges(),
                       totalmass=sum(graph.nodes[v][mass] for v in graph),status='exact' if mode in solutions else 'incomplete',
                       exact='',exactdecimal='',optimalcut=[],optimalpartition=[],componentsizes=[],componentmasses=[],
                       protectedifsourceingiant=[],nongiantcomponents=[],disruption='',searcherror=error)
            value = None
            if mode in solutions:
                state = solutions[mode][b]
                value = Fraction(air.numerator(catalog,mode,state),denominator)
                cuts = state[4]
                assert score(graph,cuts,p,a)==value
                groups = sorted(components(graph,cuts),key=lambda c:(-len(c),c))
                row.update(exact=str(value),exactdecimal=float(value),optimalcut=cuts,optimalpartition=groups,
                           componentsizes=list(map(len,groups)),componentmasses=[sum(graph.nodes[v][mass] for v in c) for c in groups],
                           protectedifsourceingiant=sorted(v for c in groups[1:] for v in c),nongiantcomponents=groups[1:],
                           disruption=sum(graph.edges[e][cost] for e in cuts))
            for method in methods:
                cuts = orders[method][:b]
                result = score(graph,cuts,p,a)
                assert value is None or result<=value
                row.update({method+'score':str(result),method+'decimal':float(result),method+'cuts':cuts,
                            method+'ratio':str(result/value) if value else '',method+'ratiodecimal':float(result/value) if value else ''})
            rows.append(row)
    stats.update(exactseconds=exactseconds,totalseconds=perf_counter()-start)
    return dict(rows=rows,stats=stats)

def decorate(payload,kind,value,full,graph):
    vertices = colorado if kind=='commuting' else air.washington
    info = candidate(full,graph,vertices)
    weight = commuting.weights if kind=='commuting' else air.objectives
    candidatevalues = {}
    for row in payload['rows']:
        key = row['prior'],row['target']
        if info['candidatepresent'] and key not in candidatevalues:
            p,a = weight(graph,*key)
            cuts = tuple(e for e in info['candidateboundary'] if graph.has_edge(*e))
            candidatevalues[key] = score(graph,cuts,p,a)
        candidatevalue = candidatevalues.get(key)
        feasible = info['candidaterepresentable'] and info['candidateboundarysize']<=row['budget']
        optimal = None if row['status']!='exact' else feasible and candidatevalue==Fraction(row['exact'])
        row.update(application=kind,parameter='kappa' if kind=='commuting' else 'tau',threshold=str(value),thresholddecimal=float(value),
                   fulln=len(full),fullm=full.number_of_edges(),fullcomponents=nx.number_connected_components(full),
                   excludedvertices=sorted(set(full)-set(graph)),**info,
                   candidatefeasible=feasible,candidatescore=str(candidatevalue) if candidatevalue is not None else '',
                   candidatedecimal=float(candidatevalue) if candidatevalue is not None else '',candidateoptimal=optimal,
                   candidateinrepresentative=any(tuple(c)==tuple(vertices) for c in row['nongiantcomponents']) if row['status']=='exact' else None)
    return payload

def signature(kind,graph,codehash):
    mass,cost = ('resident','flow') if kind=='commuting' else ('inbound','passengers')
    data = [cacheversion,codehash,kind,[(v,graph.nodes[v][mass]) for v in sorted(graph)],[(e,graph.edges[e][cost]) for e in edges(graph)]]
    return hashlib.sha256(json.dumps(data,separators=(',',':')).encode()).hexdigest()

def worker(task):
    kind,value,inputpath,cache,codehash,limit = task
    cache = savedreports.cachefolder(cache)
    start = perf_counter()
    if kind not in inputs:
        if kind=='commuting': inputs[kind] = commuting.readflows(commuting.workbookrows(inputpath))[:2]
        else: inputs[kind] = air.readgraph(inputpath)
    if kind=='commuting':
        nodes,pairs = inputs[kind]
        full = commuting.buildgraph(nodes,pairs,Fraction(value),False)
        B = 4
    else:
        base,audit = inputs[kind]
        full = filterair(base,int(value))
        B = 6
    graph = giant(full)
    if kind=='air': assert len(graph)>max(audit['componentcounts'][1:],default=0)
    key = signature(kind,graph,codehash)
    path = Path(cache)/(key+'.json')
    payload = json.loads(path.read_text()) if path.exists() else None
    reused = payload is not None and all(r['status']=='exact' for r in payload['rows'])
    if not reused: payload = analyze(kind,graph,B,limit)
    if not reused and all(r['status']=='exact' for r in payload['rows']):
        temporary = path.with_suffix('.'+str(os.getpid())+'.tmp')
        with temporary.open('x') as stream: json.dump(payload,stream,separators=(',',':'))
        temporary.replace(path)
    payload = decorate(payload,kind,Fraction(value),full,graph)
    payload.update(graphkey=key,reused=reused,seconds=perf_counter()-start)
    return kind,value,payload

def writecsv(path,rows):
    with Path(path).open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})

def runs(rows,field):
    groups,current = [],[]
    for row in sorted(rows,key=lambda r:Fraction(r['threshold'])):
        if row[field]: current.append(row['threshold'])
        elif current: groups.append(current);current = []
    if current: groups.append(current)
    return groups

def rangeformat(groups):
    return '; '.join(parametertext(g[0]) if len(g)==1 else parametertext(g[0])+' through '+parametertext(g[-1]) for g in groups) or 'none'

def parametertext(value):
    value = Fraction(value)
    decimal = f'{float(value):.9g}'
    return decimal if Fraction(decimal)==value else decimal+' ('+str(value)+')'

def groupedranges(rows,field):
    groups = {}
    for prior,target,b in sorted({(r['prior'],r['target'],r['budget']) for r in rows}):
        chosen = [r for r in rows if (r['prior'],r['target'],r['budget'])==(prior,target,b)]
        found = runs(chosen,field)
        if found: groups.setdefault((b,rangeformat(found)),[]).append(prior+'/'+target)
    return [f'b={b}; '+', '.join(names)+': '+ranges for (b,ranges),names in sorted(groups.items())]

def summary(results,paths,seconds,codehash):
    rows = {kind:[r for (k,v),p in sorted(results.items(),key=lambda item:(item[0][0],Fraction(item[0][1]))) if k==kind for r in p['rows']] for kind in ('commuting','air')}
    lines = ['EXACT RELIABILITY AND PASSENGER-THRESHOLD SWEEPS',
             'No existing application result, production function, or sequential semantics was changed.',
             'Inputs:']
    lines.extend(str(p)+' SHA256 '+hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths)
    lines += [f'Python {sys.version.split()[0]}; NetworkX {nx.__version__}; production-code SHA256 {codehash}',
              'Commuting grid: '+', '.join(parametertext(k) for k in kappas()),
              'Air grid: every integer tau from 1 through 60 inclusive.',
              'Named candidates: San Luis Valley = Alamosa, Conejos, Costilla, Mineral, Rio Grande and Saguache Counties, Colorado ('+', '.join(colorado)+'); Washington = '+', '.join(air.washington)+'.',
              'Scope: optimize each giant component. Full-graph diagnostics cover all 3109 counties, or the original 443-airport giant. Excluded original air components (sizes 3,2,2) cannot overtake any swept giant.',
              'Vertex weights remain fixed at original resident-worker or annual inbound-passenger totals; priors renormalize within each new giant. Airport weights therefore include traffic from routes later filtered out.',
              'Filters: county edges sum both flows and require one flow > kappa*its own MOE, using exact Fractions. Air edges require aggregate annual passengers >= tau after all original CLASS F/positive-passenger/contiguous-US/no-loop filters.',
              'Exactness: reuse commuting.catalog / air.partitions and air.optimize. Gomory-Hu contraction preserves all feasible partitions; local bundle enumeration and the existing proved dominance DP exhaust their combinations. No new optimizer or approximation is used.',
              'Scores use the existing Fraction component identity. BC, inverse-flow/passenger BCw, weak and strong are computed once per filtered graph, with original canonical ties; BC remains 2*NetworkX normalized=False. Rankings never recompute after cuts.',
              'candidateoptimal: the complete candidate boundary alone is feasible and globally optimal. candidateinrepresentative: the connected candidate occurs in a canonical optimum, possibly alongside other components. This latter flag makes no claim about all co-optima.',
              'optimalpartition gives every component. protectedifsourceingiant gives vertices protected conditional on the source lying in the giant remainder, not an unconditional safe set. Ratios are blank at OPT=0. Candidate connectivity uses the full filtered graph; scores are blank if any candidate vertex is outside the optimized giant.',
              'IMPORTANT: commuting ranges join CONSECUTIVE TESTED KAPPA VALUES, not certified continuous intervals. Air ranges exhaust every integer tau.',
              '', 'COMPLETION']
    for kind in rows:
        done = sum(r['status']=='exact' for r in rows[kind])
        lines.append(f"{kind}: {len({r['threshold'] for r in rows[kind]})} parameter values; {done}/{len(rows[kind])} exact objective/budget rows")
    failures = [(k,v,r['prior'],r['target'],r['budget'],r['searcherror']) for (k,v),p in results.items() for r in p['rows'] if r['status']!='exact']
    lines.append('Exact-search failures: '+(json.dumps(failures) if failures else 'none'))
    lines += ['', 'ORIGINAL-BUDGET GLOBAL OPTIMALITY RANGES']
    for kind,b in (('commuting',2),('air',3)):
        modes = commuting.objectives if kind=='commuting' else airmodes
        for prior,target,mode in modes:
            chosen = [r for r in rows[kind] if (r['prior'],r['target'],r['budget'])==(prior,target,b)]
            lines.append(f'{kind} {prior}/{target} b={b}: candidate alone optimal at '+rangeformat(runs(chosen,'candidateoptimal')))
    lines += ['', 'CANDIDATE-ALONE OPTIMALITY AT EVERY BUDGET']
    for kind in rows:
        lines.extend(kind+' '+line for line in groupedranges(rows[kind],'candidateoptimal'))
    lines.append('Unlisted budget/objective combinations have no matches on the tested grid.')
    lines += ['', 'CANDIDATE IN THE CANONICAL OPTIMAL PARTITION, POSSIBLY WITH OTHER COMPONENTS']
    for kind in rows:
        lines.extend(kind+' '+line for line in groupedranges(rows[kind],'candidateinrepresentative'))
    lines.append('Unlisted budget/objective combinations have no matches in the representative solutions.')
    lines += ['', 'CHANGES IN OPTIMALITY AT THE ORIGINAL BUDGET',
              'Each line gives the first tested parameter with the changed status and the winning non-giant components; untested transitions are bracketed by the preceding grid point.']
    changes = {}
    for kind,b in (('commuting',2),('air',3)):
        modes = commuting.objectives if kind=='commuting' else airmodes
        for prior,target,mode in modes:
            previous = None
            for row in (r for r in rows[kind] if (r['prior'],r['target'],r['budget'])==(prior,target,b)):
                status = row['candidateoptimal'],row['candidateinrepresentative']
                if previous is not None and status!=previous:
                    key = (kind,row['threshold'],b,status,tuple(map(tuple,row['nongiantcomponents'])))
                    changes.setdefault(key,[]).append(prior+'/'+target)
                previous = status
    for (kind,value,b,status,groups),names in sorted(changes.items(),key=lambda item:(item[0][0],Fraction(item[0][1]),str(item[0][2:]))):
        lines.append(f'{kind} {parametertext(value)} b={b}, '+', '.join(names)+f': candidate alone optimal={status[0]}, candidate in representative={status[1]}; winning components={groups}')
    lines += ['', 'STRUCTURAL TRANSITIONS',
              'Rows below appear whenever candidate connectivity, membership or boundary changes. Values are exact filter parameters; a transition between samples is only bracketed unless its directed estimate/MOE or passenger cutoff is identified explicitly.']
    for kind in rows:
        previous = None
        for row in [r for r in rows[kind] if r['prior']=='uniform' and r['target'] in ('counties','airports') and r['budget']==1]:
            key = row['candidatepresent'],row['candidateconnected'],row['candidateboundary'],row['candidateinternalcomponents']
            if key!=previous:
                lines.append(f"{kind} threshold={parametertext(row['threshold'])}: n={row['n']}, m={row['m']}; present={row['candidatepresent']}; connected={row['candidateconnected']}; internal sizes={list(map(len,row['candidateinternalcomponents']))}; boundary size={row['candidateboundarysize']}; missing={row['candidatemissing']}")
                previous = key
    lines += ['Commuting named-edge equalities: Mesa-Saguache drops at kappa=95/96; Costilla-Taos drops at kappa=13/12; Mineral-Rio Grande drops at kappa=46/35. These strict-equality points were explicitly tested.',
              'Air named-edge thresholds: FRD-LPS carries 7 annual passengers and drops at tau=8. PUL has only two original routes, PUL-LKE (36 passengers) and PUL-KEH (40), so it becomes isolated at tau=41 and leaves the giant. Candidate membership and other changes are measured on the filtered graph.',
              '', 'GRAPH SIZE AND BASELINE SUMMARY']
    for kind in rows:
        allrows = rows[kind]
        first,last = allrows[0],allrows[-1]
        nonzero = [r for r in allrows if r['status']=='exact' and Fraction(r['exact'])>0]
        lines.append(f"{kind}: first graph {first['n']} vertices/{first['m']} edges; last graph {last['n']}/{last['m']}; {sum(r['exact']=='0' for r in allrows)} zero-OPT rows; detailed graph sizes and all scores/ratios are in the CSV.")
        for method in methods:
            positive = [r for r in nonzero if Fraction(r[method+'score'])>0]
            optimal = sum(Fraction(r[method+'score'])==Fraction(r['exact']) for r in nonzero)
            thresholds = [dict(threshold=str(v),positive=any(Fraction(r['threshold'])==v for r in positive)) for v in sorted({Fraction(r['threshold']) for r in allrows})]
            maximum = max((Fraction(r[method+'ratio']) for r in nonzero),default=Fraction(0))
            lines.append(f"{kind} {method}: positive in {len(positive)}/{len(nonzero)} nonzero-OPT rows; optimal in {optimal}; largest ratio={float(maximum):.9g}; positive at some budget/objective at "+rangeformat(runs(thresholds,'positive')))
    lines += ['', 'SEARCH WORK AND REPRODUCIBILITY']
    for kind in rows:
        payloads = [p for (k,v),p in results.items() if k==kind]
        maximum = max((p['stats'].get('contraction',{}).get('localsubsets',0) for p in payloads),default=0)
        dpmax = max((s.get('maximumstates',0) for p in payloads for name,s in p['stats'].items() if isinstance(s,dict)),default=0)
        lines.append(f'{kind}: maximum local subsets={maximum}; maximum exact DP states={dpmax}')
    lines += [f'Elapsed wall time for this invocation: {seconds:.3f} seconds',
              f'Graphs reused from exact checkpoints: {sum(p["reused"] for p in results.values())}/{len(results)}; sum of original per-graph computation times: {sum(p["stats"]["totalseconds"] for p in results.values()):.3f} seconds (parallel tasks, not wall time)',
              'Per-graph checkpoints are temporary JSON files keyed by topology, source/target masses, flow weights, conventions and production-code hashes. They do not replace existing application outputs. Identical graph/objective instances may reuse exact calculations.',
              f'PYTHONDONTWRITEBYTECODE=1 python code/applicationsweeps.py --commuting-input "{paths[0]}" --air-input "{paths[1]}"',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py applicationsweeps',
              'Original-result regression anchors: kappa=1 and tau=1 must reproduce every existing exact score, canonical cut, baseline score and baseline cut. Original files are never rewritten.']
    return '\n'.join(lines)+'\n'

def anchors(results):
    for kind,value,filename in [('commuting','1','commutingresults.csv'),('air','1','airresults.csv')]:
        actual = {(r['prior'],r['target'],r['budget']):r for r in results[kind,value]['rows']}
        with (root/'data/results'/filename).open() as stream:
            for old in csv.DictReader(stream):
                new = actual[old['prior'],old['target'],int(old['budget'])]
                assert new['status']=='exact' and Fraction(new['exact'])==Fraction(old['exact'])
                assert list(map(list,new['optimalcut']))==json.loads(old['optimalcut'])
                for m in methods:
                    assert Fraction(new[m+'score'])==Fraction(old[m+'score'])
                    assert list(map(list,new[m+'cuts']))==json.loads(old[m+'cuts'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commuting-input',type=Path,default=root/'data/raw/table1.xlsx')
    parser.add_argument('--air-input',type=Path,default=root/'data/raw/t100_domestic_segment_2025.zip')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--cache',type=Path,default=Path(tempfile.gettempdir())/'safezones-sweeps')
    parser.add_argument('--limit',type=int,default=10000000)
    parser.add_argument('--output',type=Path,help='Separate directory for new results and report')
    args = parser.parse_args()
    destination = savedreports.prepare(args.output,('commutingsweep.csv', 'airsweep.csv'),('sweeps',))
    if args.workers<1 or args.limit<1: raise ValueError('Positive workers and search limit required')
    for path,digest in [(args.commuting_input,commuting.sourcehash),(args.air_input,air.sourcehash)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise ValueError('Unexpected input: '+str(path))
    codehash = hashlib.sha256(b''.join((root/'code'/name).read_bytes() for name in ('air.py','commuting.py','hospital.py'))).hexdigest()
    args.cache = savedreports.cachefolder(args.cache)
    args.cache.mkdir(parents=True,exist_ok=True)
    tasks = [('commuting',str(k),str(args.commuting_input),str(args.cache),codehash,args.limit) for k in kappas()]
    tasks += [('air',str(t),str(args.air_input),str(args.cache),codehash,args.limit) for t in range(1,61)]
    tasks.sort(key=lambda t:(t[1]!='1',t[0]!='air',Fraction(t[1])))
    start,results = perf_counter(),{}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker,t) for t in tasks]
        for future in as_completed(futures):
            kind,value,payload = future.result()
            results[kind,value] = payload
            rows = payload['rows']
            print(f"{len(results)}/{len(tasks)} {kind} {value}: n={rows[0]['n']}, m={rows[0]['m']}, exact={sum(r['status']=='exact' for r in rows)}/{len(rows)}, seconds={payload['seconds']:.2f}, reused={payload['reused']}",flush=True)
    anchors(results)
    for kind,filename in [('commuting','commutingsweep.csv'),('air','airsweep.csv')]:
        rows = [r for (k,v),p in sorted(results.items(),key=lambda item:Fraction(item[0][1])) if k==kind for r in p['rows']]
        writecsv(destination/filename,rows)
    savedreports.save('sweeps',summary(results,[args.commuting_input,args.air_input],perf_counter()-start,codehash),directory=destination)
    print('PASS kappa=1 and tau=1 reproduce all 40 original exact cases and all four baselines',flush=True)
    print('Wrote',destination/'commutingsweep.csv',destination/'airsweep.csv','and',destination/'sweeps.txt',flush=True)
    incomplete = sum(r['status']!='exact' for p in results.values() for r in p['rows'])
    print(f'Completed {sum(len(p["rows"]) for p in results.values())} rows; incomplete exact rows={incomplete}; wall seconds={perf_counter()-start:.3f}')
    if incomplete: raise SystemExit(1)

if __name__=='__main__':
    main()
