# Static algorithmic baseline comparison
import records as savedreports
import argparse
import csv
import hashlib
import json
import platform
import signal
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from time import perf_counter,sleep
import networkx as nx

root = Path(__file__).resolve().parent.parent
folder = root/'data/results'
import air
import commuting
from hospital import edges,score

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def signature(graph):
    return hashlib.sha256(json.dumps([sorted(graph),edges(graph)],separators=(',',':')).encode()).hexdigest()

def protection(graph,cuts=()):
    remaining = graph.copy()
    remaining.remove_edges_from(cuts)
    sizes = sorted(map(len,nx.connected_components(remaining)),reverse=True)
    return Fraction(len(graph)**2-sum(s*s for s in sizes),len(graph)),sizes

def bcorder(graph):
    values = {tuple(sorted(e)):2*v for e,v in nx.edge_betweenness_centrality(graph,normalized=False).items()}
    return tuple(sorted(values,key=lambda e:(-values[e],e))),values

def marginals(graph):
    bridges = {tuple(sorted(e)) for e in nx.bridges(graph)}
    sizes = dict.fromkeys(graph,1)
    forest = list(nx.dfs_edges(graph))
    for u,v in reversed(forest): sizes[u] += sizes[v]
    totals = {v:len(c) for c in nx.connected_components(graph) for v in c}
    gains = dict.fromkeys(edges(graph),0)
    for u,v in forest:
        edge = tuple(sorted((u,v)))
        if edge in bridges: gains[edge] = 2*sizes[v]*(totals[v]-sizes[v])
    assert sum(v>0 for v in gains.values())==len(bridges)
    return gains

def expired(signum,frame):
    raise TimeoutError('BC calculation reached its time cap')

def limited(action,cap):
    start = perf_counter()
    if cap<=0: return None,0.0,'Total BC time cap exhausted'
    previous = signal.signal(signal.SIGALRM,expired)
    try:
        signal.setitimer(signal.ITIMER_REAL,cap)
        return action(),perf_counter()-start,''
    except TimeoutError as error:
        return None,perf_counter()-start,str(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)

def step(graph,cuts,edge,seconds,ties,bc=None):
    before = protection(graph,cuts)[0]
    after,sizes = protection(graph,cuts+[edge])
    return dict(round=len(cuts)+1,edge=edge,seconds=seconds,ties=ties,orderedbc=bc,
                gain=str(after-before),score=str(after),components=sizes)

def policies(graph,b,cap=120,totalcap=300,progress=None):
    before = signature(graph)
    traces = {name:dict(steps=[],error='',attempts=[]) for name in ('initial','recomputed','greedy')}
    initial,elapsed,error = limited(lambda:bcorder(graph),min(cap,totalcap))
    attempt = dict(round=1,seconds=elapsed,completed=not error,shared=True,error=error)
    for name in ('initial','recomputed'): traces[name]['attempts'].append(attempt)
    if progress: print(f'{progress}: original BC {elapsed:.3f}s'+(' '+error if error else ''),flush=True)
    if error:
        traces['initial']['error'] = traces['recomputed']['error'] = error
    else:
        order,values = initial
        cuts = []
        for edge in order[:b]:
            traces['initial']['steps'].append(step(graph,cuts,edge,elapsed if not cuts else 0.0,None,values[edge]))
            cuts.append(edge)
        remaining,cuts = graph.copy(),[]
        spent = elapsed
        for j in range(b):
            if j:
                current,elapsed,error = limited(lambda:bcorder(remaining),min(cap,totalcap-spent))
                spent += elapsed
                traces['recomputed']['attempts'].append(dict(round=j+1,seconds=elapsed,completed=not error,shared=False,error=error))
                if progress: print(f'{progress}: recomputed BC round {j+1} {elapsed:.3f}s'+(' '+error if error else ''),flush=True)
                if error:
                    traces['recomputed']['error'] = error
                    break
                order,values = current
            edge = order[0]
            ties = sum(value==values[edge] for value in values.values())
            traces['recomputed']['steps'].append(step(graph,cuts,edge,elapsed,ties,values[edge]))
            cuts.append(edge)
            remaining.remove_edge(*edge)
    remaining,cuts = graph.copy(),[]
    for j in range(b):
        start = perf_counter()
        gains = marginals(remaining)
        best = max(gains.values())
        tied = [edge for edge in gains if gains[edge]==best]
        edge = min(tied)
        record = step(graph,cuts,edge,perf_counter()-start,len(tied))
        assert Fraction(record['gain'])==Fraction(best,len(graph))
        traces['greedy']['steps'].append(record)
        cuts.append(edge)
        remaining.remove_edge(*edge)
    assert signature(graph)==before
    return traces

def smallgraphs():
    first = nx.Graph()
    first.add_nodes_from(range(5))
    first.add_edges_from([(0,1),(0,4),(1,2),(1,3),(1,4),(2,3),(3,4)])
    second = nx.Graph()
    blocks = [tuple(f'{name}{j}' for j in range(size)) for name,size in (('R',4),('A',5),('B',7))]
    for block in blocks: second.add_nodes_from(block)
    links = {e for block in blocks for e in combinations(block,2)}
    links.update([('A0','R0'),('A1','R1'),('A0','B0'),('A1','B1'),('A2','B2')])
    second.add_edges_from(sorted(links))
    return [('A',first,Fraction(8,5)),('B',second,Fraction(6))]

def exhaustive(graph,b):
    best,cuts,count,winners = Fraction(-1),(),0,0
    for size in range(b+1):
        for chosen in combinations(edges(graph),size):
            remaining = graph.copy()
            remaining.remove_edges_from(chosen)
            value = Fraction(sum(len(graph)-len(nx.node_connected_component(remaining,s)) for s in graph),len(graph))
            count += 1
            if value>best: best,cuts,winners = value,chosen,1
            elif value==best: winners += 1
    return dict(value=best,cuts=cuts,subsets=count,winners=winners,source='Independent exhaustive source averaging')

def regressions():
    cases,lines = [],[]
    for name,graph,expected in smallgraphs():
        before = nx.node_link_data(graph)
        exact = exhaustive(graph,2)
        assert exact['value']==expected
        traces = policies(graph,2)
        for label,result in traces.items():
            assert not result['error'] and len(result['steps'])==2
            cuts = [item['edge'] for item in result['steps']]
            assert protection(graph,cuts)[0]<=expected
            assert protection(graph,cuts)[0]==(expected if name=='A' and label!='initial' else 0)
        assert [item['ties'] for item in traces['greedy']['steps']]==([7,1] if name=='A' else [42,41])
        assert nx.node_link_data(graph)==before
        cases.append((name,graph,{2:exact},traces))
        lines.append(f'PASS Case {name}: exhaustive OPT={expected}, {exact["subsets"]} subsets, {exact["winners"]} maximizing edge sets')
    graphs = [nx.path_graph(5),nx.cycle_graph(5),nx.disjoint_union(nx.path_graph(4),nx.cycle_graph(3))]
    graphs += [case[1] for case in cases]
    checked = 0
    for graph in graphs:
        for cut in ((),edges(graph)[:1]):
            remaining = graph.copy()
            remaining.remove_edges_from(cut)
            base = protection(remaining)[0]
            for edge,gain in marginals(remaining).items():
                assert Fraction(gain,len(remaining))==protection(remaining,[edge])[0]-base
                checked += 1
        traces = policies(graph,2)
        remaining = graph.copy()
        for item in traces['greedy']['steps']:
            values = {edge:protection(remaining,[edge])[0]-protection(remaining)[0] for edge in edges(remaining)}
            best = max(values.values())
            tied = sorted(edge for edge in values if values[edge]==best)
            assert item['edge']==tied[0] and item['ties']==len(tied) and Fraction(item['gain'])==best
            remaining.remove_edge(*item['edge'])
    order,values = bcorder(nx.path_graph(5))
    assert order[:2]==((1,2),(2,3)) and values[1,2]==values[2,3]==12
    for name,graph,exact,traces in cases:
        remaining = graph.copy()
        for item in traces['recomputed']['steps']:
            assert item['edge']==bcorder(remaining)[0][0]
            remaining.remove_edge(*item['edge'])
    result,seconds,error = limited(lambda:sleep(0.03),0.001)
    assert result is None and error
    lines += [f'PASS exact greedy bridge marginals: {checked} independent single-edge rescores; direct greedy tie counts and choices matched',
              'PASS BC convention, recomputation at each step, input immutability and timeout handling']
    return cases,lines

def cached(graph,name,budgets):
    target = 'airports' if name=='air' else 'counties'
    path = root/'data/results'/f'{name}results.csv'
    found = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            b = int(row['budget'])
            if row['prior']!='uniform' or row['target']!=target or b not in budgets: continue
            assert b not in found
            cuts = tuple(tuple(e) for e in json.loads(row['optimalcut']))
            value = Fraction(row['exact'])
            actual,sizes = protection(graph,cuts)
            assert len(cuts)<=b and all(graph.has_edge(*e) for e in cuts)
            assert actual==value and sizes==json.loads(row['componentsizes'])
            p,a = {v:Fraction(1,len(graph)) for v in graph},{v:1 for v in graph}
            assert score(graph,cuts,p,a)==value
            found[b] = dict(value=value,cuts=cuts,source=str(path.relative_to(root)),
                            cachedbc=Fraction(row['bcscore']),cachedcuts=json.loads(row['bccuts']),cachehash=digest(path))
    assert set(found)==set(budgets)
    assert found[max(budgets)]['value']==(Fraction(7812,443) if name=='air' else Fraction(4132,345))
    return found

def measured(name):
    start = perf_counter()
    if name=='air':
        path = root/'data/raw/t100_domestic_segment_2025.zip'
        assert digest(path)==air.sourcehash
        graph,audit = air.readgraph(path)
        expected = 443,7160
    else:
        path = root/'data/raw/table1.xlsx'
        assert digest(path)==commuting.sourcehash
        nodes,pairs,audit = commuting.readflows(commuting.workbookrows(path))
        graph = commuting.buildgraph(nodes,pairs,1,True)
        expected = 3105,24338
    assert (len(graph),graph.number_of_edges())==expected and nx.is_connected(graph)
    metadata = dict(case=name,input=str(path),inputhash=digest(path),graphhash=signature(graph),
                    constructor='air.readgraph' if name=='air' else 'commuting.readflows/workbookrows/buildgraph(k=1,giant=True)',
                    n=len(graph),m=graph.number_of_edges(),seconds=perf_counter()-start)
    return graph,metadata

def records(name,graph,exact,traces):
    rows = []
    for b,reference in exact.items():
        for label,trace in traces.items():
            steps = trace['steps'][:b]
            cuts = [item['edge'] for item in steps]
            complete = len(cuts)==b
            value,sizes = protection(graph,cuts)
            assert value<=reference['value']
            p,a = {v:Fraction(1,len(graph)) for v in graph},{v:1 for v in graph}
            assert score(graph,cuts,p,a)==value
            optimum,optsizes = protection(graph,reference['cuts'])
            assert optimum==reference['value']
            attempts = [attempt for attempt in trace['attempts'] if attempt['round']<=b]
            match = ''
            if label=='initial' and complete and 'cachedbc' in reference:
                match = value==reference['cachedbc'] and [list(e) for e in cuts]==reference['cachedcuts']
            rows.append(dict(case=name,n=len(graph),m=graph.number_of_edges(),budget=b,policy=label,
                             status='complete' if complete else 'incomplete',cuts=cuts,completedsteps=len(cuts),
                             score=str(value) if complete else '',scoredecimal=float(value) if complete else '',
                             ratio=str(value/optimum) if complete else '',ratiodecimal=float(value/optimum) if complete else '',
                             components=sizes if complete else [],partialscore='' if complete else str(value),
                             partialcomponents=[] if complete else sizes,exact=str(optimum),exactdecimal=float(optimum),
                             exactcuts=reference['cuts'],exactcomponents=optsizes,exactsource=reference['source'],
                             cachehash=reference.get('cachehash',''),cachedinitialmatch=match,
                             greedyties=[item['ties'] for item in steps] if label=='greedy' else [],
                             gains=[item['gain'] for item in steps],stepseconds=[item['seconds'] for item in steps],
                             steps=steps,bcattempts=attempts,error='' if complete else trace['error'],graphhash=signature(graph)))
    return rows

def report(rows,checks,metadata,seconds,cap,totalcap):
    lines = ['STATIC ALGORITHMIC BASELINE COMPARISON',
             'Uniform random source; unit vertex targets; Q(F)=(n^2-sum(component size squared))/n.',
             'These are fixed static edge interventions; no sequential contagion simulator or semantics is changed.',
             'Initial BC ranks the original graph once. Recomputed BC recomputes after every preceding deletion.',
             'Both use ordinary unweighted NetworkX edge_betweenness_centrality(normalized=False), multiplied by 2.',
             'Canonical edges are (min(u,v),max(u,v)); sorting is descending computed BC, then lexicographic edge.',
             'Case B uses literal A0..A4, B0..B6, R0..R3 labels, so alphabetical order is A, B, R.',
             'Computed BC ties use exact floating equality as in the existing applications; no new rounding rule is introduced.',
             'Objective Greedy compares exact integer gain numerators and uses canonical lexicographic ties.',
             'A non-bridge has zero one-step gain. A bridge splitting a current component into x,y vertices has gain 2xy/n.',
             'DFS subtree sizes evaluate all bridge gains exactly, including disconnected intermediate graphs.',
             'This is the requested marginal-gain policy, not a look-ahead or partition optimizer.',
             'All policies remove the full requested number of edges, including zero-gain choices.',
             'Each graph is run to its maximum requested budget; lower-budget outputs are prefixes of the same run.',
             'The original BC calculation is shared between Initial BC and round 1 of Recomputed BC.',
             'BC stepseconds measure centrality/ranking time; Initial BC later steps use the saved ranking and record zero additional BC time.',
             'BC budget-prefix timings overlap and must not be summed as independent trials.',
             f'BC time caps: {cap:g}s per calculation; {totalcap:g}s total BC calculation time per graph.',
             'Timeouts leave requested-budget scores/ratios blank; partial states are labeled separately.',
             'No measured-network exact optimization or existing sweep was run. Cached optima are replayed, not re-proved.',
             'The two theoretical optima are independently enumerated over all 0-, 1-, and 2-edge removal sets, using direct source averaging.',
             f'Python {platform.python_version()}; NetworkX {nx.__version__}; platform {platform.platform()}.',
             'Command: PYTHONDONTWRITEBYTECODE=1 python code/baselines.py'+
             ('' if (cap,totalcap)==(120,300) else f' --timecap {cap:g} --totalcap {totalcap:g}'),
             'baseline.py SHA256: '+digest(__file__),'','BUILT-IN REGRESSION CHECKS',*checks,'','INPUT PROVENANCE']
    lines += [json.dumps(item,sort_keys=True) for item in metadata]
    for row in rows:
        lines += ['',f'{row["case"]} b={row["budget"]} {row["policy"]}: {row["status"]}',
                  f'  n={row["n"]}, m={row["m"]}; cuts in selection order={row["cuts"]}',
                  f'  exact={row["exact"]} ({row["exactdecimal"]:.12g}); exact cut={row["exactcuts"]}; exact sizes={row["exactcomponents"]}',
                  f'  exact source={row["exactsource"]}; cache SHA256={row["cachehash"]}',
                  f'  score={row["score"]}; ratio={row["ratio"]}; component sizes={row["components"]}',
                  f'  per-round gains={row["gains"]}; greedy maximizing tie counts={row["greedyties"]}',
                  f'  step seconds={row["stepseconds"]}; BC attempts={row["bcattempts"]}']
        if row['cachedinitialmatch']!='': lines.append(f'  matches cached Initial BC cuts and score={row["cachedinitialmatch"]}')
        if row['status']=='incomplete':
            lines.append(f'  {row["error"]}; partial score={row["partialscore"]}; partial sizes={row["partialcomponents"]}')
        elif Fraction(row['score'])==0:
            assert row['components']==[row['n']]
            lines.append('  Zero protection is verified: the graph remains connected after the selected incomplete cut.')
    lines += ['','INTERPRETATION']
    for name in dict.fromkeys(row['case'] for row in rows):
        selected = [row for row in rows if row['case']==name]
        b = max(row['budget'] for row in selected)
        selected = [row for row in selected if row['budget']==b]
        lines.append(f'{name}, largest tested budget {b}: '+ '; '.join(f'{row["policy"]}={row["score"] or "incomplete"}, ratio={row["ratio"] or "unavailable"}' for row in selected))
        for row in selected:
            if row['policy']=='greedy':
                lines.append('  Objective Greedy maximizing tie counts by round: '+str(row['greedyties'])+'.')
                zero = [j+1 for j,gain in enumerate(row['gains']) if Fraction(gain)==0]
                lines.append('  Zero marginal-gain rounds: '+str(zero)+'. These are genuine objective plateaus.' if zero else '  All chosen greedy edges have strictly positive marginal gain.')
            if row['status']=='complete' and row['ratio']=='1':
                lines.append(f'  {row["policy"]} attains the verified optimum at this budget.')
            if row['status']=='complete' and row['policy']!='greedy' and row['score']=='0':
                matched = len({tuple(e) for e in row['cuts']}&{tuple(e) for e in row['exactcuts']})
                lines.append(f'  {row["policy"]} chooses {matched}/{len(row["exactcuts"])} edges of the displayed optimal boundary and leaves the graph connected.')
            if row['policy']=='greedy' and row['status']=='complete' and Fraction(row['ratio'])<1:
                lines.append(f'  Greedy component sizes {row["components"]} versus optimal component sizes {row["exactcomponents"]}.')
    lines += ['Greedy can value an existing bridge immediately, but initially non-bridge edges have zero marginal value even when a coordinated set has high value.',
              'Recomputed BC can change its choices after an unsuccessful deletion; its recorded results are measured, not inferred from Initial BC.',
              'No positive one-step gain is required by any policy, and no zero score is treated as an implementation error without component replay.',
              '',f'Completed policy/budget records: {sum(row["status"]=="complete" for row in rows)}/{len(rows)}',
              f'Incomplete policy/budget records: {sum(row["status"]!="complete" for row in rows)}',
              f'Total runtime: {seconds:.3f}s',
              'Script: code/baselines.py; outputs: data/results/baselines.csv and its entry in data/records.json were created. Existing project outputs were not overwritten.']
    return '\n'.join(lines)+'\n'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--timecap',type=float,default=120)
    parser.add_argument('--totalcap',type=float,default=300)
    args = parser.parse_args()
    if args.timecap<=0 or args.totalcap<=0: parser.error('Time caps must be positive')
    paths = [folder/'baselines.csv']
    if not args.check and (any(path.exists() for path in paths) or savedreports.exists('baselines')):
        raise FileExistsError('Refusing to overwrite existing baseline results; --check only runs built-in regressions')
    start = perf_counter()
    cases,checks = regressions()
    for line in checks: print(line,flush=True)
    if args.check: return
    rows,metadata = [],[]
    for name,graph,exact,traces in cases: rows.extend(records(name,graph,exact,traces))
    for name,budgets in (('air',(1,2,3)),('commuting',(1,2))):
        graph,info = measured(name)
        metadata.append(info)
        exact = cached(graph,name,budgets)
        print(f'{name}: n={len(graph)}, m={graph.number_of_edges()}, cached optima replayed; graph reconstruction {info["seconds"]:.3f}s',flush=True)
        traces = policies(graph,max(budgets),args.timecap,args.totalcap,name)
        rows.extend(records(name,graph,exact,traces))
    assert len(rows)==21
    text = report(rows,checks,metadata,perf_counter()-start,args.timecap,args.totalcap)
    with paths[0].open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,separators=(',',':')) if isinstance(value,(list,tuple,dict)) else value for key,value in row.items()})
    savedreports.save('baselines',text,exclusive=True)
    for row in rows:
        print(f'{row["case"]} b={row["budget"]} {row["policy"]}: {row["status"]}, Q={row["score"] or "unavailable"}, ratio={row["ratio"] or "unavailable"}, sizes={row["components"]}',flush=True)
    print(f'Completed {sum(row["status"]=="complete" for row in rows)}/{len(rows)} records in {perf_counter()-start:.3f}s')
    for path in paths: print(path)

if __name__=='__main__':
    main()
