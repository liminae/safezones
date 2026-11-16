# Independent sequential random benchmarks
import records as savedreports
import argparse
import ast
import csv
import hashlib
import json
import platform
import random
import signal
import sys
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from statistics import median
from time import perf_counter,sleep
from unittest.mock import patch
import networkx as nx

root = Path(__file__).resolve().parent.parent
folder = root/'data/results'
import safezones as core

master = 20261115
policies = ('weak','strong','bc','bcw','source-aware','threshold')
configs = dict(A=dict(model='ER',n=8,p=.24),B=dict(model='ER',n=9,p=.34),
               C=dict(model='SBM',n=8,blocks=(4,4),pin=.65,pout=.08),
               D=dict(model='SBM',n=9,blocks=(4,5),pin=.70,pout=.15))
timecap = 10
totalcap = 180

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def derived(*parts):
    text = '|'.join(map(str,(master,)+parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')

def signature(graph):
    return hashlib.sha256(json.dumps([sorted(graph),core.edge_order(graph)],separators=(',',':')).encode()).hexdigest()

def draw(name,seed):
    spec = configs[name]
    if spec['model']=='ER':
        original = nx.gnp_random_graph(spec['n'],spec['p'],seed=seed)
    else:
        p,q = spec['pin'],spec['pout']
        original = nx.stochastic_block_model(spec['blocks'],((p,q),(q,p)),seed=seed,directed=False,selfloops=False)
    graph = nx.Graph()
    graph.add_nodes_from(range(spec['n']))
    graph.add_edges_from(core.edge_order(original))
    return graph

def generate():
    accepted,history = [],[]
    for name in configs:
        count,attempt = 0,0
        while count<10:
            attempt += 1
            if attempt>10000: raise RuntimeError('Connected unique generation exceeded 10000 attempts')
            seed = derived('topology',name,attempt)
            graph = draw(name,seed)
            edges = core.edge_order(graph)
            item = dict(config=name,attempt=attempt,seed=seed,n=len(graph),m=len(edges),edges=edges,graphhash=signature(graph))
            if not nx.is_connected(graph):
                item.update(status='disconnected',components=sorted(map(len,nx.connected_components(graph)),reverse=True))
                history.append(item)
                continue
            degrees = sorted(dict(graph.degree()).values())
            duplicate = next((other for other in accepted if len(other['graph'])==len(graph)
                              and other['graph'].number_of_edges()==len(edges) and other['degrees']==degrees
                              and nx.is_isomorphic(graph,other['graph'])),None)
            if duplicate:
                item.update(status='duplicate',duplicateof=duplicate['id'],identicallylabeled=edges==core.edge_order(duplicate['graph']))
                history.append(item)
                continue
            label = f'{name}{count:02}'
            sourceseed = derived('sources',name,count)
            sources = sorted(random.Random(sourceseed).sample(sorted(graph),3))
            strengthseeds = [derived('strength',name,count,realization) for realization in range(2)]
            accepted.append(dict(id=label,config=name,index=count,attempt=attempt,topologyseed=seed,
                                 sourceseed=sourceseed,sources=sources,strengthseeds=strengthseeds,
                                 graphhash=item['graphhash'],degrees=degrees,graph=graph))
            item.update(status='accepted',id=label)
            history.append(item)
            count += 1
    return accepted,history

def strengths(item,realization):
    rng = random.Random(item['strengthseeds'][realization])
    return {edge:rng.uniform(1,10) for edge in core.edge_order(item['graph'])}

def expand(graph,infected):
    return set(infected)|{v for u in infected for v in graph.neighbors(u)}

def brute(graph,source):
    def visit(remaining,infected):
        if expand(remaining,infected)==infected: return len(graph)-len(infected)
        best = -1
        for action in [None]+list(remaining.edges):
            following = remaining.copy()
            if action is not None: following.remove_edge(*action)
            after = expand(following,infected)
            value = len(graph)-len(after) if after==infected else visit(following,after)
            best = max(best,value)
        return best
    return visit(graph.copy(),{source}|set(graph.neighbors(source)))

def replay(graph,source,cuts):
    remaining = graph.copy()
    infected = {source}|set(graph.neighbors(source))
    trace = [dict(round=0,infected=[source],cut=None),dict(round=1,infected=sorted(infected),cut=None)]
    order = iter(cuts)
    used = []
    for j in range(len(graph)):
        if expand(remaining,infected)==infected: break
        cut = next(order,None)
        if cut is not None:
            assert remaining.has_edge(*cut) and not all(v in infected for v in cut)
            remaining.remove_edge(*cut)
            used.append(cut)
        after = expand(remaining,infected)
        trace.append(dict(round=j+2,infected=sorted(after),cut=cut))
        if after==infected: break
        infected = after
    assert expand(remaining,infected)==infected
    return len(graph)-len(infected),used,trace

def validtrace(graph,source,policy,result):
    value,cuts,states = result
    assert states[0]==(frozenset({source}),None)
    infected = {source}|set(graph.neighbors(source))
    assert states[1]==(frozenset(infected),None)
    remaining,removed = graph.copy(),[]
    for j,(after,cut) in enumerate(states[2:]):
        assert expand(remaining,infected)!=infected
        if cut is not None:
            assert cut==tuple(sorted(cut)) and remaining.has_edge(*cut)
            assert not all(v in infected for v in cut)
            if policy=='threshold': assert sum(v in infected for v in cut)==1
            remaining.remove_edge(*cut)
            removed.append(cut)
        assert set(after)==expand(remaining,infected) and infected<=set(after)
        if set(after)==infected: assert j==len(states)-3
        infected = set(after)
    assert expand(remaining,infected)==infected and value==len(graph)-len(infected)
    assert cuts==removed and len(removed)==len(set(removed))

def expired(signum,frame):
    raise TimeoutError('Sequential exact per-instance time cap reached')

def limited(action,cap):
    if cap<=0: return None,0.0,'not run: total exact time cap exhausted'
    start = perf_counter()
    previous = signal.signal(signal.SIGALRM,expired)
    try:
        signal.setitimer(signal.ITIMER_REAL,cap)
        return action(),perf_counter()-start,''
    except TimeoutError as error:
        return None,perf_counter()-start,str(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)

def regeneration(items,history):
    other,again = generate()
    assert history==again and len(items)==len(other)==40
    assert Counter(item['config'] for item in items)==dict.fromkeys(configs,10)
    seeds = []
    for item,copy in zip(items,other):
        graph = item['graph']
        assert nx.is_connected(graph) and core.edge_order(graph)==core.edge_order(copy['graph'])
        assert set(graph)==set(range(configs[item['config']]['n']))
        assert item['graphhash']==signature(draw(item['config'],item['topologyseed']))
        assert item['sources']==copy['sources']==sorted(random.Random(item['sourceseed']).sample(sorted(graph),3))
        assert len(set(item['sources']))==3 and set(item['sources'])<=set(graph)
        for realization in range(2):
            values = strengths(item,realization)
            assert values==strengths(copy,realization) and all(1<=value<=10 for value in values.values())
        assert strengths(item,0)!=strengths(item,1)
        seeds += [item['topologyseed'],item['sourceseed']]+item['strengthseeds']
    assert len(seeds)==len(set(seeds))
    assert all(not nx.is_isomorphic(a['graph'],b['graph']) for a,b in combinations(items,2) if len(a['graph'])==len(b['graph']))
    return 'PASS deterministic graph/source/strength regeneration, connectedness, 40 nonisomorphic accepted topologies and separate RNG streams'

def checks():
    graphs = [nx.path_graph(3),nx.path_graph(4),nx.path_graph(5),nx.star_graph(4),nx.cycle_graph(4),nx.complete_graph(4)]
    graphs.append(nx.Graph([(0,1),(1,2),(2,0),(2,3),(3,4)]))
    count,traces = 0,0
    for graph in graphs:
        original = nx.node_link_data(graph)
        values = {edge:1+i/10 for i,edge in enumerate(core.edge_order(graph))}
        bc,bcw = core.centralities(graph,values)
        for source in sorted(graph):
            optimum,cuts = core.exact(graph,source,trace=True)
            assert optimum==brute(graph,source)==replay(graph,source,cuts)[0]
            count += 1
            with patch.object(core,'centralities',side_effect=AssertionError('Unexpected ranking recomputation')):
                for policy in policies:
                    result = core.simulate(graph,source,policy,values,bc,bcw,trace=True)
                    validtrace(graph,source,policy,result)
                    assert result[0]<=optimum
                    traces += 1
        assert nx.node_link_data(graph)==original
    graph = core.layered((1,2,2,1))
    optimum,cuts = core.exact(graph,0,trace=True)
    assert optimum==brute(graph,0)==replay(graph,0,cuts)[0]==1
    assert sum(v in {0,1,2} for v in cuts[0])==0
    assert core.exact(nx.path_graph(3),1)==0
    surviving = ((0,1),(2,3))
    values = dict.fromkeys(surviving,1.0)
    bc = {(0,1):10.0,(2,3):10.0-5e-13}
    assert core.choose_edge('bc',surviving,{2},values,bc,bc)==(0,1)
    assert core.choose_edge('source-aware',surviving,{2},values,bc,bc)==(2,3)
    bc[2,3] = 10.0-2e-12
    assert core.choose_edge('source-aware',surviving,{2},values,bc,bc)==(0,1)
    assert core.choose_edge('threshold',surviving,{2},values,bc,bc)==(2,3)
    assert core.choose_edge('threshold',surviving,{2,3},values,bc,bc) is None
    triangle = nx.Graph([(0,1),(1,2),(0,2)])
    bc,bcw = core.centralities(triangle,{(0,1):3,(1,2):3,(0,2):1})
    assert bcw[0,1]==bcw[1,2]==2 and bcw[0,2]==0
    ignored,seconds,error = limited(lambda:sleep(.03),.001)
    assert error and ignored is None
    return [f'PASS independent nonmemoized action-tree brute force: {count+1} graph/source cases, including waiting and all surviving edges',
            f'PASS {traces} policy traces, mandatory first spread, proactive optimal cuts, immutable inputs and no centrality recomputation',
            'PASS source-aware 1e-12 tie rule, threshold eligibility/waiting, inverse-strength BCw and timeout handling']

def solve(item,source,cap):
    graph = item['graph']
    result,seconds,error = limited(lambda:core.exact(graph,source,trace=True),cap)
    if error:
        return dict(status='timeout' if not error.startswith('not run') else 'unverified',value=None,cuts=[],seconds=seconds,
                    error=error,called=int(cap>0),trace=[],usedcuts=[])
    value,cuts = result
    actual,used,trace = replay(graph,source,cuts)
    assert actual==value
    return dict(status='verified',value=value,cuts=cuts,seconds=seconds,error='',called=1,trace=trace,usedcuts=used)

def evaluate(items,stage,cache,spent):
    rows = []
    for item in items:
        graph = item['graph']
        before = nx.node_link_data(graph)
        for source in item['sources']:
            key = item['id'],source
            assert key not in cache
            cache[key] = solve(item,source,min(timecap,max(0,totalcap-spent)))
            spent += cache[key]['seconds']
        for realization in range(2):
            values = strengths(item,realization)
            bc,bcw = core.centralities(graph,values)
            for source in item['sources']:
                exact = cache[item['id'],source]
                row = dict(config=item['config'],topology=item['id'],stage=stage,n=len(graph),m=graph.number_of_edges(),
                           density=nx.density(graph),degrees=item['degrees'],edges=core.edge_order(graph),
                           graphhash=item['graphhash'],attempt=item['attempt'],topologyseed=item['topologyseed'],
                           sourceseed=item['sourceseed'],selectedsources=item['sources'],source=source,realization=realization,
                           strengthseed=item['strengthseeds'][realization],strengths=[(u,v,values[u,v]) for u,v in core.edge_order(graph)],
                           exactstatus=exact['status'],exact='' if exact['value'] is None else exact['value'],
                           exactseconds=exact['seconds'],exacterror=exact['error'],exactcuts=exact['cuts'],
                           exactreplaycuts=exact['usedcuts'],exactreplay=exact['trace'],
                           originalbc=[(u,v,bc[u,v]) for u,v in core.edge_order(graph)],
                           originalbcw=[(u,v,bcw[u,v]) for u,v in core.edge_order(graph)])
                with patch.object(core,'centralities',side_effect=AssertionError('Unexpected ranking recomputation')):
                    for policy in policies:
                        value,cuts,states = core.simulate(graph,source,policy,values,bc,bcw,trace=True)
                        validtrace(graph,source,policy,(value,cuts,states))
                        if exact['status']=='verified': assert value<=exact['value']
                        row[policy] = value
                        row[policy+'ratio'] = str(Fraction(value,exact['value'])) if exact['status']=='verified' and exact['value']>0 else ''
                        row[policy+'optimal'] = value==exact['value'] if exact['status']=='verified' else ''
                        row[policy+'cuts'] = cuts
                        row[policy+'trace'] = [(sorted(infected),cut) for infected,cut in states]
                rows.append(row)
        assert nx.node_link_data(graph)==before
        result = [cache[item['id'],source] for source in item['sources']]
        print(f'{stage} {item["id"]}: n={len(graph)}, m={graph.number_of_edges()}, sources={item["sources"]}, '
              f'OPT={[r["value"] for r in result]}, exact seconds={sum(r["seconds"] for r in result):.6f}, '
              f'verified={sum(r["status"]=="verified" for r in result)}/3',flush=True)
    return rows,spent

def metrics(rows,policy):
    known = [row for row in rows if row['exactstatus']=='verified']
    positive = [row for row in known if row['exact']>0]
    ratios = [Fraction(row[policy],row['exact']) for row in positive]
    pairs = {(row['topology'],row['source']):row for row in rows}
    positivepairs = {(row['topology'],row['source']) for row in positive}
    result = dict(topologies=len({row['topology'] for row in rows}),sources=len(pairs),
                  positivesources=len(positivepairs),zerosources=sum(row['exactstatus']=='verified' and row['exact']==0 for row in pairs.values()),
                  weighted=len(rows),positiveweighted=len(positive),verifiedsources=sum(row['exactstatus']=='verified' for row in pairs.values()),
                  timeouts=sum(row['exactstatus']=='timeout' for row in pairs.values()),unverified=sum(row['exactstatus']!='verified' for row in pairs.values()),
                  positiveoptimal=100*sum(row[policy]==row['exact'] for row in positive)/len(positive) if positive else None,
                  alloptimal=100*sum(row[policy]==row['exact'] for row in known)/len(known) if known else None,
                  mean=float(sum(ratios)/len(ratios)) if ratios else None,
                  median=float(median(ratios)) if ratios else None,worst=float(min(ratios)) if ratios else None)
    means,successes = [],[]
    for name in sorted({row['topology'] for row in rows}):
        selected = [row for row in positive if row['topology']==name]
        if selected:
            bysource = {}
            for row in selected: bysource.setdefault(row['source'],[]).append(row)
            sourceaverages = [sum(Fraction(row[policy],row['exact']) for row in group)/len(group) for group in bysource.values()]
            sourcesuccess = [Fraction(sum(row[policy]==row['exact'] for row in group),len(group)) for group in bysource.values()]
            means.append(sum(sourceaverages)/len(sourceaverages))
            successes.append(sum(sourcesuccess)/len(sourcesuccess))
    result.update(positivetopologies=len(means),topologymean=float(sum(means)/len(means)) if means else None,
                  topologymedian=float(median(means)) if means else None,topologyworst=float(min(means)) if means else None,
                  topologyoptimal=100*float(sum(successes)/len(successes)) if successes else None)
    return result

def references():
    path = root/'code/tests.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='test_synthetic_regression')
    assignment = next(node for node in function.body if isinstance(node,ast.Assign)
                      and any(isinstance(target,ast.Name) and target.id=='expected' for target in node.targets))
    values = ast.literal_eval(assignment.value)
    return values,dict(path='code/tests.py',line=assignment.lineno,sha256=digest(path),kind='stored regression expectations, not a rerun')

def formatvalue(value):
    return 'undefined' if value is None else f'{value:.6f}'

def report(rows,items,history,cache,validation,pilot,reference,referenceinfo,seconds):
    tested = {row['topology'] for row in rows}
    lines = ['INDEPENDENT RANDOM-GRAPH SEQUENTIAL BENCHMARK',
             f'Master seed {master}; Python {platform.python_version()}; NetworkX {nx.__version__}.',
             'Production safezones.exact, safezones.centralities and safezones.simulate are called without modification.',
             'A0={source}; mandatory first spread on the original graph; then at most one permanent removal or wait before each simultaneous spread.',
             'Infection is permanent; stop at no new infection; no global deletion budget. Objective: final uninfected vertex count.',
             'Exact search memoizes infected vertices and surviving edges; waiting and susceptible-susceptible actions are allowed.',
             'Redundant infected-infected removals are omitted by the production exact solver.',
             'Weak/strong use smallest/largest strength. BC and BCw scores are computed on the original graph only.',
             'BCw distances are inverse strength. Production scores are unnormalized unordered NetworkX BC (half the paper ordered-pair value).',
             'All ranking and tie conventions are used unchanged. Source-aware BC uses its production 1e-12 tolerance on stored BC scores.',
             'Threshold BC considers only current infected-susceptible edges; other policies may make proactive cuts.',
             'Existing removals and infected-infected edges are excluded. Supplying cached centralities is checked to prevent recomputation.',
             'The exact solver returns cut order without wait slots. Stored exactreplay is a separately verified legal earliest-cut realization, not a transcript of internal search.',
             'Moving unrestricted cuts earlier cannot speed infection; the independent replay is required to attain the returned exact value.',
             '', 'SAMPLING AND REPRODUCIBILITY',
             'Graph attempts use nx.gnp_random_graph or nx.stochastic_block_model. No disconnected graph is repaired or reduced to a giant component.',
             'Graphs are generated from independent domain-separated RNG seeds. Accepted graphs are conditioned on connectedness and global nonisomorphism.',
             'Duplicate screening ignores labels and SBM block membership; connected isomorphic repeats are rejected, including across configurations with the same n.',
             'Therefore the accepted set is not an IID sample from the unconditioned ER/SBM distributions. It is a reproducible conditioned, distinct-topology sample.',
             'Processing order is A, B, C, D; cross-configuration duplicate rejection is documented below.',
             'Seed derivation: first 8 SHA256 bytes, big-endian, of master|domain|configuration|index.',
             'Topology domain uses attempt numbers starting at 1; sources domain uses accepted index 0..9; strength domain additionally appends realization 0 or 1.',
             'Three source vertices are sampled uniformly without replacement using a separate RNG, then sorted for evaluation.',
             'Two independently seeded Uniform(1,10) strength assignments use canonical edge order. Each draw is shared by all three sources and all six policies.',
             'Exact optimization runs once per tested topology/source; its result is reused across both strength draws.',
             'Repeated source/strength evaluations are not counted as independent graph topologies.',
             'All 40 topology choices and sources are generated before looking at exact-solver runtime or policy performance.',
             'Configurations: '+json.dumps(configs,sort_keys=True),
             '', 'FEASIBILITY PILOT',json.dumps(pilot,sort_keys=True),
             f'Limits: {timecap}s per exact instance, {totalcap}s cumulative exact calculation time.',
             'Pilot consists of the first two accepted graphs per configuration, with all three selected sources and both strength draws.',
             'Expansion occurs only if all 24 pilot optima verify and five times pilot exact time is within the cumulative cap.',
             'Pilot rows and exact results are reused in the expanded sample; pilot cases are not solved again.',
             'Timeouts have blank OPT/ratios, never an incumbent presented as exact. Hard cases are retained; there is no replacement sampling.',
             '', 'VALIDATION',*validation,
             '', 'COMPLETION',
             f'Accepted topologies generated: {len(items)}; evaluated: {len(tested)}; topology/source cases: {len(cache)}; weighted rows: {len(rows)}.',
             f'Production exact calls in benchmark: {sum(value["called"] for value in cache.values())}; verified: {sum(value["status"]=="verified" for value in cache.values())}.',
             f'Timeouts: {sum(value["status"]=="timeout" for value in cache.values())}; all unverified cases: {sum(value["status"]!="verified" for value in cache.values())}.',
             f'Zero-OPT source cases: {sum(value["value"]==0 for value in cache.values())}; retained in CSV, ratios blank.',
             'Untested accepted topologies: '+str([item['id'] for item in items if item['id'] not in tested]),
             '', 'REJECTION COUNTS', 'config | attempts | accepted | disconnected | duplicate']
    for name in configs:
        counts = Counter(item['status'] for item in history if item['config']==name)
        lines.append(f'{name} | {sum(counts.values())} | {counts["accepted"]} | {counts["disconnected"]} | {counts["duplicate"]}')
    lines += ['', 'AGGREGATION DEFINITIONS',
              'Primary success rate and all ratio statistics condition on verified OPT>0, matching the original benchmark convention.',
              'The secondary all-verified success rate includes trivial OPT=0 cases. Unverified cases are excluded from both rates and counted explicitly.',
              'Record-level ratios give equal weight to every source/strength evaluation with positive verified OPT.',
              'Topology-level ratios first average strength draws per source, then positive-OPT sources within each graph, then give each eligible graph equal weight.',
              'Topology-level success is aggregated in the same order. Graphs with no positive verified source have undefined ratio and are excluded from that graph-level mean.',
              'Topology median/minimum describe graph mean ratios, not the median/worst individual record. No independence-based confidence intervals are claimed.',
              '', 'RECORD-LEVEL RESULTS',
              'config | policy | graphs | source cases | positive sources | positive weighted rows | zero sources | timeouts | unverified | optimal positive % | optimal all verified % | mean ratio | median ratio | worst ratio']
    summaries = {}
    for name in list(configs)+['overall']:
        selected = rows if name=='overall' else [row for row in rows if row['config']==name]
        for policy in policies:
            value = metrics(selected,policy)
            summaries[name,policy] = value
            lines.append(f'{name} | {policy} | {value["topologies"]} | {value["sources"]} | {value["positivesources"]} | {value["positiveweighted"]} | '
                         f'{value["zerosources"]} | {value["timeouts"]} | {value["unverified"]} | '+
                         ' | '.join(formatvalue(value[key]) for key in ('positiveoptimal','alloptimal','mean','median','worst')))
    lines += ['', 'EQUAL-TOPOLOGY RESULTS',
              'config | policy | eligible graphs | mean graph success % | mean graph ratio | median graph ratio | minimum graph mean ratio']
    for (name,policy),value in summaries.items():
        lines.append(f'{name} | {policy} | {value["positivetopologies"]} | '+
                     ' | '.join(formatvalue(value[key]) for key in ('topologyoptimal','topologymean','topologymedian','topologyworst')))
    lines += ['', 'PER-TOPOLOGY MEAN RATIOS', 'graph | positive sources | weak | strong | BC | BCw | source-aware | threshold']
    for item in items:
        selected = [row for row in rows if row['topology']==item['id']]
        if not selected: continue
        values = [metrics(selected,policy) for policy in policies]
        lines.append(f'{item["id"]} | {values[0]["positivesources"]} | '+' | '.join(formatvalue(value['topologymean']) for value in values))
    lines += ['', 'CACHED HAND-DESIGNED COMPARISON',json.dumps(referenceinfo,sort_keys=True),
              'No original benchmark was rerun. The reference below is the stored exact-optimality regression table, not newly computed output.',
              'Reference sample: 26 topologies; 211 topology/source pairs; 8 strength draws; 1688 weighted records; 36 zero-OPT source pairs; 1400 positive-OPT records.',
              'Reference rates condition on OPT>0. Raw cached per-record synthetic outputs were not available in the inspected result locations;',
              'therefore reference mean/median ratios or equal-topology summaries are not invented from the success-rate table.',
              'family | weak | strong | BC | BCw | source-aware | threshold (positive-OPT exact-optimality %)']
    for name,values in reference.items(): lines.append(name+' | '+' | '.join(f'{value:.6f}' for value in values))
    lines += ['', 'DESCRIPTIVE COMPARISON',
              'The new graphs have n=8 or 9, three sampled sources and two strength draws; the reference uses varied designed sizes and every source.',
              'Differences are descriptive for this conditioned sample, not population-level significance claims.',
              'Bridge blocks, trees, brooms and barbells had high cached Threshold BC success; layered examples had 0% success.',
              'New positive-OPT success rates minus the cached overall rate, in percentage points:']
    for policy,old in zip(policies,reference['overall']):
        current = summaries['overall',policy]['positiveoptimal']
        lines.append(f'{policy}: '+('undefined' if current is None else f'{current:.6f}% versus {old:.6f}%; difference {current-old:+.6f}'))
    for name in configs:
        entry = summaries[name,'threshold']
        lines.append(f'{name}: Threshold BC positive-case success {formatvalue(entry["positiveoptimal"])}%, mean ratio {formatvalue(entry["mean"])}, worst ratio {formatvalue(entry["worst"])}; '
                     f'{entry["zerosources"]}/{entry["sources"]} source cases have OPT=0.')
    failures = [row for row in rows if row['exactstatus']=='verified' and row['exact']>0 and row['threshold']<row['exact']]
    lines.append(f'Threshold BC failures: {len(failures)} weighted records; {len({(row["topology"],row["source"]) for row in failures})} distinct topology/source cases; '
                 f'{len({row["topology"] for row in failures})} topologies.')
    for row in sorted(failures,key=lambda row:(Fraction(row['threshold'],row['exact']),row['topology'],row['source'],row['realization']))[:12]:
        lines.append(f'  {row["topology"]}, source={row["source"]}, draw={row["realization"]}: OPT={row["exact"]}, threshold={row["threshold"]}, '
                     f'ratio={row["thresholdratio"]}, edges={row["edges"]}, exact cuts={row["exactcuts"]}, threshold cuts={row["thresholdcuts"]}')
    lines += ['', 'RUNTIME',f'Total wall time: {seconds:.6f}s; exact calculation time: {sum(value["seconds"] for value in cache.values()):.6f}s.']
    if cache:
        key,value = max(cache.items(),key=lambda item:item[1]['seconds'])
        lines.append(f'Slowest exact case: {key}, {value["seconds"]:.6f}s, status={value["status"]}.')
    for key,value in cache.items():
        if value['status']!='verified': lines.append(f'UNVERIFIED {key}: {value}')
    lines += ['', 'FULL TOPOLOGY AND REJECTION MANIFEST',
              'Disconnected screening occurs before duplicate screening. Every attempt below records its complete edge list and seed.']
    lines += [json.dumps(item,sort_keys=True) for item in history]
    lines += ['', 'ACCEPTED SOURCE AND STRENGTH SEEDS']
    lines += [json.dumps({key:value for key,value in item.items() if key!='graph'},sort_keys=True) for item in items]
    lines += ['', 'FILES AND COMMANDS',
              'Production safezones.py SHA256: '+digest(root/'code/safezones.py'),
              'run.py SHA256: '+digest(__file__),
              'PYTHONDONTWRITEBYTECODE=1 python code/randomgraphs.py --check',
              'PYTHONDONTWRITEBYTECODE=1 python code/randomgraphs.py',
              'Script: code/randomgraphs.py; outputs: data/results/randomgraphs.csv and its entry in data/records.json are created. Existing project files remain unchanged.']
    return '\n'.join(lines)+'\n'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    paths = [folder/'randomgraphs.csv']
    if not args.check and (any(path.exists() for path in paths) or savedreports.exists('randomgraphs')): raise FileExistsError('Refusing to overwrite results; use --check for validation only')
    start = perf_counter()
    originals = {path:digest(root/path) for path in ('code/safezones.py','code/experiments.py','code/tests.py')}
    items,history = generate()
    validation = [regeneration(items,history)]+checks()
    for line in validation: print(line,flush=True)
    if args.check: return
    cache,spent = {},0.0
    pilotitems = [item for item in items if item['index']<2]
    rows,spent = evaluate(pilotitems,'pilot',cache,spent)
    verified = sum(value['status']=='verified' for value in cache.values())
    expandfull = verified==24 and 5*spent<=totalcap
    pilot = dict(graphs=len(pilotitems),sourcecases=len(cache),weightedrows=len(rows),verified=verified,
                 exactseconds=spent,slowest=max(value['seconds'] for value in cache.values()),projectedfullseconds=5*spent,
                 expanded=expandfull,decision='expand to all 40 graphs' if expandfull else 'retain explicitly limited eight-graph pilot; no hard-instance replacement')
    print('PILOT DECISION: '+json.dumps(pilot,sort_keys=True),flush=True)
    if expandfull:
        more,spent = evaluate([item for item in items if item['index']>=2],'expansion',cache,spent)
        rows += more
    expected = 240 if expandfull else 48
    assert len(rows)==expected and len(cache)*2==len(rows)
    assert sum(value['called'] for value in cache.values())<=len(cache)
    for name in configs:
        assert sum(row['config']==name for row in rows)==(60 if expandfull else 12)
    for key,value in cache.items():
        selected = [row for row in rows if (row['topology'],row['source'])==key]
        assert len(selected)==2 and len({row['exact'] for row in selected})==1
    assert all(digest(root/path)==value for path,value in originals.items())
    validation.append(f'PASS all {len(rows)*len(policies)} policy evaluations replayed; every verified score <= OPT; OPT reused across strength draws; production hashes unchanged')
    reference,referenceinfo = references()
    text = report(rows,items,history,cache,validation,pilot,reference,referenceinfo,perf_counter()-start)
    with paths[0].open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,separators=(',',':')) if isinstance(value,(list,tuple,dict)) else value for key,value in row.items()})
    savedreports.save('randomgraphs',text,exclusive=True)
    print(validation[-1],flush=True)
    for name in list(configs)+['overall']:
        selected = rows if name=='overall' else [row for row in rows if row['config']==name]
        for policy in policies:
            value = metrics(selected,policy)
            print(f'{name} {policy}: positive sources={value["positivesources"]}/{value["sources"]}; optimal={formatvalue(value["positiveoptimal"])}%; '
                  f'mean={formatvalue(value["mean"])}, median={formatvalue(value["median"])}, worst={formatvalue(value["worst"])}, '
                  f'equal-topology mean={formatvalue(value["topologymean"])}, unverified={value["unverified"]}',flush=True)
    print(f'Completed {len(rows)} weighted rows, {len(cache)} topology/source cases in {perf_counter()-start:.3f}s')
    for path in paths: print(path)

if __name__=='__main__':
    main()
