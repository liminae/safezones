# Coupled stochastic transmission checks
import records as savedreports
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from time import perf_counter
import networkx as nx

root = Path(__file__).resolve().parent.parent
folder = root/'data/results'
import air
import highschool

seed = 20261115
levels = tuple(map(Fraction,('1/4','1/2','3/4','9/10','1')))
normal = NormalDist().inv_cdf(.975)
decisioncritical = NormalDist().inv_cdf(.9875)
precision = .10

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def signature(graph):
    return hashlib.sha256(json.dumps([sorted(graph),air.edges(graph)],separators=(',',':')).encode()).hexdigest()

def find(parent,v):
    while parent[v]!=v:
        parent[v] = parent[parent[v]]
        v = parent[v]
    return v

def join(parent,sizes,u,v):
    u,v = find(parent,u),find(parent,v)
    if u==v: return 0
    if sizes[u]<sizes[v]: u,v = v,u
    loss = 2*sizes[u]*sizes[v]
    parent[v] = u
    sizes[u] += sizes[v]
    return loss

def groups(n,links,mask):
    parent,sizes = list(range(n)),[1]*n
    for i,(u,v) in enumerate(links):
        if mask>>i&1: join(parent,sizes,u,v)
    members = {}
    for v in range(n): members.setdefault(find(parent,v),[]).append(v)
    return list(members.values())

def sources():
    path = root/'data/raw/t100_domestic_segment_2025.zip'
    assert digest(path)==air.sourcehash
    graph,audit = air.readgraph(path)
    assert (len(graph),graph.number_of_edges())==(443,7160) and nx.is_connected(graph)
    cache = root/'data/results/baselines.csv'
    with cache.open(newline='') as stream:
        saved = [row for row in csv.DictReader(stream) if row['case']=='air' and row['budget']=='3']
    assert len(saved)==3 and {row['policy'] for row in saved}=={'initial','recomputed','greedy'}
    with (root/'data/results/airresults.csv').open(newline='') as stream:
        exact = [row for row in csv.DictReader(stream) if row['prior']=='uniform' and row['target']=='airports' and row['budget']=='3']
    assert len(exact)==1 and Fraction(exact[0]['exact'])==Fraction(7812,443)
    boundary = tuple(tuple(e) for e in json.loads(exact[0]['optimalcut']))
    assert boundary==(('BFI','LPS'),('ESD','LPS'),('FRD','LPS'))
    candidates = [dict(name='none',cuts=(),kind='baseline'),dict(name='coordinated',cuts=boundary,kind='full boundary')]
    deterministic = {'none':Fraction(0),'coordinated':Fraction(exact[0]['exact'])}
    for name in ('initial','recomputed','greedy'):
        row = next(row for row in saved if row['policy']==name)
        assert row['status']=='complete' and row['graphhash']==signature(graph)
        cuts = tuple(tuple(e) for e in json.loads(row['cuts']))
        assert len(cuts)==3 and all(graph.has_edge(*e) for e in cuts)
        candidates.append(dict(name=name,cuts=cuts,kind='saved fixed baseline'))
        deterministic[name] = Fraction(row['score'])
    for size in (1,2):
        for cuts in combinations(boundary,size):
            name = '+'.join('--'.join(e) for e in cuts)
            candidates.append(dict(name=name,cuts=cuts,kind='partial boundary'))
    weights,strengths,daily,schoolaudit = highschool.readraw()
    school = highschool.makegraph(weights,strengths,5300)
    assert (len(school),school.number_of_edges())==(9,12) and nx.is_connected(school)
    assert sum(weights.values())==329
    with (root/'data/results/highschooledges.csv').open(newline='') as stream:
        cachededges = {(row['left'],row['right']):int(row['seconds']) for row in csv.DictReader(stream) if row['period']=='week'}
    assert cachededges==dict(strengths)
    with (root/'data/results/highschoolstatic.csv').open(newline='') as stream:
        priors = [row for row in csv.DictReader(stream) if row['scope']=='prior' and row['theta']=='5300' and row['budget']=='1' and row['method']=='exact']
    assert {row['prior'] for row in priors}=={'class-uniform','student-uniform','half PC','half 2BIO3','half PC*'}
    paths = [path,cache,root/'data/results/airresults.csv',root/'data/results/highschooledges.csv',root/'data/results/highschoolstatic.csv',
             root/'data/raw/HighSchool2013_metadata.txt',root/'data/raw/HighSchool2013_proximity_net.csv.gz',
             root/'code/air.py',root/'code/highschool.py',root/'code/baselines.py']
    provenance = dict(files={str(p.relative_to(root)):digest(p) for p in paths},airgraph=signature(graph),schoolgraph=signature(school),
                      populations=weights,schooloriginalpriors=[{key:row[key] for key in ('prior','expected','cuts','priorvalues')} for row in priors])
    provenance['files']['source:highschool'] = hashlib.sha256(savedreports.data('source:highschool')).hexdigest()
    return graph,candidates,deterministic,school,weights,priors,provenance

def airplan(graph,candidates):
    nodes = sorted(graph)
    index = {v:i for i,v in enumerate(nodes)}
    original = air.edges(graph)
    links = [(index[u],index[v]) for u,v in original]
    positions = {e:i for i,e in enumerate(original)}
    blocked = [frozenset(positions[e] for e in candidate['cuts']) for candidate in candidates]
    touched = set().union(*blocked)
    common = [(i,u,v) for i,(u,v) in enumerate(links) if i not in touched]
    special = [(i,*links[i]) for i in sorted(touched)]
    return dict(n=len(nodes),links=links,blocked=blocked,common=common,special=special)

def realization(plan,active):
    n = plan['n']
    parent,sizes = list(range(n)),[1]*n
    base = n*n-n
    for i,u,v in plan['common']:
        if active[i]: base -= join(parent,sizes,u,v)
    special = [(i,u,v) for i,u,v in plan['special'] if active[i]]
    values,components = [],[]
    for removed in plan['blocked']:
        current,counts = parent.copy(),sizes.copy()
        value = base
        for i,u,v in special:
            if i not in removed: value -= join(current,counts,u,v)
        parts = sorted((counts[v] for v in range(n) if current[v]==v),reverse=True)
        assert sum(parts)==n and value==n*n-sum(s*s for s in parts)
        values.append(value)
        components.append(parts)
    assert all(value>=values[0] for value in values)
    assert all(values[1]>=value for removed,value in zip(plan['blocked'],values) if removed<=plan['blocked'][1])
    return values,components

def interval(values,denominator,critical=normal):
    count = len(values)
    total = sum(values)
    mean = total/(count*denominator)
    variance = count*sum(v*v for v in values)-total*total
    error = math.sqrt(variance/(count*count*(count-1)))/denominator if count>1 else 0.0
    return mean,mean-critical*error,mean+critical*error

def schooltable(graph,weights):
    nodes = sorted(graph)
    index = {v:i for i,v in enumerate(nodes)}
    original = air.edges(graph)
    links = [(index[u],index[v]) for u,v in original]
    population = [weights[v] for v in nodes]
    total,n = sum(population),len(nodes)
    values = {'class-uniform':[],'population':[]}
    sizes = []
    for mask in range(1<<len(links)):
        parts = groups(n,links,mask)
        masses = [sum(population[v] for v in part) for part in parts]
        assert sum(masses)==total and sum(map(len,parts))==n
        values['class-uniform'].append(total*n-sum(len(part)*mass for part,mass in zip(parts,masses)))
        values['population'].append(total*total-sum(mass*mass for mass in masses))
        sizes.append(sorted(map(len,parts),reverse=True))
    coefficients = {}
    for prior,table in values.items():
        coefficients[prior] = []
        for cut in [0]+[1<<i for i in range(len(links))]:
            totals = [0]*(len(links)+1)
            for mask in range(1<<len(links)): totals[mask.bit_count()] += table[mask&~cut]
            coefficients[prior].append(totals)
    return dict(nodes=nodes,edges=original,links=links,population=population,total=total,values=values,sizes=sizes,
                denominators={'class-uniform':n,'population':total},coefficients=coefficients)

def schoolmeans(table,q,prior):
    m = len(table['edges'])
    a,d = q.numerator,q.denominator
    powers = [a**k*(d-a)**(m-k) for k in range(m+1)]
    assert sum(math.comb(m,k)*powers[k] for k in range(m+1))==d**m
    return [Fraction(sum(value*power for value,power in zip(totals,powers)),d**m*table['denominators'][prior])
            for totals in table['coefficients'][prior]]

def validate(graph,candidates,deterministic,school,weights,priors,plan,table):
    checks = []
    for row in priors:
        p = {v:Fraction(w) for v,w in json.loads(row['priorvalues']).items()}
        assert set(p)==set(school) and sum(p.values())==1 and all(value>=0 for value in p.values())
        values = {e:highschool.componentvalue(school,(e,),weights,p) for e in air.edges(school)}
        assert max(values.values())==Fraction(row['expected'])
        assert {e for e,value in values.items() if value==max(values.values())}=={tuple(cut[0]) for cut in json.loads(row['optimalsets'])}
    assert sum(Fraction(1,len(graph)) for v in graph)==1
    assert sum(Fraction(1,len(school)) for v in school)==sum(Fraction(weights[v],sum(weights.values())) for v in school)==1
    checks.append('PASS all source priors normalize exactly; all five cached deterministic school prior optima replayed')
    for mask in range(1<<len(table['edges'])):
        current = nx.Graph()
        current.add_nodes_from(table['nodes'])
        current.add_edges_from(e for i,e in enumerate(table['edges']) if mask>>i&1)
        parts = list(nx.connected_components(current))
        assert sorted(map(len,parts),reverse=True)==table['sizes'][mask]
        for prior in table['values']:
            masses = {v:1 if prior=='class-uniform' else weights[v] for v in school}
            value = sum(weights.values())*sum(masses.values())-sum(sum(masses[v] for v in part)*sum(weights[v] for v in part) for part in parts)
            assert value==table['values'][prior][mask]
    checks.append('PASS all 4096 school component configurations and both weighted objectives match independent NetworkX replay')
    for prior in table['values']:
        values = schoolmeans(table,Fraction(0),prior)
        assert all(value==values[0] for value in values)
        values = schoolmeans(table,Fraction(1),prior)
        p = {v:Fraction(1,len(school)) if prior=='class-uniform' else Fraction(weights[v],sum(weights.values())) for v in school}
        for cuts,value in zip([()]+[(e,) for e in table['edges']],values):
            assert value==highschool.componentvalue(school,cuts,weights,p)
    rng = random.Random(seed+1)
    samples = [[False]*len(plan['links']),[True]*len(plan['links'])]
    samples += [[rng.random()<q for e in plan['links']] for q in (.25,.75) for j in range(4)]
    for active in samples:
        values,sizes = realization(plan,active)
        for removed,value,expected in zip(plan['blocked'],values,sizes):
            current = nx.Graph()
            current.add_nodes_from(range(plan['n']))
            current.add_edges_from(e for i,e in enumerate(plan['links']) if active[i] and i not in removed)
            parts = sorted(map(len,nx.connected_components(current)),reverse=True)
            assert parts==expected and value==plan['n']**2-sum(s*s for s in parts)
    zeros,parts = realization(plan,[False]*len(plan['links']))
    assert all(value==plan['n']*(plan['n']-1) for value in zeros)
    ones,parts = realization(plan,[True]*len(plan['links']))
    for candidate,value in zip(candidates,ones):
        name = candidate['name']
        if name in deterministic: assert Fraction(value,plan['n'])==deterministic[name]
        if candidate['kind']=='partial boundary': assert value==0
    assert Fraction(ones[1],plan['n'])==Fraction(7812,443)
    checks.append('PASS shared-union-find air realizations match independent replay for all 11 interventions in 10 configurations')
    checks.append('PASS q=0 gives zero intervention benefit, with nonzero baseline protection; q=1 matches every cached air baseline and school intervention')
    a = realization(plan,samples[2])
    assert a==realization(plan,samples[2])
    mean,low,high = interval([0,1,2],1)
    assert mean==1 and low<mean<high and interval([0]*1000,1)==(0.0,0.0,0.0)
    assert interval([4]*1000,2)==(2.0,2.0,2.0)
    checks.append('PASS fixed interventions, deterministic replay, nonnegative paired gains and confidence-interval arithmetic')
    return checks

def airexperiment(plan,candidates):
    rows,batches = [],[]
    for q in levels:
        start = perf_counter()
        rng = random.Random(seed)
        probability = float(q)
        samples = [[] for candidate in candidates]
        stats = [dict(countsum=0,countmin=plan['n'],countmax=0,largestsum=0,largestmin=plan['n'],largestmax=0,first=[]) for candidate in candidates]
        target = 1 if q==1 else 1000
        reasons = []
        while True:
            for j in range(len(samples[0]),target):
                active = [True]*len(plan['links']) if q==1 else [rng.random()<probability for e in plan['links']]
                values,parts = realization(plan,active)
                for i,(value,sizes) in enumerate(zip(values,parts)):
                    samples[i].append(value)
                    item = stats[i]
                    item['countsum'] += len(sizes)
                    item['countmin'],item['countmax'] = min(item['countmin'],len(sizes)),max(item['countmax'],len(sizes))
                    item['largestsum'] += sizes[0]
                    item['largestmin'],item['largestmax'] = min(item['largestmin'],sizes[0]),max(item['largestmax'],sizes[0])
                    if not item['first']: item['first'] = sizes
            if target!=1000: break
            for i,candidate in enumerate(candidates):
                if candidate['name'] in ('none','coordinated'): continue
                difference = [u-v for u,v in zip(samples[1],samples[i])]
                mean,low,high = interval(difference,plan['n'],decisioncritical)
                if low<=0<=high and (high-low)/2>precision: reasons.append(candidate['name'])
            if not reasons: break
            target = 3000
        elapsed = perf_counter()-start
        base = sum(samples[0])/(target*plan['n'])
        coordinated = [u-v for u,v in zip(samples[1],samples[0])]
        fullgain = sum(coordinated)/(target*plan['n'])
        for i,candidate in enumerate(candidates):
            gains = [u-v for u,v in zip(samples[i],samples[0])]
            difference = [u-v for u,v in zip(samples[1],samples[i])]
            gain,low,high = interval(gains,plan['n'])
            advantage,adlow,adhigh = interval(difference,plan['n'])
            ignored,delo,dehi = interval(difference,plan['n'],decisioncritical)
            decision = 'same intervention' if i==1 else 'coordinated higher' if delo>0 else 'candidate higher' if dehi<0 else 'unresolved'
            item = stats[i]
            rows.append(dict(experiment='air',q=str(q),qdecimal=float(q),prior='uniform airports',target='unit airports',
                             intervention=candidate['name'],kind=candidate['kind'],cuts=candidate['cuts'],budget=len(candidate['cuts']),
                             method='exact deterministic' if q==1 else 'coupled Monte Carlo',trials=target,
                             mean=sum(samples[i])/(target*plan['n']),meanexact=str(Fraction(samples[i][0],plan['n'])) if q==1 else '',
                             baseline=base,baselineexact=str(Fraction(samples[0][0],plan['n'])) if q==1 else '',
                             gain=gain,gainexact=str(Fraction(gains[0],plan['n'])) if q==1 else '',gainlow=low,gainhigh=high,
                             relativegain=gain/fullgain if fullgain else '',coordinatedadvantage=advantage,advantagelow=adlow,advantagehigh=adhigh,
                             decisionlow=delo,decisionhigh=dehi,comparison=decision,zerogainvariance=len(set(gains))==1,
                             meancomponents=item['countsum']/target,mincomponents=item['countmin'],maxcomponents=item['countmax'],
                             meanlargest=item['largestsum']/target,minlargest=item['largestmin'],maxlargest=item['largestmax'],
                             firstcomponents=item['first'],seconds=elapsed,runtimescope='shared coupled q batch',seed=seed,
                             extendedfor=reasons))
        batches.append(dict(q=str(q),trials=target,seconds=elapsed,extendedfor=reasons))
        print(f'Air q={float(q):g}: {target} realizations, {elapsed:.3f}s; coordinated gain={fullgain:.6f}',flush=True)
    return rows,batches

def schoolexperiment(table,priors):
    rows,summaries = [],[]
    tags = {}
    for row in priors:
        edge = tuple(json.loads(row['cuts'])[0])
        tags.setdefault(edge,[]).append(row['prior'])
    for q in levels:
        for prior in table['values']:
            start = perf_counter()
            values = schoolmeans(table,q,prior)
            order = sorted(range(1,len(values)),key=lambda i:(-values[i],table['edges'][i-1]))
            best,runner = order[:2]
            winners = [i for i in order if values[i]==values[best]]
            gap = values[best]-values[runner]
            summary = dict(q=str(q),prior=prior,baseline=str(values[0]),best=[table['edges'][i-1] for i in winners],
                           expected=str(values[best]),gain=str(values[best]-values[0]),runner=table['edges'][runner-1],gap=str(gap))
            summaries.append(summary)
            elapsed = perf_counter()-start
            for i,value in enumerate(values):
                edge = table['edges'][i-1] if i else None
                gain = value-values[0]
                assert gain>=0
                rows.append(dict(experiment='school',q=str(q),qdecimal=float(q),prior=prior,target='class population',
                                 intervention='--'.join(edge) if edge else 'none',kind='single edge' if edge else 'baseline',cuts=(edge,) if edge else (),
                                 budget=int(bool(i)),method='exact enumeration',trials=1<<len(table['edges']),mean=float(value),meanexact=str(value),
                                 baseline=float(values[0]),baselineexact=str(values[0]),gain=float(gain),gainexact=str(gain),
                                 isbest=i in winners,bestedges=summary['best'],bestexact=summary['expected'],bestgainexact=summary['gain'],
                                 runneredge=summary['runner'],bestsecondgap=float(gap),bestsecondgapexact=str(gap),
                                 cachedcandidatepriors=tags.get(edge,[]),seconds=elapsed,runtimescope='shared q/prior exact expectation evaluation'))
            print(f'School q={float(q):g} {prior}: best={summary["best"]}, gain={float(Fraction(summary["gain"])):.6f}, gap={float(gap):.6f}',flush=True)
    return rows,summaries

def report(rows,batches,summaries,checks,provenance,table,elapsed,enumerationseconds):
    lines = ['STOCHASTIC TRANSMISSION ROBUSTNESS',
             'Independent bond percolation on original undirected edges. Interventions are fixed BEFORE drawing any active edges.',
             'Each realization integrates over the source prior exactly: Q=a(V)-sum p(C)a(C). Source trials are not sampled separately.',
             'Gain means E[Q(F)]-E[Q(empty)], not total expected protection. Priors and target weights stay fixed across realizations.',
             'Air: 443 airports, 7160 routes, tau=1; uniform source, unit targets. Only the cached fixed interventions are compared.',
             'A deterministic optimal cut need not be stochastically optimal; no stochastic global optimality claim is made for air.',
             'School: 9 classes, 12 edges with duration >=5300 seconds, 329 metadata students including two without contact rows.',
             'School target weights are class population under BOTH source priors.',
             'No original code, result, figure or manuscript file is modified. No BC ranking or measured static optimization is rerun.',
             '', 'METHOD AND UNCERTAINTY',
             f'Python {platform.python_version()}, NetworkX {nx.__version__}; random.Random seed={seed}.',
             'Canonical sorted original edges determine RNG order. For each stochastic q the same seed is reset, coupling q levels as well as interventions.',
             'Within a trial every intervention uses the identical original-edge activity vector. Active cut edges are then suppressed.',
             'Union-find first merges active edges outside the union of all intervention sets; each intervention gets a copy and adds its surviving active special edges.',
             'This reuse is exact because all interventions share those common active edges. Vertices, including isolates, are never dropped.',
             'Air q<1 starts with 1000 trials. Requested gain CIs are pointwise paired normal 95% intervals: mean +/- 1.959964 SE.',
             'They are Monte Carlo approximations, not exact finite-sample guarantees; intervals are not clipped at zero.',
             'For comparison decisions, 97.5%-per-look paired normal intervals allow at most two looks (a union-bound approximate 95% guarantee per comparison).',
             'There is no simultaneous error guarantee across all q values or all intervention comparisons.',
             f'Extend a q batch to 3000 coupled trials only if a full-versus-baseline/partial comparison straddles zero with decision halfwidth >{precision:.2f} airport.',
             'After an extension the first 1000 trials are retained and 2000 new trials continue the same RNG stream.',
             'A comparison is unresolved unless its decision interval excludes zero; zero observed variance does not prove zero true stochastic benefit.',
             'Ratios are ratios of estimated incremental gains, not mean trial ratios and not achieved-to-stochastic-OPT ratios.',
             'At q=1 the single full-edge configuration is evaluated exactly; its degenerate intervals are deterministic identities.',
             'School enumerates all 4096 ORIGINAL edge configurations, including edges later removed. Probabilities are q^k(1-q)^(12-k).',
             'Integer coefficient sums by active-edge count and Fraction arithmetic give exact expectations for every tested rational q.',
             'All school best/runner-up differences are exact. No school Monte Carlo confidence intervals are used.',
             'Air runtimes are shared per-q coupled batches, not additive per-intervention timings.',
             f'School enumeration/coefficient construction: {enumerationseconds:.6f}s.','', 'VALIDATION',*checks,
             f'Air q=0: total expected protection 442, intervention gain 0 for every cut.']
    for prior in table['values']:
        values = schoolmeans(table,Fraction(0),prior)
        lines.append(f'School q=0 {prior}: expected protection {values[0]} ({float(values[0]):.9f}); intervention gains all 0.')
    lines += ['', 'PROVENANCE',json.dumps(provenance,sort_keys=True),
              'run.py SHA256: '+digest(__file__),
              'Command: PYTHONDONTWRITEBYTECODE=1 python code/transmission.py',
              'Validation-only command: PYTHONDONTWRITEBYTECODE=1 python code/transmission.py --check',
              '', 'AIR FIXED INTERVENTIONS']
    for row in rows:
        if row['experiment']=='air' and row['q']=='1':
            lines.append(f'{row["intervention"]}: {row["cuts"]}; deterministic Q={row["meanexact"]}; sizes={row["firstcomponents"]}')
    lines += ['', 'AIR RESULTS',
              'q | intervention | mean Q | gain [paired 95% CI] | gain/coordinated gain | coordinated advantage [paired 95% CI] | decision',
              'Positive coordinated advantage means the complete Washington boundary is better.']
    for row in rows:
        if row['experiment']!='air': continue
        lines.append(f'{row["qdecimal"]:.2f} | {row["intervention"]} | {row["mean"]:.6f} | {row["gain"]:.6f} [{row["gainlow"]:.6f}, {row["gainhigh"]:.6f}] | '
                     f'{row["relativegain"]:.6f} | {row["coordinatedadvantage"]:.6f} [{row["advantagelow"]:.6f}, {row["advantagehigh"]:.6f}] | {row["comparison"]}')
    lines += ['', 'AIR COMPONENT SANITY',
              'Every realized component-size list sums to 443, and n-sum(size squared)/n matches the union-find score exactly.',
              'q | intervention | component count mean[min,max] | largest component mean[min,max]']
    for row in rows:
        if row['experiment']=='air':
            lines.append(f'{row["qdecimal"]:.2f} | {row["intervention"]} | {row["meancomponents"]:.3f}[{row["mincomponents"]},{row["maxcomponents"]}] | '
                         f'{row["meanlargest"]:.3f}[{row["minlargest"]},{row["maxlargest"]}]')
    lines += ['', 'AIR SAMPLE SIZES AND RUNTIMES']+[json.dumps(batch) for batch in batches]
    lines += ['', 'SCHOOL PRIOR DISTINCTION',
              'Cached student-uniform means probability proportional to class population, not uniform over classes.',
              'At q=1, student-uniform and class-uniform BOTH prefer PC--PSI*. They do not exhibit a deterministic prior switch.',
              'The cached comparison includes three concentrated priors: half PC, half 2BIO3, half PC*.',
              'For a half X prior, p(X)=1/2 and the remaining half is distributed in proportion to population outside X.',
              'The cached half 2BIO3 optimum is 2BIO3--PC; half PC* selects PC--PC*. Thus the cache has THREE distinct optimal bridges, not only two.',
              'All three are included among the 12 tested single-edge interventions under both requested stochastic priors.',
              'No concentrated-prior stochastic experiment is silently substituted for the requested two priors.',
              'Class populations: '+str(provenance['populations']),
              'Retained class edges: '+str(table['edges']),
              '', 'SCHOOL EXACT BEST AND SECOND-BEST',
              'q | source prior | baseline | best edge(s) | expected protected students | incremental gain | runner-up | best-minus-second']
    for item in summaries:
        lines.append(f'{float(Fraction(item["q"])):.2f} | {item["prior"]} | {item["baseline"]} | {item["best"]} | '
                     f'{item["expected"]} ({float(Fraction(item["expected"])):.9f}) | {item["gain"]} ({float(Fraction(item["gain"])):.9f}) | '
                     f'{item["runner"]} | {item["gap"]} ({float(Fraction(item["gap"])):.9f})')
    lines += ['', 'SCHOOL EVERY SINGLE-EDGE INTERVENTION', 'q | source prior | intervention | exact expected students | exact incremental benefit | optimal']
    for row in rows:
        if row['experiment']=='school':
            lines.append(f'{row["qdecimal"]:.2f} | {row["prior"]} | {row["intervention"]} | {row["meanexact"]} ({row["mean"]:.9f}) | '
                         f'{row["gainexact"]} ({row["gain"]:.9f}) | {row["isbest"]}')
    lines += ['', 'INTERPRETATION', '1. Coordinated airport cut versus fixed algorithmic baselines:']
    for q in levels:
        current = [row for row in rows if row['experiment']=='air' and row['q']==str(q)]
        selected = [row for row in current if row['intervention'] in ('initial','recomputed','greedy')]
        lines.append(f'q={float(q):g}: '+ '; '.join(f'{row["intervention"]}: {row["comparison"]}, advantage={row["coordinatedadvantage"]:.6f}, decision interval=[{row["decisionlow"]:.6f},{row["decisionhigh"]:.6f}]' for row in selected))
    lines.append('2. Completing the boundary versus every proper nonempty subset:')
    for q in levels:
        selected = [row for row in rows if row['experiment']=='air' and row['q']==str(q) and row['kind']=='partial boundary']
        unresolved = [row['intervention'] for row in selected if row['comparison']!='coordinated higher']
        tight = min(selected,key=lambda row:row['coordinatedadvantage'])
        lines.append(f'q={float(q):g}: '+('higher than all six partial cuts' if not unresolved else 'not established for '+str(unresolved))+
                     f'; smallest full-minus-partial gain={tight["coordinatedadvantage"]:.6f}, paired 95% CI=[{tight["advantagelow"]:.6f},{tight["advantagehigh"]:.6f}] ({tight["intervention"]}).')
    lines.append('3. Source-prior dependence for the requested school priors:')
    for q in levels:
        selected = [item for item in summaries if item['q']==str(q)]
        different = selected[0]['best']!=selected[1]['best']
        lines.append(f'q={float(q):g}: preferred-edge sets '+('DIFFER' if different else 'agree')+'; '+str([(item['prior'],item['best']) for item in selected]))
    lines += ['4. Persistence, weakening, reversals and scope:',
              'Air comparisons above concern incremental intervention benefit; substantial baseline stochastic protection is not credited to edge removal.',
              'At q=1 the fixed BC and partial-boundary cuts have zero benefit. For q<1 inspect their positive gains: random failures can supply missing parts of a cut.',
              'The complete boundary and individual rankings are compared on the same realizations, so differences do not reflect separately sampled networks.',
              'School switches across q and across priors are reported explicitly above; the cached concentrated-prior switch is a different comparison.',
              'No result establishes that the coordinated air cut is globally optimal under percolation.',
              '5. Model limitations:',
              'Bond percolation specifies a static set of transmitting edges and terminal reachability. It does not model recovery times, exposure timing, repeated contacts, temporal ordering or dynamic intervention.',
              'Uniform independent q ignores heterogeneous contact durations/passenger volumes and correlations between edge activations.',
              'A class is a single fully affected unit when reached; within-class stochastic spread is not modeled.',
              'These are robustness tests of static reachability interventions, not simulations or validations of full SIR dynamics.',
              'Normal Monte Carlo intervals may miss rare events when observed variance is zero; school enumeration and q=1 checks have no Monte Carlo uncertainty.',
              '',f'Total rows: {len(rows)}; air={sum(row["experiment"]=="air" for row in rows)}; school={sum(row["experiment"]=="school" for row in rows)}.',
              f'Total runtime: {elapsed:.3f}s.',
              'Script: code/transmission.py; outputs: data/results/transmission.csv and its entry in data/records.json are created.']
    return '\n'.join(lines)+'\n'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    outputs = [folder/'transmission.csv']
    if not args.check and (any(path.exists() for path in outputs) or savedreports.exists('transmission')): raise FileExistsError('Refusing to overwrite results; --check validates without writing')
    start = perf_counter()
    graph,candidates,deterministic,school,weights,priors,provenance = sources()
    plan = airplan(graph,candidates)
    tick = perf_counter()
    table = schooltable(school,weights)
    enumerationseconds = perf_counter()-tick
    checks = validate(graph,candidates,deterministic,school,weights,priors,plan,table)
    for line in checks: print(line,flush=True)
    if args.check: return
    rows,batches = airexperiment(plan,candidates)
    schoolrows,summaries = schoolexperiment(table,priors)
    rows += schoolrows
    assert len(rows)==185
    assert signature(graph)==provenance['airgraph'] and signature(school)==provenance['schoolgraph']
    assert all(hashlib.sha256(savedreports.data(path)).hexdigest()==value for path,value in provenance['files'].items())
    checks.append('PASS all input-file hashes and graph topologies unchanged after the full experiment')
    text = report(rows,batches,summaries,checks,provenance,table,perf_counter()-start,enumerationseconds)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with outputs[0].open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,separators=(',',':')) if isinstance(value,(dict,list,tuple)) else value for key,value in row.items()})
    savedreports.save('transmission',text,exclusive=True)
    print(f'Completed {len(rows)} records in {perf_counter()-start:.3f}s',flush=True)
    for path in outputs: print(path)

if __name__=='__main__':
    main()
