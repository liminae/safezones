# Class-level SocioPatterns application
import records as savedreports
import argparse
import csv
import gzip
import hashlib
import json
import math
import tempfile
from collections import Counter,defaultdict
from datetime import datetime,timezone
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parent.parent
policies = ('weak','strong','bc','bcw','source-aware','threshold')
page = 'https://sociopatterns.org/datasets/high-school-contact-and-friendship-networks/'

def readraw():
    raw = root/'data/raw'
    manifest = json.loads(savedreports.data('source:highschool'))
    for row in manifest:
        assert hashlib.sha256((raw/row['file']).read_bytes()).hexdigest()==row['sha256']
    meta = {}
    for line in (raw/'HighSchool2013_metadata.txt').read_text().splitlines():
        student,group,gender = line.split()
        student = int(student)
        assert student not in meta
        meta[student] = group
    weights = dict(sorted(Counter(meta.values()).items()))
    contacts,times = set(),Counter()
    within,between,embedded = Counter(),Counter(),Counter()
    daily = defaultdict(Counter)
    active = defaultdict(set)
    daytimes = defaultdict(set)
    mismatches = Counter()
    seen = set()
    paris = ZoneInfo('Europe/Paris')
    with gzip.open(raw/'HighSchool2013_proximity_net.csv.gz','rt') as stream:
        for line in stream:
            fields = line.split()
            assert len(fields)==5
            t,u,v = map(int,fields[:3])
            assert u!=v and t%20==0
            assert u in meta and v in meta
            key = t,min(u,v),max(u,v)
            assert key not in seen
            seen.add(key)
            for student,label in zip((u,v),fields[3:]):
                if label!=meta[student]: mismatches[student,label,meta[student]] += 1
            start = datetime.fromtimestamp(t-20,paris)
            end = datetime.fromtimestamp(t-1,paris)
            assert start.date()==end.date()
            day = start.date().isoformat()
            a,b = sorted((meta[u],meta[v]))
            daily[day][a,b] += 20
            active[day].update((u,v))
            daytimes[day].add(t)
            if a==b: within[a] += 20
            else: between[a,b] += 20
            embedded[tuple(sorted(fields[3:]))] += 20
            contacts.update((u,v))
            times[t] += 1
    ordered = sorted(times)
    assert all((b-a)%20==0 for a,b in zip(ordered,ordered[1:]))
    total = 20*sum(times.values())
    assert total==sum(within.values())+sum(between.values())
    embeddedwithin = sum(value for (a,b),value in embedded.items() if a==b)
    differing = []
    for edge in sorted(set(between)|{e for e in embedded if e[0]!=e[1]}):
        if between[edge]!=embedded[edge]:
            differing.append(dict(edge=edge,metadata=between[edge],embedded=embedded[edge]))
    audit = dict(manifest=manifest,metadata=len(meta),contactstudents=len(contacts),weights=weights,
                 metadataonly=sorted(set(meta)-contacts),contactonly=sorted(contacts-set(meta)),
                 rows=sum(times.values()),timestamps=len(times),days=len(daily),
                 firstutc=datetime.fromtimestamp(min(times),timezone.utc).isoformat(),
                 lastutc=datetime.fromtimestamp(max(times),timezone.utc).isoformat(),
                 firstlocal=datetime.fromtimestamp(min(times),paris).isoformat(),
                 lastlocal=datetime.fromtimestamp(max(times),paris).isoformat(),
                 intervalstart=datetime.fromtimestamp(min(times)-20,paris).isoformat(),
                 totalseconds=total,withinseconds=sum(within.values()),crossseconds=sum(between.values()),
                 withinpercent=100*sum(within.values())/total,crosspercent=100*sum(between.values())/total,
                 withinbyclass=dict(within),embeddedwithin=embeddedwithin,
                 embeddedcross=total-embeddedwithin,changededges=differing,
                 mismatches=[dict(student=student,contactlabel=old,metadatalabel=new,rows=count)
                             for (student,old,new),count in mismatches.items()],
                 daily=[dict(day=day,rows=sum(values.values())//20,active=len(active[day]),
                             activeclasses=sorted({meta[v] for v in active[day]}),
                             activemembers=dict(sorted(Counter(meta[v] for v in active[day]).items())),
                             timestamps=len(daytimes[day]),
                             start=datetime.fromtimestamp(min(daytimes[day])-20,paris).isoformat(),
                             end=datetime.fromtimestamp(max(daytimes[day]),paris).isoformat(),
                             within=sum(v for (a,b),v in values.items() if a==b),
                             cross=sum(v for (a,b),v in values.items() if a!=b))
                        for day,values in sorted(daily.items())])
    return weights,between,daily,audit

def makegraph(weights,strengths,theta=0):
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from((v,dict(population=weights[v])) for v in sorted(weights))
    graph.add_edges_from((a,b,dict(strength=value)) for (a,b),value in sorted(strengths.items())
                         if a!=b and value>=theta)
    return graph

def geometry(graph):
    import networkx as nx
    from safezones import edge_order
    components = sorted((sorted(c) for c in nx.connected_components(graph)),key=lambda c:c[0])
    connected = len(components)==1
    return dict(nodes=len(graph),edges=graph.number_of_edges(),density=nx.density(graph),connected=connected,
                components=components,componentcount=len(components),
                connectivity=nx.edge_connectivity(graph) if connected else None,
                bridges=sorted(tuple(sorted(e)) for e in nx.bridges(graph)),
                articulations=sorted(nx.articulation_points(graph)),
                rank=graph.number_of_edges()-len(graph)+len(components),
                meandegree=2*graph.number_of_edges()/len(graph),edgelist=edge_order(graph))

def valuesandbc(graph):
    from safezones import edge_order,centralities
    values = {e:graph.edges[e]['strength'] for e in edge_order(graph)}
    bc,bcw = centralities(graph,values)
    return values,bc,bcw

def componentvalue(graph,cuts,weights,prior):
    import networkx as nx
    remaining = graph.copy()
    remaining.remove_edges_from(cuts)
    return sum(weights.values())-sum(sum(prior[v] for v in group)*sum(weights[v] for v in group)
                                    for group in nx.connected_components(remaining))

def identity(graph,cuts,weights,prior):
    import networkx as nx
    assert set(graph)==set(weights)==set(prior) and sum(prior.values())==1
    assert all(value>=0 for value in prior.values()) and all(value>0 for value in weights.values())
    remaining = graph.copy()
    remaining.remove_edges_from(cuts)
    component = componentvalue(graph,cuts,weights,prior)
    direct = sum(prior[s]*sum(weights[v] for v in graph if v not in nx.node_connected_component(remaining,s)) for s in graph)
    cut = sum(prior[s]*weights[t] for s in graph for t in graph if not nx.has_path(remaining,s,t))
    assert component==direct==cut
    return component

def weightedexact(graph,weights):
    from safezones import edge_order,first_spread,spread
    nodes = tuple(sorted(graph))
    original = edge_order(graph)
    choices = {}
    @lru_cache(None)
    def best(infected,remaining):
        if spread(nodes,remaining,infected)==infected:
            return sum(weights[v] for v in nodes if v not in infected)
        actions = [None]+[e for e in remaining if not all(v in infected for v in e)]
        value = -1
        for action in actions:
            surviving = tuple(e for e in remaining if e!=action)
            after = spread(nodes,surviving,infected)
            candidate = best(after,surviving)
            if candidate>value:
                value = candidate
                choices[infected,remaining] = action
        return value
    def solve(source):
        infected = first_spread(graph,source)
        remaining = original
        value = best(infected,remaining)
        actions = []
        while spread(nodes,remaining,infected)!=infected:
            action = choices[infected,remaining]
            actions.append(action)
            remaining = tuple(e for e in remaining if e!=action)
            infected = spread(nodes,remaining,infected)
        return value,actions
    solve.cache = best.cache_info
    return solve

def replay(graph,source,actions,weights):
    from safezones import edge_order,first_spread,spread
    nodes = tuple(sorted(graph))
    remaining = edge_order(graph)
    infected = first_spread(graph,source)
    history = [sorted({source}),sorted(infected)]
    details = []
    for step,action in enumerate(actions,1):
        assert spread(nodes,remaining,infected)!=infected
        before = infected
        if action is not None:
            assert action in remaining
            remaining = tuple(e for e in remaining if e!=action)
        infected = spread(nodes,remaining,infected)
        details.append(dict(step=step,before=sorted(before),action=action,after=sorted(infected)))
        if infected!=before: history.append(sorted(infected))
    assert spread(nodes,remaining,infected)==infected
    safe = sorted(set(graph)-infected)
    return sum(weights[v] for v in safe),safe,history,details

def weightedpolicy(graph,source,policy,weights,values,bc,bcw):
    from safezones import simulate
    count,cuts,states = simulate(graph,source,policy,values,bc,bcw,trace=True)
    actions = [action for after,action in states[2:]]
    result = replay(graph,source,actions,weights)
    assert count==len(result[1])
    return result[0],actions

def cutdetails(graph,cuts,weights,safe=None):
    import networkx as nx
    remaining = graph.copy()
    remaining.remove_edges_from(cuts)
    membership = {v:i for i,c in enumerate(nx.connected_components(remaining)) for v in c}
    bridges = {tuple(sorted(e)) for e in nx.bridges(graph)}
    return [dict(classes=e,sizes=[weights[v] for v in e],seconds=graph.edges[e]['strength'],
                 minutes=graph.edges[e]['strength']/60,originalbridge=e in bridges,
                 separatesfinalcomponents=membership[e[0]]!=membership[e[1]],
                 finalsafeendpoints=sum(v in safe for v in e) if safe is not None else None)
            for e in cuts]

def thresholdlevels(weights,strengths):
    durations = sorted(strengths.values())
    reasons = defaultdict(list)
    reasons[0].append('unthresholded week')
    for q in (.25,.5,.75,1):
        reasons[durations[math.ceil(q*len(durations))-1]].append(f'observed nearest-rank quantile {q:g}')
    previous = geometry(makegraph(weights,strengths))
    connectedlimit = 0
    firstbridge = None
    firstforest = None
    for value in sorted(set(durations)):
        current = geometry(makegraph(weights,strengths,value+1))
        if previous['connected'] and current['connected'] and current['connectivity']!=previous['connectivity']:
            reasons[value+1].append('first integer threshold at edge connectivity '+str(current['connectivity']))
        if current['bridges'] and firstbridge is None:
            firstbridge = value+1
            reasons[value+1].append('first bridges')
        if previous['connected'] and current['connected'] and current['bridges']!=previous['bridges']:
            reasons[value+1].append('connected bridge set changes')
        if previous['connected'] and not current['connected']:
            connectedlimit = value
            reasons[value].append('largest connected threshold')
            reasons[value+1].append('first disconnection')
        if current['rank']==0 and firstforest is None:
            firstforest = value+1
            reasons[value+1].append('first forest')
        previous = current
    reasons[max(durations)+1].append('no edges remain')
    return dict(sorted(reasons.items())),connectedlimit,firstbridge,firstforest

def priors(weights,strengths):
    total = sum(weights.values())
    degrees = {v:sum(value for e,value in strengths.items() if v in e) for v in weights}
    largest = min(weights,key=lambda v:(-weights[v],v))
    central = min(weights,key=lambda v:(-degrees[v],v))
    peripheral = min(weights,key=lambda v:(degrees[v],v))
    result = {'student-uniform':{v:Fraction(w,total) for v,w in weights.items()},
              'class-uniform':{v:Fraction(1,len(weights)) for v in weights}}
    for v in dict.fromkeys((largest,central,peripheral)):
        result['half '+v] = {u:Fraction(1,2) if u==v else Fraction(w,2*(total-weights[v])) for u,w in weights.items()}
    return result,dict(largest=largest,central=central,peripheral=peripheral,weighteddegree=degrees)

def staticrecords(graph,weights,prior,theta,budget,scope='intervention',priorname='student-uniform'):
    import networkx as nx
    from safezones import edge_order
    edges = edge_order(graph)
    candidates = list(combinations(edges,budget))
    scores = {cuts:componentvalue(graph,cuts,weights,prior) for cuts in candidates}
    optimum = max(scores.values())
    optimal = [cuts for cuts,value in scores.items() if value==optimum]
    values,bc,bcw = valuesandbc(graph)
    orders = {'bc':sorted(edges,key=lambda e:(-bc[e],e)),
              'bcw':sorted(edges,key=lambda e:(-bcw[e],e)),
              'strong':sorted(edges,key=lambda e:(-values[e],e)),
              'weak':sorted(edges,key=lambda e:(values[e],e))}
    selected = {'exact':optimal[0]}
    if scope=='intervention': selected.update({name:tuple(sorted(order[:budget])) for name,order in orders.items()})
    rows = []
    for name,cuts in selected.items():
        value = scores[cuts]
        assert identity(graph,cuts,weights,prior)==value
        remaining = graph.copy()
        remaining.remove_edges_from(cuts)
        components = sorted((sorted(c) for c in nx.connected_components(remaining)),key=lambda c:c[0])
        sizes = [sum(weights[v] for v in c) for c in components]
        reason = 'no component separation; no protected population'
        if value:
            reason = 'bridge separation' if any(e in geometry(graph)['bridges'] for e in cuts) else 'joint cut; no individual bridge required'
        rows.append(dict(scope=scope,theta=theta,budget=budget,prior=priorname,method=name,
                         expected=value,expecteddecimal=float(value),optimum=optimum,optimumdecimal=float(optimum),
                         ratio=Fraction(value,optimum) if optimum else None,cuts=cuts,
                         optimumties=len(optimal),evaluated=len(candidates),
                         optimalsets=optimal if name=='exact' and optimum else [],
                         priorvalues={v:str(p) for v,p in prior.items()},components=components,componentweights=sizes,
                         priorcomponents=[str(sum(prior[v] for v in c)) for c in components],
                         removedseconds=sum(values[e] for e in cuts),details=cutdetails(graph,cuts,weights),reason=reason))
    return rows

def sequentialrecords(graph,weights,prior,theta):
    from safezones import edge_order
    original = edge_order(graph)
    values,bc,bcw = valuesandbc(graph)
    solve = weightedexact(graph,weights)
    rows = []
    optimal = {}
    for source in sorted(graph):
        optimum,actions = solve(source)
        optimal[source] = optimum
        for name in ('exact',)+policies:
            value,sequence = (optimum,actions) if name=='exact' else weightedpolicy(graph,source,name,weights,values,bc,bcw)
            protected,safe,history,trace = replay(graph,source,sequence,weights)
            assert value==protected and value<=optimum
            cuts = [e for e in sequence if e is not None]
            details = cutdetails(graph,cuts,weights,safe)
            for detail,step in zip(details,[step for step in trace if step['action'] is not None]):
                detail.update(step=step['step'],before=step['before'],
                              timing='threshold' if sum(v in step['before'] for v in step['action'])==1 else 'proactive',
                              role=('both endpoints eventually infected','final infected-safe boundary','both endpoints remain safe')[detail['finalsafeendpoints']])
            rows.append(dict(kind='source',theta=theta,source=source,prior=prior[source],method=name,
                             value=value,optimum=optimum,ratio=Fraction(value,optimum) if optimum else None,
                             optimal=value==optimum,safe=safe,actions=sequence,cuts=cuts,history=history,trace=trace,
                             removedseconds=sum(values[e] for e in cuts),details=details))
    for name in ('exact',)+policies:
        selected = [r for r in rows if r['method']==name]
        positive = [r for r in selected if r['optimum']]
        mass = sum(prior[r['source']] for r in positive)
        expected = sum(prior[r['source']]*r['value'] for r in selected)
        exactexpected = sum(prior[s]*optimal[s] for s in graph)
        rows.append(dict(kind='summary',theta=theta,method=name,expected=expected,expecteddecimal=float(expected),
                         exactexpected=exactexpected,exactexpecteddecimal=float(exactexpected),
                         meanratio=sum(prior[r['source']]*r['ratio'] for r in positive)/mass if positive else None,
                         unweightedmeanratio=sum(r['ratio'] for r in positive)/len(positive) if positive else None,
                         expectedratio=expected/exactexpected if exactexpected else None,
                         worstratio=min((r['ratio'] for r in positive),default=None),
                         optimalsources=sum(r['optimal'] for r in selected),sources=len(graph),
                         optimalrate=Fraction(sum(r['optimal'] for r in selected),len(graph)),
                         optimalpriormass=sum(prior[r['source']] for r in selected if r['optimal']),
                         nontrivial=len(positive),zeros=len(graph)-len(positive),nontrivialmass=mass,
                         states=solve.cache().currsize))
    assert edge_order(graph)==original
    return rows

def writecsv(path,rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(tuple,list,dict)) else v for k,v in row.items()})

def edgerecords(graph,weights,period):
    values,bc,bcw = valuesandbc(graph)
    return [dict(period=period,left=e[0],right=e[1],leftsize=weights[e[0]],rightsize=weights[e[1]],
                 seconds=value,minutes=value/60,intervals=value//20,bcstored=bc[e],bcordered=2*bc[e],
                 bcwstored=bcw[e],bcwordered=2*bcw[e]) for e,value in sorted(values.items())]

def report(audit,week,daily,thresholds,static,sequential,selection,limit,firstbridge,forest,elapsed):
    weights = audit['weights']
    lines = ['SOCIOPATTERNS HIGH SCHOOL 2013: CLASS-LEVEL APPLICATION',f'Seconds: {elapsed:.3f}',
             'Source: '+page,'Documentation: each row is an active undirected contact interval [t-20,t], UNIX seconds.',
             'Reference: Mastrandrea, Fournet and Barrat, PLoS ONE 10(9), e0136497 (2015).',
             'Dataset terms: Creative Commons Attribution-NonCommercial-ShareAlike; see official source.',
             'Only proximity contacts and metadata used. Raw files preserved; no friendship, Facebook or diary inputs.',
             '', 'A. DATASET AUDIT']
    for record in audit['manifest']: lines.append(json.dumps(record,sort_keys=True))
    for key in ('metadata','contactstudents','weights','metadataonly','contactonly','rows','timestamps','days',
                'firstutc','lastutc','firstlocal','lastlocal','intervalstart','totalseconds','withinseconds',
                'crossseconds','withinpercent','crosspercent','withinbyclass','mismatches','embeddedwithin',
                'embeddedcross','changededges'):
        lines.append(key+': '+json.dumps(audit[key],sort_keys=True))
    lines += ['All records have five fields, no self-contact or repeated unordered pair/timestamp, and timestamps on the20-second grid.',
              'Duration20 seconds is specified by the documentation, not an independently measured duration column.',
              'Durations are summed pair-contact seconds; concurrent pairs accumulate separately, not elapsed observation time.',
              'Class membership is taken from metadata throughout. Metadata329 students, including two without contact rows, defines target weights and priors.',
              'Day bins are Europe/Paris (CET in December2013). No interval crosses local midnight.',
              '', 'B. WEEK CLASS GRAPH',json.dumps(week,sort_keys=True),
              'All36 edges have ordinary ordered-pair BC=2; inverse-duration weighted BC is listed below.',
              'edge | students | contact seconds | ordered BC | ordered BCw']
    with (root/'data/results/highschooledges.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['period']=='week':
                lines.append(f"{row['left']}--{row['right']} | {row['leftsize']},{row['rightsize']} | {row['seconds']} | {row['bcordered']} | {row['bcwordered']}")
    lines += ['', 'C. DAILY VARIATION',
              'Every daily graph keeps all9 metadata classes and the same class weights; activity is reported separately.']
    for row in daily: lines.append(json.dumps(row,sort_keys=True))
    lines += ['', 'D. THRESHOLD SWEEP',
              'Retain duration>=theta. Quantiles use observed edge strengths and nearest-rank quartiles; add connectivity/bridge changes, forest onset, and empty graph.',
              'A first-integer breakpoint d+1 represents exclusion of a duration-d tie; continuous theta>d gives the same transition.',
              f'Largest connected theta={limit}s ({limit/60:.6f}min). First bridges at integer theta={firstbridge}s.',
              f'First forest at theta={forest}s, already disconnected. There is no connected tree baseline in this sweep.',
              'Disconnected levels are descriptive only; no static intervention or sequential optimization is applied to them.',
              'Rank=m-n+components. Filtered contact time is baseline preprocessing, separate from intervention contact loss.',
              'theta seconds | minutes | edges | density | components | connectivity | bridges | rank | mean degree | filtered seconds']
    for row in thresholds:
        lines.append(f"{row['theta']} | {row['minutes']:.4f} | {row['edges']} | {row['density']:.4f} | "
                     f"{row['components']} | {row['connectivity']} | {row['bridges']} | {row['rank']} | {row['meandegree']:.4f} | {row['filteredseconds']}")
    lines += ['', 'E. WEIGHTED IDENTITY',
              'W=sum(class sizes)=329; student-uniform p(c)=W(c)/329.',
              'Component formula W-sum_i p(C_i)W(C_i), direct source averaging, and sum_s,t p(s)W(t)1(disconnected) agree using Fraction.',
              'Identity checked for the uncut baseline and every reported static intervention/prior optimum, plus standalone regression fixtures.',
              'This is an application-only weighted-target extension, not a change to the manuscript theorem.',
              '', 'F. STATIC INTERVENTIONS',
              'All single edges and all distinct pairs evaluated. BC/BCw/strong/weak select the initial top b individual edges, without recomputation.',
              'Only connected theta levels evaluated. When OPT=0 ratios are NA and all budget-feasible sets are tied; no useful intervention is identified.',
              'Full components, weights, priors, alternative optimal sets, contact losses and edge details are in highschoolstatic.csv.',
              'theta | budget | method | expected students (exact; decimal) | ratio | removed class pairs | contact seconds | reason']
    for row in static:
        if row['scope']=='intervention':
            lines.append(f"{row['theta']} | {row['budget']} | {row['method']} | {row['expected']}; {row['expecteddecimal']:.6f} | "
                         f"{row['ratio']} | {row['cuts']} | {row['removedseconds']} | {row['reason']}")
    lines += ['', 'G. WEIGHTED SEQUENTIAL APPLICATION',
              'One mandatory first spread. Then one permanent removal or wait before each simultaneous spread. Proactive cuts allowed; no global budget.',
              'Application-only exact search memoizes infected set and surviving edges; terminal value sums protected student weights.',
              'Policies call production simulate with original retained-graph BC and inverse-strength BCw; only terminal population evaluation changes.',
              'Centralities are computed once per baseline and never recomputed during contagion. Production functions are unchanged.',
              'Expected students uses class-size prior. Mean ratio is prior-weighted conditional on OPT>0; worst ratio also excludes OPT0.',
              'Exact-optimality count includes all9 sources, so all-zero baselines trivially score9/9. Extra unweighted means and ratios of expectations are in the CSV.',
              'theta | policy | expected students | conditional mean ratio | worst ratio | exactly optimal /9 | OPT0 sources']
    for row in sequential:
        if row['kind']=='summary':
            mean = f"{float(row['meanratio']):.6f}" if row['meanratio'] is not None else 'NA'
            worst = f"{float(row['worstratio']):.6f}" if row['worstratio'] is not None else 'NA'
            lines.append(f"{row['theta']} | {row['method']} | {row['expected']} = {row['expecteddecimal']:.6f} | "
                         f"{mean} | {worst} | {row['optimalsources']}/9 | {row['zeros']}")
    lines += ['', 'Exact sequential source solutions (all zero-valued cases have no useful intervention):']
    for row in sequential:
        if row['kind']=='source' and row['method']=='exact':
            lines.append(f"theta={row['theta']} source={row['source']} protected={row['value']} safe={row['safe']} actions={row['actions']} contactloss={row['removedseconds']}s details={json.dumps(row['details'])}")
    lines += ['', 'H. SOURCE-PRIOR SENSITIVITY',json.dumps(selection,sort_keys=True),
              '50% prior assigns half to the selected class and the remainder proportional to sizes of other classes.',
              'Class choices use week weighted degree, not a recomputed ranking at each theta.',
              'theta | prior | best single-edge expectation | representative edge | number of tied optimal edges']
    for row in static:
        if row['scope']=='prior':
            lines.append(f"{row['theta']} | {row['prior']} | {row['expected']} = {row['expecteddecimal']:.6f} | {row['cuts']} | {row['optimumties']}")
    lines += ['', 'I. INTERPRETATION',
              'An illustrative class-separation intervention suppresses mixing between the two named classes, such as shared activities or schedule overlap.',
              'Static bridges separate entire student-weighted components; pairs can be valuable jointly even when neither edge is a bridge.',
              'Sequential CSV traces identify frontier/proactive timing and final safe endpoints for every removed edge, including all optimal source solutions.',
              'Contact loss is cumulative observed cross-class pair-contact seconds on the removed ties; not money, elapsed closure time, or a feasibility estimate.',
              'Static optimization chooses one source-blind cut set with budget1 or2. Sequential optimization observes the source and has no global removal budget.',
              'Their expectations therefore compare different information and resource constraints, not timing alone.',
              'No public-health recommendation is made.',
              '', 'J. FIGURES',
              'figures/exploratory/highschool/contacts.pdf: week cross-class contact seconds and class sizes; diagonal omitted.',
              'figures/exploratory/highschool/thresholds.pdf: baseline density/rank and connected-baseline weighted optimal protection.',
              'figures/exploratory/highschool/ratios.pdf: conditional source-prior-weighted ratios for the six fixed-action policies.',
              '', 'K. LIMITATIONS AND MODELING CONCERNS',
              'Metadata conflict: student28 is2BIO2 in metadata but2BIO1 in1321 contact rows. All results use metadata; embedded-label alternatives change durations.',
              'Metadata includes students2 and478 with no contact rows. These remain in class weights; this differs from the327-node individual contact graph.',
              'A class vertex is an indivisible infection unit: any infection in a class exposes its whole student weight under this model.',
              'Retaining one observed cross-class contact makes deterministic transmission possible across the whole class pair.',
              'The complete week graph has sequential OPT0 for every source, and static budgets1/2 also save0.',
              'Thresholding removes weaker observed routes before the process. Benefits at higher theta depend on this changed baseline, not just on the scored interventions.',
              'Class graph strengths are pair-contact seconds, without class-size normalization. Strength affects rankings, not transmission speed.',
              'Controlled infection rounds are not calibrated to20-second sensor intervals. Aggregation discards chronological contact order.',
              'Daily graphs are robustness descriptions of the aggregation; not a temporal transmission model.',
              'Connected-baseline results must not be extrapolated to disconnected levels or interpreted as causal intervention effects.',
              'No manuscript edit, production semantic change, or Git use.',
              '', 'L. REPRODUCIBILITY',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py highschool',
              'PYTHONDONTWRITEBYTECODE=1 python code/highschool.py',
              'PYTHONDONTWRITEBYTECODE=1 python code/tests.py',
              'PYTHONDONTWRITEBYTECODE=1 python code/highschool.py --plot',
              'Raw provenance: data/records.json [source:highschool]; SHA256 validated before analysis.']
    return '\n'.join(lines)+'\n'

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def read(name):
        with (root/'data/results'/name).open() as stream: return list(csv.DictReader(stream))
    edges = [r for r in read('highschooledges.csv') if r['period']=='week']
    thresholds = read('highschoolthresholds.csv')
    static = read('highschoolstatic.csv')
    sequential = read('highschoolsequential.csv')
    destination = root/'figures/exploratory/highschool'
    destination.mkdir(parents=True,exist_ok=True)
    def save(fig,name,caption):
        fig.text(.02,.01,caption,ha='left',va='bottom',fontsize=8,color='.3')
        fig.savefig(destination/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(Path(tempfile.gettempdir())/('highschool-'+name+'.png'),dpi=160,bbox_inches='tight')
        plt.close(fig)
    weights = {r['left']:int(r['leftsize']) for r in edges}
    weights.update({r['right']:int(r['rightsize']) for r in edges})
    classes = sorted(weights)
    matrix = np.full((9,9),np.nan)
    for row in edges:
        i,j = classes.index(row['left']),classes.index(row['right'])
        matrix[i,j] = matrix[j,i] = float(row['seconds'])/60
    fig,ax = plt.subplots(figsize=(7,6))
    palette = plt.get_cmap('Blues').copy()
    palette.set_bad('#eeeeee')
    picture = ax.imshow(matrix,cmap=palette,norm=LogNorm(vmin=np.nanmin(matrix),vmax=np.nanmax(matrix)))
    ax.set_xticks(range(9),classes,rotation=45,ha='right')
    ax.set_yticks(range(9),[f'{v} ({weights[v]})' for v in classes])
    ax.set_ylabel('Class (metadata students)')
    for i in range(9):
        for j in range(9):
            if i==j: continue
            color = 'white' if picture.norm(matrix[i,j])>.55 else '#18232b'
            ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=7,color=color)
    fig.colorbar(picture,ax=ax,pad=.025,fraction=.04,label='Cross-class contact minutes (log scale)')
    fig.subplots_adjust(bottom=.23,left=.17,right=.88,top=.98)
    save(fig,'contacts','High School 2013, five-day aggregate; metadata class assignments and all 329 metadata students.\nPair-contact minutes, rounded in cells; diagonal omitted. Within-class contact accounts for 93.49%.')
    fig,axes = plt.subplots(3,1,figsize=(7.5,7),sharex=True)
    x = [float(r['theta'])/60 for r in thresholds]
    axes[0].plot(x,[float(r['density']) for r in thresholds],'o-',color='#176078',ms=4)
    axes[0].set_ylabel('Density')
    axes[1].plot(x,[float(r['rank']) for r in thresholds],'s-',color='#444444',ms=4)
    axes[1].set_ylabel('Cycle rank')
    for budget,style in ((1,'--'),(2,':')):
        selected = [r for r in static if r['scope']=='intervention' and r['method']=='exact' and int(r['budget'])==budget]
        axes[2].plot([float(r['theta'])/60 for r in selected],[float(r['expecteddecimal']) for r in selected],style,
                     marker='s',ms=4,label=f'Static, {budget} cut'+('s' if budget==2 else ''),color='.35' if budget==1 else '.65')
    selected = [r for r in sequential if r['kind']=='summary' and r['method']=='exact']
    axes[2].plot([float(r['theta'])/60 for r in selected],[float(r['expecteddecimal']) for r in selected],'-o',
                 ms=4,label='Sequential',color='#176078')
    limit = max(float(r['theta']) for r in thresholds if r['connected']=='True')/60
    for ax in axes:
        ax.axvline(limit,color='.5',lw=.8,ls='--')
        ax.axvspan(limit,max(x),color='.94',zorder=-2)
        ax.set_xscale('symlog',linthresh=1)
        ax.grid(axis='y',color='.9',lw=.6)
    axes[2].set(ylabel='Expected students protected',xlabel='Minimum retained contact duration θ (minutes; symlog scale)')
    axes[2].legend(frameon=False,fontsize=8,loc='upper left')
    fig.subplots_adjust(bottom=.16,hspace=.16,left=.12,right=.97,top=.98)
    save(fig,'thresholds','Nine classes; student-uniform prior. Lines connect selected thresholds. Gray region is disconnected.\nStatic: source unknown, 1 or 2 cuts before spread. Sequential: source known, no global removal cap.')
    fig,ax = plt.subplots(figsize=(7.5,4.5))
    labels = ('Weak','Strong','BC','BCw','Source-aware BC','Threshold BC')
    markers = ('o','s','^','v','D','x')
    styles = (':','--','-','--',':','-')
    colors = ('.7','.5','.2','.4','#176078','#176078')
    for policy,label,marker,style,color in zip(policies,labels,markers,styles,colors):
        selected = [r for r in sequential if r['kind']=='summary' and r['method']==policy and r['meanratio']]
        ax.plot([float(r['theta'])/60 for r in selected],[float(Fraction(r['meanratio'])) for r in selected],
                marker=marker,ls=style,color=color,ms=5,lw=1.5,label=label)
    ax.set(xlabel='Minimum retained contact duration θ (minutes)',ylabel='Conditional mean policy / OPT',ylim=(-.03,1.05))
    counts = [r for r in sequential if r['kind']=='summary' and r['method']=='exact' and r['meanratio']]
    ax.set_xticks([float(r['theta'])/60 for r in counts],
                  [f"{float(r['theta'])/60:.1f}\nN={r['nontrivial']}" for r in counts])
    ax.grid(axis='y',color='.9',lw=.6)
    ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2))
    fig.subplots_adjust(bottom=.35,left=.12,right=.98,top=.97)
    save(fig,'ratios','Connected graphs; student-prior-weighted means conditional on OPT>0. N counts eligible source classes.\nAll-zero baselines omitted; lines connect selected thresholds. The six production action rules are unchanged.')
    print('Created three PDFs in figures/exploratory/highschool/')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot',action='store_true')
    args = parser.parse_args()
    if args.plot:
        plot()
        return
    import networkx as nx
    tick = perf_counter()
    weights,strengths,days,audit = readraw()
    graph = makegraph(weights,strengths)
    week = geometry(graph)
    allpriors,selection = priors(weights,strengths)
    prior = allpriors['student-uniform']
    identity(graph,[],weights,prior)
    print('AUDIT',audit['metadata'],'metadata students;',audit['contactstudents'],'contact students;',audit['rows'],'rows;',audit['days'],'days',flush=True)
    print('METADATA CONFLICT',audit['mismatches'],flush=True)
    print('WEEK',json.dumps(week,sort_keys=True),flush=True)
    edges = edgerecords(graph,weights,'week')
    daily = []
    for day,contact in sorted(days.items()):
        current = makegraph(weights,contact)
        values,bc,bcw = valuesandbc(current)
        strongest = min(values,key=lambda e:(-values[e],e))
        highest = max(bc.values())
        bestedges = sorted(e for e in bc if abs(bc[e]-highest)<=1e-12)
        row = dict(next(r for r in audit['daily'] if r['day']==day),**geometry(current),
                   strongest=strongest,strongestseconds=values[strongest],highestbc=2*highest,highestbcedges=bestedges)
        daily.append(row)
        edges.extend(edgerecords(current,weights,day))
    writecsv(root/'data/results/highschooledges.csv',edges)
    levels,limit,firstbridge,forest = thresholdlevels(weights,strengths)
    thresholds,static,sequential = [],[],[]
    connected = {}
    for theta,reasons in levels.items():
        current = makegraph(weights,strengths,theta)
        row = dict(theta=theta,minutes=theta/60,reasons=reasons,**geometry(current),
                   retainedseconds=sum(d['strength'] for a,b,d in current.edges(data=True)))
        row['filteredseconds'] = sum(strengths.values())-row['retainedseconds']
        row['baselineprotected'] = componentvalue(current,[],weights,prior)
        thresholds.append(row)
        if row['connected']:
            connected[theta] = current
            for budget in (1,2): static.extend(staticrecords(current,weights,prior,theta,budget))
            sequential.extend(sequentialrecords(current,weights,prior,theta))
        print('THRESHOLD',theta,'seconds;',row['edges'],'edges;',row['componentcount'],'components;',
              'lambda',row['connectivity'],'rank',row['rank'],flush=True)
    representatives = sorted({0,min(theta for theta,g in connected.items() if nx.edge_connectivity(g)==2),firstbridge,limit})
    for theta in representatives:
        for name,distribution in allpriors.items():
            static.extend(staticrecords(connected[theta],weights,distribution,theta,1,'prior',name))
    writecsv(root/'data/results/highschoolthresholds.csv',thresholds)
    writecsv(root/'data/results/highschoolstatic.csv',static)
    writecsv(root/'data/results/highschoolsequential.csv',sequential)
    text = report(audit,week,daily,thresholds,static,sequential,selection,limit,firstbridge,forest,perf_counter()-tick)
    savedreports.save('highschool',text)
    print('WROTE',len(edges),'edge rows,',len(thresholds),'thresholds,',len(static),'static rows,',len(sequential),'sequential rows',flush=True)
    print('CONNECTED LEVELS',len(connected),'largest theta',limit,'first bridges',firstbridge,'first forest',forest,flush=True)
    print('SECONDS',round(perf_counter()-tick,3),flush=True)

if __name__=='__main__':
    main()
