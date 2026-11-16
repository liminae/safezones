# Competing bottleneck experiment and exact formulas
import records as savedreports
import csv
import json
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from time import perf_counter
import networkx as nx
from air import edges, objectives, partitions, optimize, numerator, rankings
from hospital import components, score
from collections import Counter
from functools import cache
import sys

root = Path(__file__).resolve().parent.parent

families = dict(A=(6,2,(3,4,5,6,8,10,12)),B=(9,3,(4,5,6,7,8,10,12)))

sizes = (12,16,20,24,30)

def cases():
    return [(name,m,h) for name,(r,k,widths) in families.items() for m in sizes for h in widths]

def build(name,m,h):
    if name not in families: raise ValueError('Unknown family')
    r,k,widths = families[name]
    if isinstance(m,bool) or not isinstance(m,int) or m not in sizes: raise ValueError('Unknown outside split')
    if isinstance(h,bool) or not isinstance(h,int) or h not in widths: raise ValueError('Unknown gateway width')
    R,A,B = tuple(range(r)),tuple(range(r,r+60-m)),tuple(range(r+60-m,r+60))
    links = {e for block in (R,A,B) for e in combinations(block,2)}
    links.update((R[i],A[i]) for i in range(k))
    links.update((A[i],B[i]) for i in range(h))
    graph = nx.Graph()
    graph.add_nodes_from(range(r+60))
    graph.add_edges_from((u,v,dict(strength=1)) for u,v in sorted(links))
    return graph,R,A,B

def evaluate(name,m,h):
    start = perf_counter()
    graph,R,A,B = build(name,m,h)
    r,k,widths = families[name]
    adapted = graph.copy()
    nx.set_node_attributes(adapted,1,'inbound')
    nx.set_edge_attributes(adapted,1,'passengers')
    catalog = partitions(adapted,k)
    values,stats = optimize(catalog,'uu')
    cuts = values[k][4]
    exact = Fraction(numerator(catalog,'uu',values[k]),len(graph))
    p,a = objectives(adapted,'uniform','airports')
    assert score(graph,cuts,p,a)==exact
    boundary = tuple((R[i],A[i]) for i in range(k))
    gateway = tuple((A[i],B[i]) for i in range(h))
    candidate = score(graph,boundary,p,a)
    assert candidate==Fraction(2*r*60,r+60)<=exact
    order = rankings(adapted)['bc']
    ranks = {e:i+1 for i,e in enumerate(order)}
    chosen = order[:k]
    value = score(graph,chosen,p,a)
    assert value<=exact
    candidateload = Fraction(r*60,k)
    gatewayload = Fraction((r+len(A))*len(B),h)
    row = dict(family=name,r=r,k=k,budget=k,a_size=len(A),b_size=len(B),gateway_width=h,n=len(graph),
               m_edges=graph.number_of_edges(),exact=float(exact),candidate_score=float(candidate),
               candidate_optimal=candidate==exact,bc_score=float(value),bc_ratio=float(value/exact),
               bc_cuts=chosen,candidate_boundary=boundary,candidate_boundary_ranks=[ranks[e] for e in boundary],
               gateway_ranks=[ranks[e] for e in gateway],top_bc_is_candidate=tuple(sorted(chosen))==boundary,
               optimal_cut=cuts,optimal_partition=sorted(components(graph,cuts),key=lambda c:(-len(c),c)),
               candidate_load=float(candidateload),gateway_load=float(gatewayload),load_ratio=float(gatewayload/candidateload),
               gateway_edges=gateway,exact_fraction=str(exact),candidate_fraction=str(candidate),bc_fraction=str(value),
               bc_ratio_fraction=str(value/exact),load_ratio_fraction=str(gatewayload/candidateload),
               candidate_load_fraction=str(candidateload),gateway_load_fraction=str(gatewayload),
               exact_subsets=catalog['stats']['localsubsets'],exact_states=stats['maximumstates'])
    return row,dict(case=(name,m,h),seconds=perf_counter()-start)

def experiment(progress=False):
    rows,times = [],[]
    for name,m,h in cases():
        row,timing = evaluate(name,m,h)
        rows.append(row)
        times.append(timing)
        if progress and h==families[name][2][-1]:
            print(f'Completed {len(rows)}/70 exact searches: family {name}, |B|={m}',flush=True)
    return rows,times

def mistakes(rows,threshold):
    return [row for row in rows if (Fraction(row['load_ratio_fraction'])>threshold)!=(Fraction(row['bc_ratio_fraction'])<1)]

def bestthreshold(rows):
    levels = sorted({Fraction(row['load_ratio_fraction']) for row in rows})
    intervals = [(None,levels[0])]+list(zip(levels,levels[1:]))+[(levels[-1],None)]
    candidates = []
    for low,high in intervals:
        threshold = high-1 if low is None else low if high is None else (low+high)/2
        candidates.append((len(mistakes(rows,threshold)),threshold,low,high))
    best = min(item[0] for item in candidates)
    return [item for item in candidates if item[0]==best]

def interval(low,high):
    return f'(-infinity,{high})' if low is None else f'[{low},infinity)' if high is None else f'[{low},{high})'

def write(rows):
    with (root/'data/results/competition.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,tuple)) else v for k,v in row.items()})

def report(rows,times,seconds):
    lines = ['COMPETING BOTTLENECK EXPERIMENT',
             '70 deterministic graphs: 35 Family A (r=6,k=b=2,n=66), 35 Family B (r=9,k=b=3,n=69).',
             'R labels 0,...,r-1; A labels r,...,r+60-m-1; B labels r+60-m,...,r+59.',
             'Every block is a clique. Candidate edges (R[i],A[i]), i<k; gateway edges (A[i],B[i]), i<h.',
             'All strengths are 1; uniform source and unit target; fixed static cuts before contagion.',
             'The production air.partitions/air.optimize(mode=uu) solver is reused without changes.',
             'Unit inbound/passengers attributes on a copy provide unit node and edge accounting.',
             'Completeness: existing min-cut contraction, exhaustive feasible bundle subsets and exact dominance DP.',
             'No search-limit exceptions are suppressed. Every returned cut is scored on the original graph.',
             'The production air.rankings(graph)[bc] supplies the fixed original-graph unweighted BC ordering:',
             'twice NetworkX unnormalized undirected BC, then canonical lexicographic edges for computed ties.',
             'CSV boundary ranks align with candidate_boundary; gateway ranks align with gateway_edges.',
             'Cuts/partitions/rank lists are JSON; additional Fraction columns retain exact arithmetic.',
             '', 'STRUCTURAL CHECK',
             'Splitting a clique of size s needs at least s-1 edges. Each block here has s-1>k.',
             'Thus every block remains connected after <=k cuts. Because h>k, A and B remain connected to each other.',
             'The only possible nontrivial partition is R versus A union B, requiring all k candidate-boundary edges.',
             'Therefore that boundary is the unique disconnecting cut within budget and is globally optimal.',
             'Any other top-k BC set leaves the graph connected: BC protection is either the optimum or zero.',
             '',f'Exact searches completed: {len(rows)}/70',
             f'Candidate optimal: {sum(row["candidate_optimal"] for row in rows)}/{len(rows)}',
             f'Maximum local subsets: {max(row["exact_subsets"] for row in rows)}',
             f'Maximum DP states: {max(row["exact_states"] for row in rows)}']
    for name,(r,k,widths) in families.items():
        selected = [row for row in rows if row['family']==name]
        table = {(row['b_size'],row['gateway_width']):row for row in selected}
        ratios = [Fraction(row['bc_ratio_fraction']) for row in selected]
        lines.extend(['',f'FAMILY {name}: {len(selected)} graphs; exact score {selected[0]["exact_fraction"]}',
                      f'BC success/intermediate/zero: {sum(v==1 for v in ratios)}/{sum(0<v<1 for v in ratios)}/{sum(v==0 for v in ratios)}',
                      'BC ratio grid: m | '+ ' | '.join(f'h={h}' for h in widths)])
        for m in sizes:
            lines.append(f'  {m} | '+' | '.join(table[m,h]['bc_ratio_fraction'] for h in widths))
        lines.append('Candidate-boundary ranks: m | '+' | '.join(f'h={h}' for h in widths))
        for m in sizes:
            lines.append(f'  {m} | '+' | '.join(str(table[m,h]['candidate_boundary_ranks']) for h in widths))
        lines.append('Success/failure ranges by outside split:')
        for m in sizes:
            good = [h for h in widths if table[m,h]['bc_ratio']==1]
            bad = [h for h in widths if table[m,h]['bc_ratio']<1]
            lines.append(f'  m={m}: success h={good}; failure h={bad}')
        lines.append('BC ratio nondecreasing with h for every m: '+str(all(all(table[m,h]['bc_ratio']<=table[m,j]['bc_ratio'] for h,j in zip(widths,widths[1:])) for m in sizes)))
        lines.append('BC ratio nonincreasing with m for every h: '+str(all(all(table[m,h]['bc_ratio']>=table[j,h]['bc_ratio'] for m,j in zip(sizes,sizes[1:])) for h in widths)))
    lines.extend(['','LOAD-RATIO CLASSIFICATION',
                  'candidate_load=r*60/k; gateway_load=(r+|A|)*|B|/h; load_ratio=gateway_load/candidate_load.',
                  'These are structural proxies, not exact BC formulas. Predict failure iff load_ratio>threshold;',
                  'equality predicts success. Observed failure means BC/exact<1.',
                  'Threshold fitting uses these same cells (descriptive in-sample error, not held-out validation).',
                  'Every interval between distinct exact rational ratios, including both tails, is considered.',
                  'All minimizing intervals are reported; the first interval supplies the deterministic representative.'])
    for name in ('pooled','A','B'):
        selected = rows if name=='pooled' else [row for row in rows if row['family']==name]
        wrong = mistakes(selected,Fraction(1))
        falsepositives = sum(Fraction(row['load_ratio_fraction'])>1 for row in wrong)
        falsenegatives = len(wrong)-falsepositives
        fits = bestthreshold(selected)
        errors,threshold,low,high = fits[0]
        lines.extend([f'{name}: threshold 1 errors={len(wrong)}/{len(selected)}, false positives={falsepositives}, false negatives={falsenegatives}',
                      '  Misclassified at 1 (family,m,h,load ratio,BC ratio): '+str([(row['family'],row['b_size'],row['gateway_width'],row['load_ratio_fraction'],row['bc_ratio_fraction']) for row in wrong]),
                      f'  Best threshold representative: {threshold} = {float(threshold):.9f}; errors={errors}/{len(selected)}',
                      '  All minimizing threshold intervals: '+', '.join(interval(low,high) for errors,threshold,low,high in fits),
                      '  Misclassified at best representative: '+str([(row['family'],row['b_size'],row['gateway_width'],row['load_ratio_fraction'],row['bc_ratio_fraction']) for row in mistakes(selected,fits[0][1])])])
    mixed = [row for row in rows if 0<sum(e in row['candidate_boundary'] for e in row['bc_cuts'])<row['budget']]
    failed = [row for row in rows if row['bc_ratio']==0]
    lines.extend(['','MECHANISM',
                  'With budget k, even one selected edge away from the candidate boundary prevents complete separation.',
                  f'Top-k selections mixing candidate and other edges: {len(mixed)}.',
                  f'Zero-score cases selecting only A-B gateway edges: {sum(set(row["bc_cuts"])<=set(row["gateway_edges"]) for row in failed)}/{len(failed)}.',
                  'The R-A endpoints coincide with the first k gateway endpoints in A. Gateway edges need not share',
                  'cross-block shortest paths equally, so the average-load proxy can miss individual-edge competition.',
                  'Rank lists and selected edges in the CSV permit direct inspection without treating the proxy as BC.',
                  'Results describe this deterministic clique/matching family; they are not a universal centrality threshold.'])
    slowest = max(times,key=lambda t:t['seconds'])
    lines.extend(['',f'Experiment runtime: {seconds:.3f} seconds',
                  f'Slowest graph: {slowest["case"]}, {slowest["seconds"]:.3f} seconds including BC',
                  'Existing application outputs and production semantics were not changed.'])
    return '\n'.join(lines)+'\n'

def main():
    start = perf_counter()
    rows,times = experiment(progress=True)
    write(rows)
    text = report(rows,times,perf_counter()-start)
    savedreports.save('competition',text)
    print(text,end='')

proof = '''EXACT THEORY OF THE COMPETING BOTTLENECK CONSTRUCTION

Assumptions and notation
Let k>=1, k<h<=min(a,m), k<=r, and assume r-1>k, a-1>k,
m-1>k. Labels within each block are indexed from 1 in this discussion.
R_i A_i, i<=k, are the candidate edges; A_j B_j, j<=h, are the
gateway edges. The code uses consecutive zero-based labels instead.
BC is the unnormalized ordered-pair edge betweenness. Write beta=BC/2
while counting unordered source-target pairs, and multiply by 2 at the end.

Unique minimum cut and optimal static partition
For a clique of size s, a nontrivial partition with t vertices on one side
has t(s-t)>=s-1 crossing edges. Thus deleting at most k edges cannot
disconnect any of the three cliques. At least one A-B edge survives,
because h>k, so A and B remain in the same connected component. If even
one R-A edge survives, the whole graph remains connected. Consequently
the only disconnecting edge set of size at most k is precisely all k
candidate edges. This proves edge connectivity k and uniqueness of the
minimum edge-cut set (the two sides of its partition are unordered).
For a uniform source and unit targets, the component identity stated as
Proposition 5.1 gives n-sum_C |C|^2/n. Every other allowed cut gives zero,
whereas the candidate cut gives 2r(a+m)/(r+a+m)>0. Hence its cut set and
partition {R,A union B} are uniquely optimal for budget k.

Shortest-path counting
A shortest path never detours through a third clique to travel within
one clique. Paths between R and A do not use B; paths between A and B do
not use R. A shortest R-B path crosses each matching exactly once.
For two cliques joined by t matching edges, between matched endpoints
with different indices there are two shortest paths; between an unmatched
endpoint and a matched endpoint there is one; between two unmatched
endpoints there are t. The matching edges share all cross-block pairs
symmetrically. It follows that each R-A edge receives ra/k from R-A
pairs, and each A-B edge receives am/h from A-B pairs.

The additional R-B paths are as follows. In this table i<=k; j denotes
the target index in B; x>k denotes an unmatched vertex of R.
Source/target                 Length  Count  Gateway shares
R_i,B_i                        2       1    gateway i: 1
R_i,B_j, j<=k, j!=i             3       3    gateway i: 1/3; gateway j: 2/3
R_i,B_j, k<j<=h                3       2    gateways i and j: 1/2 each
R_i,B_j, j>h                   3       1    gateway i: 1
R_x,B_j, j<=k                  3       1    gateway j: 1
R_x,B_j, k<j<=h                4      2k    gateway j: 1/2; each shared gateway: 1/(2k)
R_x,B_j, j>h                   4       k    each shared gateway: 1/k

For example, the three paths in the second row are
R_i A_i B_i B_j, R_i A_i A_j B_j, and R_i R_j A_j B_j.
The 2k paths in the sixth row are, for each i<=k,
R_x R_i A_i B_i B_j and R_x R_i A_i A_j B_j.
These exhaust the possible shortest routes; any other route adds a step.

All k candidate edges are symmetric, so they divide the rm R-B pairs
equally. For a fixed shared gateway A_i B_i with i<=k, the first k sources
contribute k+(h-k)/2+(m-h)=(2m+k-h)/2. The r-k remaining sources contribute
(r-k)[1+(h-k)/(2k)+(m-h)/k]=(r-k)(2m+k-h)/(2k).
The total is r(2m+k-h)/(2k). For an unshared gateway A_j B_j, k<j<=h,
only pairs targeting B_j contribute; each of the r sources contributes
1/2. Thus its R-B contribution is r/2.

The three cross-block formulas are therefore
C = BC(R_i A_i) = 2r(a+m)/k,
S = BC(A_i B_i), i<=k = 2am/h + r(2m+k-h)/k,
T = BC(A_j B_j), k<j<=h = 2am/h + r.
In particular S-T=r(2m-h)/k>0. There are exactly k edges of type S.

Internal edge classes and their pair contributions
In R, s means a matched vertex and u means an unmatched vertex.
In A and B, s means index <=k, t means k<index<=h, and u means index>h.
All vertices with the same type are interchangeable under simultaneous
permutations of matched indices. Classes with too few endpoints are absent.
The following table gives UNORDERED contributions to a single edge.
The 'within' column is its own endpoint pair in a clique. Zero entries
mean no shortest paths of that vertex-pair type use the edge.

Edge class      within       R-A             A-B             R-B
candidate       0            ra/k            0               rm/k
shared gateway  0            0               am/h            r(2m+k-h)/(2k)
other gateway   0            0               am/h            r/2
Rss             1            1               0               2/3
Rsu             1            a/k             0               m/k
Ruu             1            0               0               0
Ass             1            1               1               2/3
Ast             1            r/k             1               r/(2k)
Asu             1            r/k             m/h             0
Att             1            0               1               0
Atu             1            0               m/h             0
Auu             1            0               0               0
Bss             1            0               1               2/3
Bst             1            0               1               r/(2k)
Bsu             1            0               a/h             r/k
Btt             1            0               1               0
Btu             1            0               a/h             0
Buu             1            0               0               0

To justify the internal entries, a within-clique edge can occur on a
cross-block shortest path only as the first, last, or intervening clique
step. The two-clique path counts above give contribution 1 for two matched
endpoints, and opposite-clique-size/matching-size for one unmatched endpoint.
For R-B pairs the seven-row table supplies all remaining entries. For
example an Ast or Bst edge receives 1/2 from its matched R source and
(r-k)/(2k) from unmatched R sources, totaling r/(2k). An Rss, Ass, or Bss
edge receives 1/3 from each of the two oppositely indexed pairs, totaling
2/3. An Rsu edge receives 1+(m-k)/k=m/k from R-B pairs; a Bsu edge receives
1+(r-k)/k=r/k. All other R-B entries are zero. Multiplying each row sum
by 2 gives the ordered-pair formula for that internal edge class.

Why no internal edge can interfere with the boundary comparison
Put c=C/2=r(a+m)/k. Half the BC of any internal edge is at most
M=max{11/3, 1+(a+m)/k, 2+3r/(2k), 1+r/k+m/h, 1+r/k+a/h}.
Every term is strictly smaller than c under the stated assumptions:
c>6>11/3; c-[1+(a+m)/k]=(r-1)(a+m)/k-1>0;
c-[2+3r/(2k)]=(r/k)(a+m-3/2)-2>0.
Since h>k,
c-[1+r/k+m/h]>[r(a-1)+(r-1)m-k]/k>0,
and the final term follows by interchanging a and m.
Thus every internal edge has BC<C; S>T; and the only possible competitors
to the k candidate edges are the k shared gateways.

Candidate proposition: exact BC failure threshold
For G(r,a,m;k,h) satisfying the assumptions, let canonical tie breaking
order R labels before A labels before B labels. The candidate boundary
is the unique optimal static budget-k cut and has value 2r(a+m)/(r+a+m).
Define Delta=2kam-rh(2a+h-k). If Delta>0, the top k individual BC edges
are exactly the shared gateways A_i B_i, i<=k. Removing them leaves the
graph connected and gives zero protection. If Delta<0, the top k edges
are exactly the candidate boundary and BC attains the optimum. If Delta=0,
those two classes tie, and canonical lexicographic ordering selects the
candidate boundary. Thus under exact-score canonical ties, BC fails iff
                         2kam > rh(2a+h-k).
Proof: the path counts give S-C=Delta/(kh). There are k shared gateways,
all unshared gateways have smaller BC than S, and every internal edge
has BC<C. These comparisons prove the ranking claims; the minimum-cut
argument proves the protection claims. At equality, every candidate edge
has its first endpoint in R and precedes every gateway edge lexicographically.
Floating-point implementations need not preserve an exact tie; the saved
70 graphs are checked for Delta=0 separately below.

Family simplifications
Both families have r=3k, a=60-m. Failure is equivalent to
              2m(60-m) > 3h(120-2m+h-k).
Family A (r=6,k=2): 2m(60-m) > 3h(118-2m+h).
Family B (r=9,k=3): 2m(60-m) > 3h(117-2m+h).
Equivalently h is strictly below the positive root
H=[-(2a-k)+sqrt((2a-k)^2+8kam/r)]/2.
The right-hand side rh(2a+h-k) is strictly increasing for h>0 here.
Threshold equality counts as success under exact canonical tie breaking.
These inequalities come from shortest-path enumeration, not a fitted proxy.
'''

def terms(r,a,m,k,h):
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (r,a,m,k,h)):
        raise ValueError('Require positive integer parameters')
    if not k<h<=min(a,m) or k>r: raise ValueError('Infeasible matchings')
    values = dict(candidate=(0,Fraction(r*a,k),0,Fraction(r*m,k)),
                  shared=(0,0,Fraction(a*m,h),Fraction(r*(2*m+k-h),2*k)),
                  other=(0,0,Fraction(a*m,h),Fraction(r,2)),
                  Rss=(1,1,0,Fraction(2,3)),Rsu=(1,Fraction(a,k),0,Fraction(m,k)),Ruu=(1,0,0,0),
                  Ass=(1,1,1,Fraction(2,3)),Ast=(1,Fraction(r,k),1,Fraction(r,2*k)),Asu=(1,Fraction(r,k),Fraction(m,h),0),
                  Att=(1,0,1,0),Atu=(1,0,Fraction(m,h),0),Auu=(1,0,0,0),
                  Bss=(1,0,1,Fraction(2,3)),Bst=(1,0,1,Fraction(r,2*k)),Bsu=(1,0,Fraction(a,h),Fraction(r,k)),
                  Btt=(1,0,1,0),Btu=(1,0,Fraction(a,h),0),Buu=(1,0,0,0))
    return {name:tuple(map(Fraction,value)) for name,value in values.items()}

def formulas(r,a,m,k,h):
    return {name:2*sum(value) for name,value in terms(r,a,m,k,h).items()}

def kind(u,v,r,a,k,h):
    u,v = sorted((u,v))
    if u<r<=v<r+a: return 'candidate'
    if r<=u<r+a<=v: return 'shared' if u-r<k else 'other'
    if v<r: return 'R'+('s' if u<k else 'u')+('s' if v<k else 'u')
    block,offset = ('A',r) if v<r+a else ('B',r+a)
    return block+''.join('s' if w-offset<k else 't' if w-offset<h else 'u' for w in (u,v))

def margin(r,a,m,k,h):
    return 2*k*a*m-r*h*(2*a+h-k)

def pathcounts(graph,r,a):
    values = {tuple(sorted(e)):[Fraction(0) for i in range(4)] for e in graph.edges}
    for s in sorted(graph):
        predecessors = nx.predecessor(graph,s)
        @cache
        def paths(v):
            if v==s: return ((s,),)
            return tuple(path+(v,) for u in predecessors[v] for path in paths(u))
        for t in sorted(v for v in graph if v>s):
            left = 0 if s<r else 1 if s<r+a else 2
            right = 0 if t<r else 1 if t<r+a else 2
            column = 0 if left==right else 1 if (left,right)==(0,1) else 2 if (left,right)==(1,2) else 3
            routes = ((s,t),) if graph.has_edge(s,t) else paths(t)
            share = Fraction(1,len(routes))
            for path in routes:
                for u,v in zip(path,path[1:]): values[tuple(sorted((u,v)))][column] += share
        paths.cache_clear()
    return {e:tuple(value) for e,value in values.items()}

def verify():
    start = perf_counter()
    with (root/'data/results/competition.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    checks = dict(graphs=0,edges=0,relevant=0,classes=Counter(),formulaerrors=[],floaterrors=[],outcomeerrors=[],
                  cuterrors=[],ties=[],floatunequal=0,floatmaximum=0.0,relevantunequal=0,relevantmaximum=0.0,
                  relevantroundoff=[],exactfailures=Counter())
    for row in rows:
        name,r,a,m,k,h = row['family'],int(row['r']),int(row['a_size']),int(row['b_size']),int(row['k']),int(row['gateway_width'])
        case = name,m,h
        assert min(r-1,a-1,m-1)>k and h>k
        graph,R,A,B = build(name,m,h)
        values = formulas(r,a,m,k,h)
        contributions = terms(r,a,m,k,h)
        counted = pathcounts(graph,r,a)
        computed = {tuple(sorted(e)):2*value for e,value in nx.edge_betweenness_centrality(graph,normalized=False).items()}
        for e in sorted(counted):
            label = kind(*e,r,a,k,h)
            checks['classes'][label] += 1
            checks['edges'] += 1
            checks['relevant'] += label in ('candidate','shared','other')
            if counted[e]!=contributions[label]: checks['formulaerrors'].append((case,e,label,counted[e],contributions[label]))
            error = abs(computed[e]-float(values[label]))
            checks['floatmaximum'] = max(checks['floatmaximum'],error)
            checks['floatunequal'] += error!=0
            if label in ('candidate','shared','other'):
                checks['relevantunequal'] += error!=0
                checks['relevantmaximum'] = max(checks['relevantmaximum'],error)
                if error: checks['relevantroundoff'].append((case,e,label,str(values[label]),computed[e],error))
            if error>1e-9: checks['floaterrors'].append((case,e,label,computed[e],str(values[label]),error))
        delta = margin(r,a,m,k,h)
        assert values['shared']-values['candidate']==Fraction(delta,k*h)
        assert values['shared']>values['other']
        assert all(value<values['candidate'] for label,value in values.items() if label not in ('candidate','shared','other'))
        assert delta==k*(2*m*(60-m)-3*h*(120-2*m+h-k))
        boundary = tuple((R[i],A[i]) for i in range(k))
        shared = tuple((A[i],B[i]) for i in range(k))
        predicted = shared if delta>0 else boundary
        order = sorted(counted,key=lambda e:(-values[kind(*e,r,a,k,h)],e))
        assert tuple(sorted(order[:k]))==predicted
        saved = tuple(sorted(tuple(e) for e in json.loads(row['bc_cuts'])))
        observed = Fraction(row['bc_ratio_fraction'])<1
        if observed!=(delta>0) or saved!=predicted: checks['outcomeerrors'].append((case,delta,str(values['candidate']),str(values['shared']),predicted,saved,row['bc_ratio_fraction']))
        optimum = Fraction(2*r*(a+m),r+a+m)
        optimumcuts = tuple(tuple(e) for e in json.loads(row['optimal_cut']))
        groups = {frozenset(c) for c in json.loads(row['optimal_partition'])}
        if (Fraction(row['exact_fraction'])!=optimum or Fraction(row['candidate_fraction'])!=optimum
            or Fraction(row['bc_fraction'])!=(0 if delta>0 else optimum)
            or Fraction(row['bc_ratio_fraction'])!=int(delta<=0)
            or optimumcuts!=boundary or groups!={frozenset(R),frozenset(A+B)}):
            checks['cuterrors'].append((case,row['exact_fraction'],optimumcuts))
        if delta==0: checks['ties'].append(case)
        checks['exactfailures'][name] += delta>0
        checks['graphs'] += 1
    checks['seconds'] = perf_counter()-start
    return rows,checks

def theoryreport(rows,checks):
    lines = [proof,'VERIFICATION AGAINST THE EXISTING 70 GRAPHS',
             'Only competition.build and the saved competition.csv are used; no optimizer or earlier sweep is rerun.',
             'An independent verifier enumerates all shortest paths from BFS predecessor DAGs and accumulates',
             'Fraction shares separately for within-block, R-A, A-B, and R-B pairs. All four columns are checked,',
             'for every edge, against the derived table. This is a rational path count, not a numerical fit.',
             f'Graphs checked: {checks["graphs"]}; all edges: {checks["edges"]}; cross-block edges: {checks["relevant"]}',
             'Edge-class counts: '+str(dict(sorted(checks['classes'].items()))),
             f'Exact rational contribution-table mismatches: {len(checks["formulaerrors"])}',
             f'NetworkX float discrepancies exceeding 1e-9: {len(checks["floaterrors"])}',
             f'Maximum absolute ordered-BC float error: {checks["floatmaximum"]:.16g}',
             f'NetworkX floats differing bitwise from the rounded rational formula: {checks["floatunequal"]}/{checks["edges"]}',
             f'Cross-block maximum float error: {checks["relevantmaximum"]:.16g}; unequal rounded floats: {checks["relevantunequal"]}/{checks["relevant"]}',
             'NetworkX uses floating arithmetic. Agreement with exact Fraction enumeration is exact; agreement',
             'with NetworkX is numerical to the stated error, not a claim of bitwise or rational equality.',
             f'Analytical outcomes/selected-cut mismatches: {len(checks["outcomeerrors"])}',
             f'Saved optimal value/cut/partition mismatches: {len(checks["cuterrors"])}',
             'Exact candidate/shared-gateway ties: '+str(checks['ties'])]
    for label in ('formulaerrors','floaterrors','outcomeerrors','cuterrors'):
        if checks[label]:
            lines.append(label+':')
            lines.extend(str(item) for item in checks[label])
    lines.extend(['','TRANSITION WIDTHS',
                  'The inequality is evaluated exactly; success means Delta<=0. Lists are restricted to the saved grid.',
                  'family | m | strict-failure widths | success widths | smallest positive integer success width'])
    for name in ('A','B'):
        subset = [row for row in rows if row['family']==name]
        for m in sorted({int(row['b_size']) for row in subset}):
            selected = [row for row in subset if int(row['b_size'])==m]
            r,a,k = int(selected[0]['r']),int(selected[0]['a_size']),int(selected[0]['k'])
            bad = [int(row['gateway_width']) for row in selected if margin(r,a,m,k,int(row['gateway_width']))>0]
            good = [int(row['gateway_width']) for row in selected if margin(r,a,m,k,int(row['gateway_width']))<=0]
            first = next((h for h in range(k+1,min(a,m)+1) if margin(r,a,m,k,h)<=0),None)
            lines.append(f'{name} | {m} | {bad} | {good} | {first}')
        lines.append(f'Family {name}: {checks["exactfailures"][name]} predicted failures, {len(subset)-checks["exactfailures"][name]} predicted successes')
    lines.extend(['','The first feasible integer success width is an algebraic consequence of the inequality;',
                  'no graphs outside the existing 70-row grid were generated or searched.',
                  f'Verification runtime: {checks["seconds"]:.3f} seconds',
                  'The fitted load-ratio threshold is not used in the formulas, proof, or classifier.'])
    if checks['relevantroundoff']:
        lines.extend(['','NUMERICAL ROUNDOFF APPENDIX',
                      'Every cross-block edge for which NetworkX differs from the exact formula rounded to float.',
                      'These are not rational path-count mismatches; no missing edge class or condition is involved.',
                      'Columns: (family,m,h), edge, class, exact ordered BC, NetworkX ordered BC, absolute float difference.'])
        lines.extend(str(item) for item in checks['relevantroundoff'])
    return '\n'.join(lines)+'\n'

def theory():
    rows,checks = verify()
    text = theoryreport(rows,checks)
    savedreports.save('competitiontheory',text)
    print(text,end='')
    assert not any(checks[name] for name in ('formulaerrors','floaterrors','outcomeerrors','cuterrors'))

if __name__=='__main__':
    modes = {'--theory':theory}
    selected = [flag for flag in modes if flag in sys.argv]
    if len(selected)>1: raise SystemExit('Choose one analysis mode')
    action = modes[selected[0]] if selected else main
    if selected: sys.argv.remove(selected[0])
    if '--help' in sys.argv or '-h' in sys.argv:
        print('Additional analysis modes: --theory')
    else: action()
