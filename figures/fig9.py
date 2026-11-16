# Saved Washington pilot analysis and figure
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import sys
from time import perf_counter

root = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'code'))
import records as savedreports
import washington as pilot
folder = Path(__file__).resolve().parent

def forbidden(*args,**kwargs):
    raise AssertionError('Optimization is forbidden in this analysis')

def extract():
    pilot.solve = forbidden
    pilot.air.partitions = forbidden
    pilot.air.optimize = forbidden
    base = pilot.original()
    cases = {c['variant']:c for c in pilot.variants(base)}
    report = savedreports.text('washington')
    with (pilot.folder/'washington.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        saved = [(reader.line_num,row) for row in reader]
    assert len(saved)==44 and len(cases)==34
    rows = []
    for line,row in saved:
        case = cases[row['variant']]
        graph = case['graph']
        b = int(row['budget'])
        assert row['exactstatus']=='exact' and not row['exacterror']
        assert row['graphhash']==case['graphhash']
        assert int(row['n'])==len(graph)==443 and int(row['m'])==graph.number_of_edges()
        for key in ('added','removed','seeds'):
            assert json.loads(row[key])==json.loads(json.dumps(case[key]))
        for key,vertices in (('r',pilot.R),('u',pilot.U)):
            assert json.loads(row[key+'boundary'])==list(map(list,pilot.boundary(graph,vertices)))
            assert int(row[key+'boundarysize'])==len(pilot.boundary(graph,vertices))
            assert int(row[key+'internaledges'])==graph.subgraph(vertices).number_of_edges()
            assert Fraction(row[key+'density'])==Fraction(graph.subgraph(vertices).number_of_edges(),len(vertices)*(len(vertices)-1)//2)
        assert all(row[key]=='True' for key in ('graphconnected','rconnected','rcomplementconnected','uconnected','ucomplementconnected'))
        p,a = pilot.air.objectives(graph,'uniform','airports')
        exact,bc = Fraction(row['exact']),Fraction(row['bcscore'])
        cuts,chosen = json.loads(row['optimalcut']),json.loads(row['bccuts'])
        assert len(cuts)<=b and len(chosen)==b
        assert pilot.score(graph,cuts,p,a)==exact>=bc==pilot.score(graph,chosen,p,a)
        groups = sorted(pilot.components(graph,cuts),key=lambda c:(-len(c),c))
        assert json.loads(row['optimalpartition'])==list(map(list,groups))
        assert json.loads(row['optimalcomponents'])==list(map(len,groups))
        assert json.loads(row['bccomponents'])==sorted(map(len,pilot.components(graph,chosen)),reverse=True)
        assert Fraction(row['bcratio'])==bc/exact
        for key,value in (('exactdecimal',exact),('bcdecimal',bc),('bcratiodecimal',bc/exact)):
            assert abs(float(row[key])-float(value))<1e-12
        fields = ('variant','seeds','level','budget')
        display = dict(row)
        for key in ('seeds','optimalcomponents','added','removed'): display[key] = json.dumps(json.loads(row[key]))
        summary = ' | '.join(display[key] for key in fields)+f' | {row["rboundarysize"]}/{row["uboundarysize"]} | '+ ' | '.join(display[key] for key in ('rdensity','exact','bcscore','bcratio','candidateoptimal','rinoptimal','optimalcomponents'))
        assert summary in report
        modification = row['variant']+' seeds='+display['seeds']+' add='+display['added']+' remove='+display['removed']
        assert modification in report
        out = dict(csvline=line,variant=row['variant'],family=row['feature'],status='measured' if row['feature']=='original' else 'synthetic',
                   parameter=row['feature'],level=row['level'] if row['feature']!='original' else 'not applicable',seeds=row['seeds'],budget=b,
                   budgettype='fixed and matching' if b==3 and int(row['rboundarysize'])==3 else 'fixed' if b==3 else 'matching',
                   n=443,m=row['m'],rinternaledges=row['rinternaledges'],uinternaledges=row['uinternaledges'],rdensity=row['rdensity'],udensity=row['udensity'],
                   rboundarysize=row['rboundarysize'],uboundarysize=row['uboundarysize'],exact=str(exact),exactdecimal=float(exact),
                   bc=str(bc),bcdecimal=float(bc),ratio=str(bc/exact),ratiodecimal=float(bc/exact),bcpositive=bc>0,bcoptimal=bc==exact,
                   optimalcut=row['optimalcut'],optimalpartition=row['optimalpartition'],optimalcomponents=row['optimalcomponents'],bccuts=row['bccuts'],
                   rboundary=row['rboundary'],uboundary=row['uboundary'],rboundaryranks=row['rboundaryranks'],graphhash=row['graphhash'])
        for key,vertices in (('r',pilot.R),('u',pilot.U)):
            boundary = pilot.boundary(graph,vertices)
            value = pilot.score(graph,boundary,p,a)
            assert value==Fraction(7812 if key=='r' else 11180,443)
            feasible = len(boundary)<=b
            assert not feasible or value<=exact
            out[key+'feasible'] = feasible
            out[key+'score'] = str(value) if feasible else 'infeasible'
            out[key+'decimal'] = float(value) if feasible else ''
            out[key+'optimal'] = value==exact if feasible else 'infeasible'
            out[key+'inrepresentative'] = any(set(c)==vertices for c in groups)
            if key=='r':
                assert (row['candidatefeasible']=='True')==feasible
                assert Fraction(row['candidatescore'])==value
                assert (row['candidateoptimal']=='True')==(feasible and value==exact)
                assert (row['rinoptimal']=='True')==out[key+'inrepresentative']
        kind = 'R' if cuts==json.loads(row['rboundary']) else 'U' if cuts==json.loads(row['uboundary']) else 'other'
        out['representative'] = kind
        out['anotheroptimalcut'] = 'yes: saved representative' if kind=='other' else 'unverified: no co-optimal enumeration'
        rows.append(out)
    for feature,levels in pilot.levels.items():
        for level in levels:
            group = [r for r in rows if r['family']==feature and int(r['level'])==level and r['budget']==3]
            ratios = [Fraction(r['ratio']) for r in group]
            summary = f'{feature} | {level} | {len(group)} | {sum(r["roptimal"] is True for r in group)}/{len(group)} | {float(min(ratios)):.6f}..{float(max(ratios)):.6f}'
            assert summary in report
    for line in report.splitlines():
        if len(line.split())==2 and len(line.split()[1])==64:
            path,digest = line.split()
            savedreports.verify(path,digest)
    assert sum(r['bcpositive'] for r in rows)==4 and sum(r['bcoptimal'] for r in rows)==1
    assert sum(r['roptimal'] is True for r in rows)==28 and sum(not r['rfeasible'] for r in rows)==6
    assert sum(r['uoptimal'] is True for r in rows)==4 and sum(r['representative']=='other' for r in rows)==12
    return rows

def references(rows):
    return '; '.join(f'{r["variant"]}/b={r["budget"]} (results.csv:{r["csvline"]})' for r in rows)

def textfiles(rows):
    positive = [r for r in rows if r['bcpositive']]
    suboptimal = [r for r in rows if r['rfeasible'] and r['roptimal'] is False]
    unavailable = [r for r in rows if not r['rfeasible']]
    uoptimal = [r for r in rows if r['uoptimal'] is True]
    interpretation = '''WASHINGTON PILOT: SAVED-RESULT INTERPRETATION

Scope: 34 distinct graphs and 44 graph/budget cases; 1 measured baseline and 33 synthetic counterfactuals. All 443 vertices are retained. No optimization was rerun. Saved exact results remain the source of optimality certification; all saved cut scores and partitions were independently replayed with exact rational arithmetic.

The original measured graph has OPT=7812/443=17.6343 expected protected airports at b=3, achieved by isolating R. Top-3 BC has value zero. Its R-boundary ranks are 1, 223, 210 in canonical boundary-edge order.

BC is zero in 40 of 44 cases and positive in four. It is optimal only for boundary1s1 at b=1. The two seeds at width 1 have identical OPT but different matching-budget BC performance: zero versus one in BC/OPT. With b=3, both save R but miss two additionally separable singleton airports.

Internal placement matters. At 9 internal edges, seed 0 gives OPT=8694/443 and BC/OPT=1160/1449; seed 1 gives OPT=7812/443 and BC=0. At 15 edges, the optima are 7828/443 and 7812/443, with zero BC in both. At each tested level 21, 27, 33, 36 (and the measured 30), R alone is optimal and BC is zero. This is not a universal density threshold: only two placement seeds were examined and intermediate levels were not tested.

Across families, equal total edge count and equal budget also fail to determine OPT. The complete list has eight edge-count groups, all at b=3: 7139, 7145, 7157, 7158, 7159, 7161, 7162, 7163. The first two are the seed-placement comparisons above; the other six change where edges are modified. Every case and exact value is listed in evidence.txt.

Changing R width at fixed b=3: widths 1 and 2 permit R plus extra singleton separation, improving OPT to 9542/443 and 8678/443. Widths 4, 5, 6 make the R cut INFEASIBLE; OPT instead is 2646/443 from three singletons. At matching budget, R alone is optimal at widths 1, 2, 4, 5; at width 6 the larger U cut is feasible and better (11180/443). The measured width 3 is optimal at b=3.

Changing U width at b=3: width 3 makes U optimal (11180/443) for both seeds. At tested widths 4, 5, 7, 8, 9, and measured width 6, R alone is optimal (7812/443). BC remains zero throughout this family.

R alone is optimal in 28 cases, feasible but suboptimal in 10, and infeasible in 6. U is feasible and optimal in four cases; it is infeasible in the other 40. A different saved optimal cut occurs in 12 cases. These are representative solutions, not an enumeration of co-optima: uniqueness and the absence of alternative optimal cuts are not established.

Figure 9 uses three paired panels: exact expected protection above, BC/OPT below. Solid blue is fixed b=3; dashed green is matching b=|delta(R)|. Circles/squares distinguish placement seeds; coincident symbols are nested. Black diamonds mark the same measured baseline, repeated for reference in each panel. Vertical dotted guides mark measured parameter values. Zero scores are retained; the figure states their exact count. The two curves at 15 internal edges nearly overlap because their exact difference is only 16/443. All non-diamond cases are synthetic, not observed alternative routes. Lines guide the eye between tested levels and are not interpolated experiments.

The pilot summary's numerical claims, 44 result lines, conditional summaries, graph modification log and four input/code hashes all agree with the saved data and reconstruction. Its fixed-budget 0/2 R-optimal counts at widths 4-6 denote infeasibility, not a zero-valued R intervention. Runtime/platform are recorded provenance and were not remeasured. No substantive numerical correction to the saved pilot report was needed.
'''
    savedreports.save('washingtoninterpretation',interpretation)
    lines = ['EXACT EVIDENCE FOR POTENTIAL MANUSCRIPT CLAIMS','',
             'All line references below are to data/results/washington.csv unless stated otherwise. Fractions are authoritative; decimals are display values.',
             'Evidence covers this finite pilot only. No statistical sampling claim, universal threshold, uniqueness claim, or enumeration of all co-optima is supported. Optimality is inherited from the saved exhaustive run; this audit replays witnesses without rerunning it.','',
             '1. Baseline: results.csv:2; n=443, m=7160, R internal=30, R boundary=3, U boundary=6, OPT=7812/443, BC=0, R optimal; ranks=[1,223,210].',
             '2. Counts: all rows 2-45; 34 graph hashes, 44 cases, all exactstatus=exact. BC zero in 40 cases; positive cases:',references(positive)]
    for r in positive:
        lines.append(f'  {r["variant"]}/b={r["budget"]}: OPT={r["exact"]}, BC={r["bc"]}, BC/OPT={r["ratio"]}.')
    lines += ['3. BC optimal only: '+references([r for r in rows if r['bcoptimal']]),
              '4. R feasible but suboptimal (10): '+references(suboptimal),
              '5. R infeasible (6), NOT zero objective: '+references(unavailable),
              '6. U feasible and optimal (4): '+references(uoptimal),
              '7. Other saved optimal cut (12): '+references([r for r in rows if r['representative']=='other']),
              '8. Equal edge counts, different OPT: internal9s0 versus internal9s1 (lines 3-4): 8694/443 versus 7812/443. Internal15s0 versus internal15s1 (lines 5-6): 7828/443 versus 7812/443. Both comparisons have b=3 and identical R/U boundary counts. These are all such within-family equal-level/equal-budget differences.',
              '9. Equal width and OPT, different BC: boundary1s0/b=1 versus boundary1s1/b=1 (lines 14,16), BC 0 versus 7812/443. Boundary identity/placement matters.',
              '10. Dense internal levels tested: lines 7-13 and measured line 2; all R optimal, BC=0. Full clique has seed aliases [0,1] and is counted only once.',
              '11. Fixed b=3 R widths 1,2: lines 15,17,19,21; optimal sizes [432,9,1,1] or [433,9,1]. R remains in saved representative but R alone is not optimal.',
              '12. Fixed b=3 R widths 4,5,6: lines 22,24,26,28,30,32; R infeasible, OPT=2646/443, sizes [440,1,1,1]. Matching b=4,5 restores R optimality (lines 23,25,27,29); b=6 instead selects U (31,33).',
              '13. U widths: lines 34-45; width 3 selects U with OPT=11180/443; widths 4,5,7,8,9 select R with OPT=7812/443. BC=0 in all twelve. Measured width 6: line 2.','',
              'PILOT SUMMARY AUDIT',
              'report.txt:3-17: construction, fixed vertices, connected regions/complements, uniform static objective, seeded changes and candidate semantics checked against pilot.py:20-152 and reconstructed graphs. Boundary changes also change U internal density, as stated.',
              'report.txt:7: ordered-pair, unweighted, original-per-variant BC and canonical ties match pilot.py:121-122 and code/air.py:273-284. No centrality or optimizer was rerun for this analysis.',
              'report.txt:18-20: solver and cap descriptions match pilot.py:100-115; completeness argument is code/air.py:21-68. Saved exact status is verified; replay is an independent witness check, not a new upper-bound proof.',
              'report.txt:21,29: environment and 103.175-second original runtime are saved provenance, not independently measurable from this audit. No claim to reproduce that runtime.',
              'report.txt:23-26: all four SHA256 hashes checked against current files.',
              'report.txt:28-30: graph/case/exact counts and baseline match saved CSV.',
              'report.txt:34-77: every row regenerated from CSV and checked verbatim.',
              'report.txt:81-97: all 17 conditional summary lines independently regenerated and checked verbatim. Infeasible R cuts must not be interpreted as a zero-value candidate.',
              'report.txt:100-133: all 34 complete edge-change log lines checked against CSV and reconstructed graphs.',
              'report.txt:135 onward: figure/reproduction recommendations are prescriptions, not numerical findings. Current visualization preserves zero observations and separates infeasibility.','',
              'ALL EQUAL-TOTAL-EDGE-COUNT / EQUAL-BUDGET GROUPS WITH DIFFERENT OPTIMA']
    groups = {}
    for r in rows: groups.setdefault((int(r['m']),r['budget']),[]).append(r)
    contrasts = [(key,group) for key,group in sorted(groups.items()) if len({r['exact'] for r in group})>1]
    assert len(contrasts)==8
    for (edges,budget),group in contrasts:
        lines.append(f'm={edges}, b={budget}: '+ '; '.join(f'{r["variant"]} OPT={r["exact"]} (CSV:{r["csvline"]})' for r in group))
    lines += ['', 'CASE-BY-CASE EVIDENCE']
    for r in rows:
        lines.append(f'CSV:{r["csvline"]} {r["variant"]}, seeds={r["seeds"]}, b={r["budget"]}, OPT={r["exact"]}, BC={r["bc"]}, ratio={r["ratio"]}; R score={r["rscore"]}, R optimal={r["roptimal"]}; U score={r["uscore"]}, U optimal={r["uoptimal"]}; representative={r["representative"]}, sizes={r["optimalcomponents"]}; other optimal cut={r["anotheroptimalcut"]}.')
    lines += ['', 'REPRODUCTION',
              'PYTHONDONTWRITEBYTECODE=1 python figures/fig9.py',
              'The figure fragment uses PGFPlots, xcolor dvipsnames, graphicx and the TikZ shapes.geometric library. Preview build files are not stored in the repository.',
              'The generated TeX has no percent characters. fig9.tex is an input fragment scaled to columnwidth. All coordinate values derive from the saved Washington CSV; the original derived-data checksum is retained in data/records.json.',
              'FIGURE NOTE: internal 36 seeds are aliases of one graph. The measured graph is repeated in all panels for reference, and in both budget traces in panel b, without becoming a new experimental case. All other traces are synthetic. Both seed markers are nested where coordinates coincide.']
    savedreports.save('washingtonevidence','\n'.join(lines)+'\n')

def figure(rows):
    parts = [r'\resizebox{\columnwidth}{!}{',r'\begin{tikzpicture}',
             r'\tikzset{every node/.style={font=\fontsize{8}{9}\selectfont}}',
             r'\draw[RoyalBlue,line width=0.8pt] (0,15.30)--(0.4,15.30);',
             r'\node[anchor=west] at (0.48,15.30) {Fixed $b=3$};',
             r'\draw[ForestGreen,dashed,line width=0.8pt] (3.35,15.30)--(3.75,15.30);',
             r'\node[anchor=west] at (3.83,15.30) {Matching $b=|\delta(R)|$};',
             r'\fill[black!70] (0.20,14.92) circle (1.5pt);',
             r'\node[anchor=west] at (0.48,14.92) {Seed 0};',
             r'\draw[black!70] (1.80,14.87) rectangle (1.90,14.97);',
             r'\node[anchor=west] at (2.08,14.92) {Seed 1};',
             r'\node[diamond,fill=black,inner sep=1.4pt] at (3.95,14.92) {};',
             r'\node[anchor=west] at (4.20,14.92) {Measured baseline};',
             r'\node[anchor=west,font=\fontsize{7}{8}\selectfont,text=black!70] at (0,14.55) {OPT: expected protected airports under a uniform source};']
    configs = [('internal','(a) R internal connectivity',14.15,9,36,30,[9,15,21,27,30,33,36],r'Internal edges in $R$'),
               ('boundary','(b) R boundary width',9.25,1,6,3,[1,2,3,4,5,6],r'$|\delta(R)|$'),
               ('competing','(c) U boundary width',4.35,3,9,6,[3,4,5,6,7,8,9],r'$|\delta(U)|$')]
    original = next(r for r in rows if r['family']=='original')
    for feature,title,top,low,high,anchor,ticks,label in configs:
        parts.append(f'\\node[anchor=west,font=\\fontsize{{9}}{{10}}\\selectfont\\bfseries] at (0,{top}) {{{title}}};')
        for outcome,offset,height,ymax,yticks,ylabel in [('exactdecimal',2.37,2.05,27,'0,10,20',r'Exact OPT'),('ratiodecimal',3.77,1.12,1.10,'0,0.5,1',r'BC/OPT')]:
            pad = 0.45 if feature=='internal' else 0.13
            opts = [f'at={{(0.95cm,{top-offset}cm)}}','anchor=south west','scale only axis','width=6.55cm',f'height={height}cm',
                    'axis lines=left','axis line style={black!80,line width=0.5pt}','tick style={black!70,line width=0.4pt}',
                    r'tick label style={font=\fontsize{8}{9}\selectfont}',r'label style={font=\fontsize{8}{9}\selectfont}',
                    'ymajorgrids=true','grid style={black!10,line width=0.3pt}',f'xmin={low-pad}',f'xmax={high+pad}',
                    'ymin=-0.06' if outcome=='ratiodecimal' else 'ymin=0',f'ymax={ymax}',f'ytick={{{yticks}}}',
                    f'xtick={{{",".join(map(str,ticks))}}}',f'ylabel={{{ylabel}}}','clip=false',
                    'scaled ticks=false',r'/pgf/number format/fixed',r'/pgf/number format/precision=2']
            if outcome=='exactdecimal': opts += ['xticklabels={}','xtick style={draw=none}']
            else: opts += [f'xlabel={{{label}}}',r'xlabel style={yshift=1pt}']
            parts.append('\\begin{axis}[\n'+',\n'.join(opts)+'\n]')
            parts.append(f'\\draw[black!35,densely dotted,line width=0.5pt] (axis cs:{anchor},0)--(axis cs:{anchor},{ymax});')
            if outcome=='ratiodecimal':
                parts.append(f'\\draw[black!25,line width=0.4pt] (axis cs:{low-pad},1)--(axis cs:{high+pad},1);')
            if feature=='internal' and outcome=='exactdecimal':
                parts.append(r'\node[anchor=west,font=\fontsize{7}{8}\selectfont,text=black!65] at (axis cs:10,5) {At 15 edges: seed gap $=16/443$};')
            if feature=='boundary' and outcome=='exactdecimal':
                parts.append(r'\fill[black,opacity=0.035] (axis cs:3.5,0) rectangle (axis cs:6.13,27);')
                parts.append(r'\node[anchor=north,font=\fontsize{7}{8}\selectfont,text=black!65] at (axis cs:4.8,26.7) {$R$ infeasible at $b=3$};')
            for budget in (['fixed','matching'] if feature=='boundary' else ['fixed']):
                color,style = ('RoyalBlue','solid') if budget=='fixed' else ('ForestGreen','dashed')
                for seed in (1,0):
                    group = [r for r in rows if r['family']==feature and seed in json.loads(r['seeds']) and (r['budget']==3 if budget=='fixed' else r['budget']==int(r['rboundarysize']))]
                    points = [(int(r['level']),r[outcome]) for r in group]+[(anchor,original[outcome])]
                    points.sort()
                    coords = ' '.join(f'({x},{y:.14g})' for x,y in points)
                    mark = 'square' if seed==1 else '*'
                    size = ('2.4pt' if seed==1 else '1.3pt') if budget=='fixed' else ('1.7pt' if seed==1 else '0.7pt')
                    width = '0.8pt' if budget=='fixed' else '0.6pt'
                    parts.append(f'\\addplot[color={color},{style},line width={width},mark={mark},mark size={size},mark options={{solid}}] coordinates {{{coords}}};')
            parts.append(f'\\addplot[only marks,color=black,mark=diamond*,mark size=2.6pt] coordinates {{({anchor},{original[outcome]:.14g})}};')
            parts.append(r'\end{axis}')
    parts += [r'\node[anchor=west,font=\fontsize{7}{8}\selectfont] at (0,-0.40) {BC is zero in 40 of 44 cases; all non-baseline graphs are synthetic.};',r'\end{tikzpicture}',r'}']
    tex = '\n'.join(parts)+'\n'
    assert '%' not in tex
    (folder/'fig9.tex').write_text(tex)

def main():
    start = perf_counter()
    rows = extract()
    textfiles(rows)
    figure(rows)
    print('PASS: 34 graphs reconstructed; 44 saved cut and partition replays; no optimization called')
    print('PASS: all 44 report rows, 17 conditional summaries, 34 modification logs and 4 provenance hashes')
    print('BC: 4 positive, 1 optimal, 40 zero; R: 28 optimal, 10 feasible/suboptimal, 6 infeasible; U: 4 optimal, 40 infeasible')
    print('Updated figures/fig9.tex and the Washington interpretation/evidence entries in data/records.json')
    print(f'Analysis runtime: {perf_counter()-start:.3f} seconds')

if __name__=='__main__':
    main()
