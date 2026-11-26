# Final Washington pilot figure
import argparse
import csv
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


panels = (('internal','(a) R Internal Connectivity',9,36,30,(9,15,21,27,30,33,36),8.05),
          ('boundary','(b) R Boundary Width',1,6,3,(1,2,3,4,5,6),5.00),
          ('competing','(c) U Boundary Width',3,9,6,(3,4,5,6,7,8,9),1.95))
def trace(rows,feature,matching,seed):
    original = next(r for r in rows if r['family']=='original')
    anchor = next(p[4] for p in panels if p[0]==feature)
    selected = [r for r in rows if r['family']==feature and seed in json.loads(r['seeds'])
                and r['budget']==(int(r['rboundarysize']) if matching else 3)]
    points = [(int(r['level']),Fraction(r['exact']),r) for r in selected]+[(anchor,Fraction(original['exact']),original)]
    points.sort(key=lambda p:p[0])
    assert len({p[0] for p in points})==len(points)
    return points

def figure(rows):
    parts = [r"""\begin{tikzpicture}[x=1cm,y=1cm]
\fill[RoyalBlue] (0.64,10.66) circle (1.7pt);
\node[anchor=west,inner sep=0pt,font=\fontsize{8}{9}\selectfont] at (0.86,10.66) {Fixed Budget};
\fill[ForestGreen] (3.04,10.66) circle (1.7pt);
\node[anchor=west,inner sep=0pt,font=\fontsize{8}{9}\selectfont] at (3.26,10.66) {Matching Budget};
\fill[black] (5.96,10.66)--(6.04,10.75)--(6.12,10.66)--(6.04,10.57)--cycle;
\node[anchor=east,inner sep=0pt,font=\fontsize{8}{9}\selectfont] at (7.40,10.66) {Baseline};
\node[anchor=center,rotate=90,font=\fontsize{8}{9}\selectfont] at (0.20,5.72) {Expected Protected Airports};"""]
    original = next(r for r in rows if r['family']=='original')
    for feature,title,low,high,anchor,ticks,bottom in panels:
        def xpoint(value): return 1.36+6.04*(value-low)/(high-low)
        def ypoint(value): return bottom+1.32*float(value)/27
        parts += [f'\\node[anchor=west,font=\\fontsize{{9}}{{10}}\\selectfont\\bfseries] at (0.48,{bottom+1.72:.3f}) {{{title}}};',
                  f'\\draw[black!75,line width=0.55pt] (1.360,{bottom+1.32:.3f})--(1.360,{bottom:.3f})--(7.400,{bottom:.3f});']
        for value in (0,10,20):
            y = ypoint(value)
            parts += [f'\\draw[black!65,line width=0.5pt] (1.320,{y:.3f})--(1.360,{y:.3f});',
                      f'\\node[anchor=east,font=\\fontsize{{7.5}}{{8.5}}\\selectfont,text=black!75] at (1.240,{y:.3f}) {{{value}}};']
        for value in ticks:
            x = xpoint(value)
            parts += [f'\\draw[black!70,line width=0.5pt] ({x:.3f},{bottom:.3f})--({x:.3f},{bottom-0.045:.3f});',
                      f'\\node[anchor=north,font=\\fontsize{{7.5}}{{8.5}}\\selectfont,text=black!80] at ({x:.3f},{bottom-0.09:.3f}) {{{value}}};']
        for matching in ((False,True) if feature=='boundary' else (False,)):
            color = 'ForestGreen' if matching else 'RoyalBlue'
            for seed in (0,1):
                points = [(xpoint(x),ypoint(y),row) for x,y,row in trace(rows,feature,matching,seed)]
                coords = ' -- '.join(f'({x:.4f},{y:.4f})' for x,y,row in points)
                parts.append(f'\\draw[{color},line width=0.75pt] {coords};')
                for x,y,row in points:
                    if row['family']=='original': continue
                    if seed==0:
                        parts.append(f'\\fill[{color}] ({x:.4f},{y:.4f}) circle (1.25pt);')
                    else:
                        parts.append(f'\\draw[{color},fill=white,line width=0.75pt] ({x-0.042:.4f},{y-0.042:.4f}) rectangle ({x+0.042:.4f},{y+0.042:.4f});')
        x,y = xpoint(anchor),ypoint(Fraction(original['exact']))
        parts.append(f'\\fill[black] ({x-0.095:.4f},{y:.4f})--({x:.4f},{y+0.095:.4f})--({x+0.095:.4f},{y:.4f})--({x:.4f},{y-0.095:.4f})--cycle;')
    parts += [r'\end{tikzpicture}','']
    return '\n'.join(parts)

def main():
    parser = argparse.ArgumentParser(description='Reconstruct the final three-plot Figure 9 from saved Washington results')
    parser.add_argument('--check',action='store_true',help='Verify saved TeX without writing figures or provenance')
    args = parser.parse_args()
    start = perf_counter()
    rows = extract()
    tex = figure(rows).encode()
    output = folder/'fig9.tex'
    if args.check:
        if output.read_bytes()!=tex: raise SystemExit('FAIL Figure 9 differs from its generator')
        print('PASS Figure 9: non-writing source agreement')
    else:
        output.write_bytes(tex)
        print(output)
    print('PASS: 34 graphs reconstructed; all 44 saved cuts, partitions, BC scores and exact values checked; no optimization')
    print('PASS: two placement seeds, fixed/matching budgets, three plots and three measured baseline markers')
    print(f'Figure validation runtime: {perf_counter()-start:.3f} seconds')
if __name__=='__main__':
    main()
