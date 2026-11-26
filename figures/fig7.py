# Final threshold sweep figure
import argparse
import csv
from fractions import Fraction
from pathlib import Path

root = Path(__file__).resolve().parent.parent
commuting = root/'data/results/commutingsweep.csv'
air = root/'data/results/airsweep.csv'
output = root/'figures/fig7.tex'
colors = {1:'RoyalBlue',2:'BrickRed',3:'ForestGreen'}
def load(path,prior,target):
    result = {1:[],2:[],3:[]}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            b = int(row['budget'])
            if row['prior']==prior and row['target']==target and b in result:
                assert row['status']=='exact' and not row['searcherror']
                x,y = Fraction(row['threshold']),Fraction(row['exact'])
                assert abs(float(x)-float(row['thresholddecimal']))<1e-14
                assert abs(float(y)-float(row['exactdecimal']))<1e-12
                result[b].append((x,y))
    for points in result.values():
        points.sort()
        assert len({x for x,y in points})==len(points)
    return result

def coords(points,counts):
    assert sum(counts)==len(points)
    words = []
    for x,y in points:
        label = repr(float(x)) if x==Fraction(46,35) else f'{float(x):.8g}'
        words.append(f'({label},{float(y):.8g})')
    lines,start = [],0
    for count in counts:
        lines.append(' '.join(words[start:start+count]))
        start += count
    return '\n'.join(lines)

def figure():
    county = load(commuting,'uniform','counties')
    airport = load(air,'uniform','airports')
    parts = [r'\resizebox{0.98\columnwidth}{!}{%',r'\begin{tikzpicture}','']
    for panel,values in (('a',county),('b',airport)):
        countyplot = panel=='a'
        parts += [r'\begin{axis}[','width=8.55cm,','height=4.45cm,',
                  'at={(0,4.95cm)},' if countyplot else 'at={(0,0)},',r'''anchor=south west,
axis lines=left,
axis line style={black!80,line width=0.5pt},
tick style={black!70,line width=0.45pt},
tick label style={font=\scriptsize},
label style={font=\small},
legend style={
draw=none,
fill=none,
font=\scriptsize,
legend columns=3,
at={(0.5,1.03)},
anchor=south,
column sep=8pt
},
ymajorgrids=true,
grid style={black!10,line width=0.35pt},''',
                  'xmin=0.75,' if countyplot else 'xmin=1,',
                  'xmax=1.50,' if countyplot else 'xmax=60,','ymin=0,',
                  'ymax=44,' if countyplot else 'ymax=27,',
                  'xtick={0.75,1.00,1.25,1.50},' if countyplot else 'xtick={1,10,20,30,40,50,60},',
                  'ytick={0,10,20,30,40},' if countyplot else 'ytick={0,5,10,15,20,25},',
                  r'xlabel={Reliability Multiplier $\kappa$},' if countyplot else r'xlabel={Passenger Threshold $\tau$},',
                  'ylabel={Expected protected counties},' if countyplot else 'ylabel={Expected protected airports},','clip=false',']']
        counts = (5,4,5,4,5,4,4,4,2) if countyplot else (5,)*12
        for b in (1,2,3):
            parts += [f'\\addplot+[color={colors[b]},line width=0.95pt,mark=none] coordinates {{',
                      coords(values[b],counts),'};',f'\\addlegendentry{{$b={b}$}}','']
        parts += [r'\node[anchor=north west,font=\small] at (rel axis cs:0.01,0.99) {('+panel+')};',r'\end{axis}','']
    parts += [r'\end{tikzpicture}%',r'}']
    return '\n'.join(parts)

def main():
    parser = argparse.ArgumentParser(description='Reconstruct Figure 7 from saved exact sweep records')
    parser.add_argument('--check',action='store_true',help='Verify saved TeX without writing any files')
    args = parser.parse_args()
    text = figure().encode()
    if args.check:
        if output.read_bytes()!=text: raise SystemExit('FAIL Figure 7 differs from its generator')
        print('PASS Figure 7: 291 saved points, exact rational threshold ordering, non-writing source agreement')
    else:
        output.write_bytes(text)
        print(output)
if __name__=='__main__':
    main()
