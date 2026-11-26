# Final discrete competing bottleneck figure
import argparse
import csv
from fractions import Fraction
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'code'))
from competition import margin
output = root/'figures/fig8.tex'
families = dict(A=(6,2,(3,4,5,6,8,10,12)),B=(9,3,(4,5,6,7,8,10,12)))
sizes = (12,16,20,24,30)
def load():
    with (root/'data/results/competition.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    records = {}
    for row in rows:
        name,m,h = row['family'],int(row['b_size']),int(row['gateway_width'])
        r,k,widths = families[name]
        assert int(row['r'])==r and int(row['k'])==k and int(row['a_size'])==60-m
        ratio = Fraction(row['bc_ratio'])
        assert ratio in (0,1) and ratio==Fraction(row['bc_ratio_fraction'])
        delta = margin(r,60-m,m,k,h)
        assert delta!=0 and ratio==int(delta<0)
        key = name,m,h
        assert key not in records
        records[key] = int(ratio)
    assert len(rows)==70 and set(records)=={(name,m,h) for name,(_,_,widths) in families.items() for m in sizes for h in widths}
    return records

def tiles(color,column,rows):
    if not rows: return []
    if len(rows)==1: return [f'\\tile{{{color}}}{{{column}}}{{{rows[0]}}}']
    labels = f'{rows[0]},...,{rows[-1]}' if color=='succ' and len(rows)>=4 else ','.join(map(str,rows))
    return [r'\foreach \r in {'+labels+'} {\\tile{'+color+'}{'+str(column)+r'}{\r}}']

def transition(r,k,widths):
    heights = [sum(margin(r,60-m,m,k,h)>0 for h in widths) for m in sizes]
    points = [(0,heights[0])]
    for i in range(1,len(sizes)):
        if heights[i]!=heights[i-1]: points += [(i,heights[i-1]),(i,heights[i])]
    return points+[(len(sizes),heights[-1])]

def figure():
    records = load()
    parts = [r'''\resizebox{0.98\columnwidth}{!}{
\begin{tikzpicture}[x=1.25cm,y=0.62cm,font=\small]

\definecolor{fail}{RGB}{225,194,177}
\definecolor{succ}{RGB}{188,204,224}

\newcommand{\tile}[3]{
\path[fill=#1,draw=white,line width=0.4pt]
(#2-1,#3-1) rectangle (#2,#3);
}

\begin{scope}[shift={(0,19.15)}]
\def\legy{0.32}

\def\cA{0.95}
\def\cB{2.50}
\def\cC{4.05}

\draw[fill=fail,draw=black!60,line width=0.35pt]
(\cA-0.42,\legy-0.12) rectangle (\cA-0.10,\legy+0.12);
\node[anchor=west,font=\scriptsize] at (\cA,\legy) {Fail};

\draw[fill=succ,draw=black!60,line width=0.35pt]
(\cB-0.52,\legy-0.12) rectangle (\cB-0.20,\legy+0.12);
\node[anchor=west,font=\scriptsize] at (\cB-0.10,\legy) {Succeed};

\draw[black,line width=0.8pt] (\cC-0.45,\legy) -- (\cC-0.05,\legy);
\node[anchor=west,font=\scriptsize] at (\cC+0.05,\legy) {Theory};
\end{scope}
''']
    for name,(r,k,widths) in families.items():
        offset,label = ('10.55','a') if name=='A' else ('0','b')
        parts += [f'\\begin{{scope}}[shift={{(0,{offset})}}]','',
                  r'\node[anchor=center,font=\small] at (-1.05,7.55) {('+label+')};',
                  r'\node[anchor=center,font=\small] at (2.5,7.55) {Family '+name+'};','']
        for column,m in enumerate(sizes,1):
            for value,color in ((0,'fail'),(1,'succ')):
                parts += tiles(color,column,[i for i,h in enumerate(widths,1) if records[name,m,h]==value])
            parts.append('')
        parts += [r'\draw[black!70,line width=0.5pt] (0,0) rectangle (5,7);','',r'\foreach \x/\lab in {',
                  ',\n'.join(f'{i+0.5:g}/{m}' for i,m in enumerate(sizes)),r'''}{
\node[anchor=north,font=\scriptsize] at (\x,-0.22) {\lab};
}

\foreach \y/\lab in {''',
                  ',\n'.join(f'{i+0.5:g}/{h}' for i,h in enumerate(widths)),r'''}{
\node[anchor=east,font=\scriptsize] at (-0.18,\y) {\lab};
}

\node[anchor=north,font=\small] at (2.5,-0.78) {Block Size $m$};
\node[rotate=90,anchor=south,font=\small] at (-0.72,3.5) {Gateway $h$};

\draw[black,line width=0.8pt]''',
                  ' --\n'.join(f'({x},{y})' for x,y in transition(r,k,widths))+';','',r'\end{scope}','']
    parts += [r'\end{tikzpicture}',r'}','']
    return '\n'.join(parts)

def main():
    parser = argparse.ArgumentParser(description='Reconstruct the discrete Figure 8 from all 70 saved classifications')
    parser.add_argument('--check',action='store_true',help='Verify saved TeX without writing any files')
    args = parser.parse_args()
    text = figure().encode()
    if args.check:
        if output.read_bytes()!=text: raise SystemExit('FAIL Figure 8 differs from its generator')
        print('PASS Figure 8: all 70 classifications and discrete theoretical transitions, non-writing source agreement')
    else:
        output.write_bytes(text)
        print(output)
if __name__=='__main__':
    main()
