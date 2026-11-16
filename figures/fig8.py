# Competing bottleneck phase diagram
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
fills = {0:'black!10',1:'black!40'}

def bounds(values):
    return [values[0]-(values[1]-values[0])/2]+[(a+b)/2 for a,b in zip(values,values[1:])]+[values[-1]+(values[-1]-values[-2])/2]

with (root/'data/results/competition.csv').open(newline='') as source:
    rows = list(csv.DictReader(source))
expected = {(name,m,h) for name,(r,k,widths) in families.items() for m in sizes for h in widths}
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
assert len(rows)==70 and set(records)==expected

parts = [r'\resizebox{0.98\columnwidth}{!}{',r'\begin{tikzpicture}']
xbounds = bounds(sizes)
for name,(r,k,widths) in families.items():
    offset = '4.95cm' if name=='A' else '0'
    label = 'a' if name=='A' else 'b'
    parts.extend([r'\begin{axis}[',
        'width=8.55cm,height=4.45cm,',
        f'at={{(0,{offset})}},anchor=south west,',
        'axis lines=left,axis on top,',
        'axis line style={black,line width=0.5pt},',
        'tick style={black,line width=0.45pt},',
        r'tick label style={font=\scriptsize},label style={font=\small},',
        'xmin=10,xmax=33,ymin=2.5,ymax=13,',
        'xtick={12,16,20,24,30},',
        'ytick={'+','.join(map(str,widths))+'},',
        'xlabel={Competing Block Size},ylabel={Gateway Width},',
        r'legend style={draw=none,fill=none,font=\scriptsize,legend columns=2,at={(0.5,1.05)},anchor=south,column sep=8pt},',
        'clip=true',']'])
    ybounds = bounds(widths)
    for i,m in enumerate(sizes):
        for j,h in enumerate(widths):
            fill = fills[records[name,m,h]]
            parts.append(f'\\path[fill={fill},draw=white,line width=0.25pt] '
                         f'(axis cs:{xbounds[i]:g},{ybounds[j]:g}) rectangle '
                         f'(axis cs:{xbounds[i+1]:g},{ybounds[j+1]:g});')
    curve = f'(-(2*(60-x)-{k})+sqrt((2*(60-x)-{k})^2+8*{k}*(60-x)*x/{r}))/2'
    parts.append(r'\addplot[black,line width=0.55pt,mark=none,domain=10:33,samples=201,forget plot] {'+curve+'};')
    if name=='A':
        for ratio,fill in fills.items():
            parts.append(f'\\addlegendimage{{area legend,draw=none,fill={fill}}}')
            parts.append(r'\addlegendentry{$\mathrm{BC}/\mathrm{OPT}='+str(ratio)+'$}')
    parts.append(r'\node[anchor=north west,font=\small,inner sep=2pt] at (rel axis cs:0.01,0.99) {('+label+')};')
    parts.append(r'\end{axis}')
parts.extend([r'\end{tikzpicture}','}'])
text = '\n'.join(parts)+'\n'
assert '%' not in text and text.count(' rectangle ')==70
output.write_text(text)
print(output)
print('Records represented: 70/70 (35 per panel)')
print('Analytical boundary: 70/70 outcomes correctly separated; no equality cases')
