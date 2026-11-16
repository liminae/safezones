import csv
from pathlib import Path

root = Path(__file__).resolve().parent.parent
commuting = root / 'data/results/commutingsweep.csv'
air = root / 'data/results/airsweep.csv'
output = root / 'figures/fig7.tex'

def load(path, prior, target):
    result = {1: [], 2: [], 3: []}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            b = int(row['budget'])
            if row['prior'] == prior and row['target'] == target and b in result:
                result[b].append(
                    (float(row['thresholddecimal']), float(row['exactdecimal']))
                )
    for b in result:
        result[b].sort()
    return result

def coords(points):
    return ' '.join(f'({x:.8g},{y:.8g})' for x, y in points)

county = load(commuting, 'uniform', 'counties')
airport = load(air, 'uniform', 'airports')

colors = {
    1: 'RoyalBlue',
    2: 'BrickRed',
    3: 'ForestGreen'
}

parts = [r'''\resizebox{0.98\columnwidth}{!}{%
\begin{tikzpicture}

\begin{axis}[
width=8.55cm,
height=4.45cm,
at={(0,4.95cm)},
anchor=south west,
axis lines=left,
axis line style={black!80,line width=0.5pt},
tick style={black!70,line width=0.45pt},
tick label style={font=\scriptsize},
label style={font=\small},
legend style={draw=none,fill=none,font=\scriptsize,legend columns=3,at={(0.5,1.03)},anchor=south,column sep=8pt},
ymajorgrids=true,
grid style={black!10,line width=0.35pt},
xmin=0.75,
xmax=1.50,
ymin=0,
ymax=44,
xtick={0.75,1.00,1.25,1.50},
ytick={0,10,20,30,40},
xlabel={Reliability Multiplier $\kappa$},
ylabel={Protected Counties},
clip=false
]''']

for b in (1, 2, 3):
    parts.append(
        f'\\addplot+[color={colors[b]},line width=0.95pt,mark=none] '
        f'coordinates {{{coords(county[b])}}};'
    )
    parts.append(f'\\addlegendentry{{$b={b}$}}')

parts.append(r'''\node[anchor=north west,font=\small] at (rel axis cs:0.01,0.99) {(a)};
\end{axis}

\begin{axis}[
width=8.55cm,
height=4.45cm,
at={(0,0)},
anchor=south west,
axis lines=left,
axis line style={black!80,line width=0.5pt},
tick style={black!70,line width=0.45pt},
tick label style={font=\scriptsize},
label style={font=\small},
legend style={draw=none,fill=none,font=\scriptsize,legend columns=3,at={(0.5,1.03)},anchor=south,column sep=8pt},
ymajorgrids=true,
grid style={black!10,line width=0.35pt},
xmin=1,
xmax=60,
ymin=0,
ymax=27,
xtick={1,10,20,30,40,50,60},
ytick={0,5,10,15,20,25},
xlabel={Passenger Threshold $\tau$},
ylabel={Protected Airports},
clip=false
]''')

for b in (1, 2, 3):
    parts.append(
        f'\\addplot+[color={colors[b]},line width=0.95pt,mark=none] '
        f'coordinates {{{coords(airport[b])}}};'
    )
    parts.append(f'\\addlegendentry{{$b={b}$}}')

parts.append(r'''\node[anchor=north west,font=\small] at (rel axis cs:0.01,0.99) {(b)};
\end{axis}

\end{tikzpicture}%
}
''')

output.write_text('\n'.join(parts))
print(output)
