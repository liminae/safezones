# Saved reports and reproducibility records
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
path = root/'data/records.json'
def load():
    return json.loads(path.read_text())
def text(name):
    return load()['reports'][name]
def exists(name):
    return name in load()['reports']
def guard(destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Output already exists: '+str(destination)+'; use --output with a separate directory')
    return destination
def prepare(directory,files=(),reports=()):
    destination = Path(directory) if directory is not None else root/'data/results'
    for name in files: guard(destination/name)
    for name in reports:
        if directory is None:
            raise FileExistsError('Published reports are read-only: '+str(path)+'; use --output with a separate directory')
        guard(destination/(name+'.txt'))
    destination.mkdir(parents=True,exist_ok=True)
    return destination
def cachefolder(directory):
    destination = Path(directory).resolve()
    if destination.is_relative_to(root.resolve()):
        raise ValueError('Use a cache directory outside the repository')
    return destination
def save(name,value,exclusive=True,directory=None):
    if directory is not None:
        if Path(name).name!=name: raise ValueError('Report name must be a filename')
        with (Path(directory)/(name+'.txt')).open('x') as stream: stream.write(value)
        return
    saved = load()
    if name in saved['reports']: raise FileExistsError('Saved report exists: '+name)
    if path.resolve()==(root/'data/records.json').resolve():
        raise FileExistsError('Published records are read-only; save to a separate directory')
    saved['reports'][name] = value
    path.write_text(json.dumps(saved,indent=2)+'\n')
def data(name):
    saved = load()
    name = saved['paths'].get(str(name),str(name))
    if name.startswith('report:'): return saved['reports'][name[7:]].encode()
    if name.startswith('source:'): return saved['datasets'][name[7:]].encode()
    return (root/name).read_bytes()
def verify(name,digest):
    if hashlib.sha256(data(name)).hexdigest()==digest: return
    rows = [r for r in load()['sources'] if r['previous']==name and r['previoussha256']==digest]
    assert len(rows)==1,(name,digest)
    row = rows[0]
    assert hashlib.sha256((root/row['current']).read_bytes()).hexdigest()==revisions().get(row['current'],row['currentsha256']),name
def revisions():
    current = {}
    figures = root/'figures/checks.json'
    if figures.exists():
        revised = json.loads(figures.read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==revised['records'],'Historical records changed'
        current = revised['files']
        allowed = {'code/records.py','code/tests.py'}|{f'figures/fig{n}.{suffix}' for n in (7,8,9) for suffix in ('py','tex')}
        assert set(current)==allowed,'Unexpected figure revision paths'
        assert set(revised['reference'])=={f'figures/fig{n}.tex' for n in (7,8,9)}
        for name,digest in revised['reference'].items():
            assert current[name]==digest,name
    safety = root/'code/checks.json'
    if safety.exists():
        revised = json.loads(safety.read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==revised['records'],'Historical records changed'
        assert hashlib.sha256(figures.read_bytes()).hexdigest()==revised['figures'],'Figure checks changed'
        previous = {row['current']:row['currentsha256'] for row in load()['sources']}
        previous.update(current)
        for name,row in revised['files'].items():
            assert Path(name).parent==Path('code') and Path(name).suffix=='.py','Unexpected safety revision path'
            assert previous[name]==row['previous'],name
            current[name] = row['current']
    for name,digest in current.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    return current
def check():
    saved = load()
    current = revisions()
    for name,digest in saved['artifacts'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==current.get(name,digest),name
    for name,digest in saved['reporthashes'].items():
        assert hashlib.sha256(saved['reports'][name].encode()).hexdigest()==digest,name
    for name,digest in saved.get('datasethashes',{}).items():
        assert hashlib.sha256(saved['datasets'][name].encode()).hexdigest()==digest,name
    for row in saved['sources']:
        assert hashlib.sha256((root/row['current']).read_bytes()).hexdigest()==current.get(row['current'],row['currentsha256']),row['current']
    return len(saved['artifacts'])+len(saved['reporthashes'])+len(saved.get('datasethashes',{}))
if __name__=='__main__':
    if len(sys.argv)==2: print(text(sys.argv[1]),end='')
    else:
        print('Saved reports: '+', '.join(load()['reports']))
        print('Read a report: python code/records.py NAME')
        print('Run all regressions: python -B code/tests.py')
        print('Numerical environment: Python 3.13.12; networkx==3.7')
