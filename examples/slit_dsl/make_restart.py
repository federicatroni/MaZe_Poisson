"""Write a MaZe restart csv (type,x,y,z [A],vx,vy,vz [a.u.]) from the last COMPLETE frame of a solute output. Prints that frame's iter.
Usage: make_restart.py solute.csv|solute.bin out.csv [N=40]   (a .bin is read with maze_poisson.myio.output.read_maze_bin)"""
import os, sys, numpy as np
f, out = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
B = 0.529177210903
if f.endswith('.bin'):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # repo root
    from maze_poisson.myio.output import read_maze_bin
    s = read_maze_bin(f, float64=True)
    c = s.groupby('iter').size(); it = int(c.index[c.values == N].max())
    fr = s[s.iter == it].sort_values('particle')
    rows = [(r.charge, r.x, r.y, r.z, repr(float(r.vx)), repr(float(r.vy)), repr(float(r.vz))) for r in fr.itertuples()]
else:
    lines = open(f).read().splitlines()[1:]
    raw = [l.split(',') for l in lines[-3*N:] if l.count(',') == 11]
    its = sorted({int(float(r[1])) for r in raw})
    for it in reversed(its):
        fr = [r for r in raw if int(float(r[1])) == it]
        if len(fr) == N: break
    fr.sort(key=lambda r: int(float(r[2])))
    rows = [(float(r[0]), float(r[3]), float(r[4]), float(r[5]), r[6], r[7], r[8]) for r in fr]
with open(out, 'w') as g:
    g.write('type,x,y,z,vx,vy,vz\n')
    for q, x, y, z, vx, vy, vz in rows:
        g.write(f"{'P' if q > 0 else 'M'},{x*B:.10f},{y*B:.10f},{z*B:.10f},{vx},{vy},{vz}\n")
print(it)
