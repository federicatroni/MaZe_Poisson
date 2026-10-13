"""Write a MaZe restart csv (type,x,y,z [A],vx,vy,vz [a.u.]) from the last COMPLETE frame of a solute.csv. Prints that frame's iter.
Usage: make_restart.py solute.csv out.csv [N=40]"""
import sys, numpy as np
f, out = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
lines = open(f).read().splitlines()[1:]
rows = [l.split(',') for l in lines[-3*N:] if l.count(',') == 11]
its = sorted({int(float(r[1])) for r in rows})
for it in reversed(its):
    fr = [r for r in rows if int(float(r[1])) == it]
    if len(fr) == N: break
fr.sort(key=lambda r: int(float(r[2])))
B = 0.529177210903
with open(out, 'w') as g:
    g.write('type,x,y,z,vx,vy,vz\n')
    for r in fr:
        q = float(r[0]); x, y, z = (float(v)*B for v in r[3:6])
        g.write(f"{'P' if q > 0 else 'M'},{x:.10f},{y:.10f},{z:.10f},{r[6]},{r[7]},{r[8]}\n")
print(it)
