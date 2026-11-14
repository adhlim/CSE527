"""Judge the ten pre-registered predictions from summary.csv (setting A, seed-mean AUC@10). Written before the run.
usage: python md_predictions.py RESULTS_DIR"""
import sys
from pathlib import Path
import pandas as pd
S = pd.read_csv(Path(sys.argv[1]) / 'summary.csv'); S = S[S.setting == 'A'].set_index('config')['auc@10_mean']
need = [f'C{i}' for i in range(15)]; missing = [c for c in need if c not in S.index]
assert not missing, f'missing configurations in setting A: {missing}'
v = S.to_dict()
PRED = [('P1', 'C1 within 0.5 of C0', abs(v['C1'] - v['C0']) <= 0.5, v['C1'] - v['C0']),
        ('P2', 'C2 within 1.5 of C0', abs(v['C2'] - v['C0']) <= 1.5, v['C2'] - v['C0']),
        ('P3', 'C6 - C7 >= 2.0', v['C6'] - v['C7'] >= 2.0, v['C6'] - v['C7']),
        ('P4', 'C4 >= C3 and C5 >= C3', v['C4'] >= v['C3'] and v['C5'] >= v['C3'], min(v['C4'], v['C5']) - v['C3']),
        ('P5', '|C8 - C6| <= 1.0', abs(v['C8'] - v['C6']) <= 1.0, v['C8'] - v['C6']),
        ('P6', 'C9 - C6 >= 1.0', v['C9'] - v['C6'] >= 1.0, v['C9'] - v['C6']),
        ('P7', '|C10 - C9| <= 1.0', abs(v['C10'] - v['C9']) <= 1.0, v['C10'] - v['C9']),
        ('P8', '|C12 - C9| <= 1.5', abs(v['C12'] - v['C9']) <= 1.5, v['C12'] - v['C9']),
        ('P9', 'C13 < C9', v['C13'] < v['C9'], v['C13'] - v['C9']),
        ('P10', 'C14 < C2', v['C14'] < v['C2'], v['C14'] - v['C2'])]
out = pd.DataFrame([dict(prediction=p, statement=s, confirmed=bool(ok), observed_difference=round(float(d), 2)) for p, s, ok, d in PRED])
out.to_csv(Path(sys.argv[1]) / 'predictions.csv', index=False); print(out.to_string(index=False)); print('confirmed', int(out.confirmed.sum()), 'of 10')
