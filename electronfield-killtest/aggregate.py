"""Summarize the five paired experiments; regenerate aggregate.json and summary.csv."""
import csv
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
files=[root/'results.json'] + sorted(root.glob('results_seed*.json'))
if len(files)!=5:
    raise RuntimeError(f'Expected 5 result files, found {len(files)}')
rs=[json.loads(f.read_text()) for f in files]
arms=['energy_only','energy_density_potential']
metrics=['energy_mae_hartree','density_l1_fraction','potential_weighted_rmse_hartree','scf_residual_l1_fraction']
records=[]; agg={"seeds":[r['setup']['seed'] for r in rs],"splits":{},"notes":"Mean/std over five independent dataset+parameter initialization seeds; no statistical significance established."}
for split in ['train','test_id','test_ood']:
    agg['splits'][split]={}
    for metric in metrics:
        aa={arm: np.array([r['arms'][arm]['metrics'][split][metric] for r in rs]) for arm in arms}
        a,b=aa.values()
        agg['splits'][split][metric]={arm:{"mean":float(vals.mean()),"std":float(vals.std(ddof=1)),"values":vals.tolist()}
                                       for arm,vals in aa.items()}
        agg['splits'][split][metric].update({"relative_gain_percent":float(100*(1-b.mean()/a.mean())) if a.mean()!=0 else None,
                                           "paired_wins_out_of_5":int(np.sum(b<a)),
                                           "mean_paired_difference":float(np.mean(a-b))})
        for i,r in enumerate(rs):
            records.append({'seed':r['setup']['seed'],'split':split,'metric':metric,
                            'energy_only':float(a[i]),'energy_density_potential':float(b[i]),
                            'field_minus_energy':float(b[i]-a[i])})
(root/'aggregate.json').write_text(json.dumps(agg,indent=2)+'\n')
with (root/'summary.csv').open('w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=records[0].keys()); wr.writeheader();wr.writerows(records)
for split in ['test_id','test_ood']:
    print(split)
    for metric in metrics[:3]:
        z=agg['splits'][split][metric]
        print(metric,'energy',round(z['energy_only']['mean'],5),'fields',round(z['energy_density_potential']['mean'],5),
              'improvement_pct',round(z['relative_gain_percent'],1),'wins',z['paired_wins_out_of_5'])
