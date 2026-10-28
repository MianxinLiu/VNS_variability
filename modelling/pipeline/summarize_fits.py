"""Assemble regional/current QC and a staged single-seed Figure 3."""
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
MODEL=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MODEL/'core'))
import dmf_rww_ei_jax_gpu_fc_only as model

OUT=MODEL/'output'/'fit_250'
PYGPU=sys.executable

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int,default=50)
    parser.add_argument('--input-npz',type=Path,required=True)
    parser.add_argument('--fit-dir',type=Path,default=OUT)
    parser.add_argument('--reuse-stability-from',type=Path,default=None)
    parser.add_argument('--exclude-candidate',type=int,action='append',default=[])
    parser.add_argument('--analysis-name',default=None)
    args=parser.parse_args()
    fit_dir=args.fit_dir
    paths=sorted(fit_dir.glob('selected_???.json'))
    records=[json.loads(p.read_text()) for p in paths]
    records=[r for r in records if int(r['candidate_id']) not in args.exclude_candidate]
    if len(records)<args.count:
        raise ValueError(f'Expected at least {args.count} eligible solutions, found {len(records)}')
    records=records[:args.count]
    analysis=fit_dir/(args.analysis_name or f'analysis_{args.count:02d}')
    analysis.mkdir(parents=True,exist_ok=True)
    data=np.load(args.input_npz,allow_pickle=True)
    labels=data['roi_labels'];left,right=model.default_motor_rois(labels);motor=np.asarray(left+right,int)
    sources=np.setdiff1d(np.arange(len(labels)),motor)
    sc=[model.prepare_sc(model.sanitize_matrix(model.finite_nanmean(data['SC'][:,i],axis=0))) for i in range(2)]
    weights=np.mean([s[motor].mean(axis=0) for s in sc],axis=0);weights[motor]=0;weights/=weights.sum()
    metrics=[];regional=[];currents=[];parameters=[]
    for r in records:
        cid=r['candidate_id'];seed=r['seed']
        for i,session in enumerate(('baseline','followup6w')):
            rec=r['results'][session];s=np.asarray(rec['summary']);p=np.asarray(rec['params'])
            metrics.append(dict(candidate_id=cid,seed=seed,session=session,fc_r=rec['fc_r'],
                SE=rec['SE'],DFA=rec['DFA'],I_E=s[2,motor].mean(),
                S_E_over_S_I=s[6,motor].mean(),motor_r_E=s[4,motor].mean()))
            regional.append(dict(candidate_id=cid,session=session,I_E=s[2]@weights,
                S_E_over_S_I=s[6]@weights,r_E=s[4]@weights))
            currents.append(dict(candidate_id=cid,seed=seed,session=session,
                local_E=(p[1]*.15*s[0,motor]).mean(),network_E=(p[0]*.15*(sc[i]@s[0])[motor]).mean(),
                inhibition_magnitude=(p[3]*s[1,motor]).mean(),simulated_I_E=s[2,motor].mean()))
        change=np.asarray(r['results']['followup6w']['params'])-np.asarray(r['results']['baseline']['params'])
        for name,delta in zip(('G','w_EE','w_EI','w_IE'),change):
            parameters.append(dict(candidate_id=cid,parameter=name,delta=delta))
    for name,rows in (('all_seed_metrics',metrics),('regional',regional),('current_terms',currents),
                      ('parameters',parameters)):
        write_csv(analysis/f'{name}.csv',rows)
    stability_output=analysis/'stability.csv'
    if not stability_output.exists():
        reused=[]
        if args.reuse_stability_from is not None:
            with args.reuse_stability_from.open(newline='') as f:
                reused=[row for row in csv.DictReader(f)
                        if int(row['candidate_id']) in {int(r['candidate_id']) for r in records}]
        reused_ids={int(row['candidate_id']) for row in reused}
        requested_ids={int(r['candidate_id']) for r in records}
        if not reused_ids.issubset(requested_ids):
            raise ValueError('Reused stability includes candidates outside the selected fit set')
        missing_ids=sorted(requested_ids-reused_ids)
        fresh=[]
        if missing_ids:
            fresh_output=analysis/'stability_new_only.csv'
            if not fresh_output.exists():
                subprocess.run([PYGPU,str(Path(__file__).with_name('analyze_fixed_point.py')),
                    '--input-npz',str(args.input_npz),'--fit-dir',str(fit_dir),'--output',str(fresh_output),
                    '--candidate-ids',*[str(cid) for cid in missing_ids]],check=True,cwd=MODEL.parent)
            with fresh_output.open(newline='') as f:
                fresh=list(csv.DictReader(f))
        rows=reused+fresh
        if len(rows)!=2*len(records):
            raise ValueError(f'Expected {2*len(records)} stability rows, found {len(rows)}')
        rows.sort(key=lambda row:(int(row['selected_id']),
                                0 if row['session']=='baseline' else 1))
        write_csv(stability_output,rows)

if __name__=='__main__':main()
