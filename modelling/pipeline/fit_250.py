"""Resume-safe single-seed multirate FC/SE/DFA screening; no FIC."""
import argparse,csv,json,subprocess,sys,tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from skopt import forest_minimize
MODEL=Path(__file__).resolve().parents[1]
REPO=MODEL.parent
sys.path.insert(0,str(MODEL/'core'))
import fit_support as robust
import dmf_rww_ei_jax_gpu_fc_only as base
from simulate_motor_neural import simulate_neural
import multirate_bw as bw

OUTPUT=MODEL/'output'/'fit_250'
N_TR=250
EEGPY=sys.executable
METRIC_SPEC=('native 1000-Hz S_E; 0.5-80-Hz band-pass; '
             'SE fit 1-40 Hz; DFA windows 10-2000 ms by 100 ms')
FOLLOWUP_METHOD=('35 evaluations from four reproducible 5%-of-log-range '
                 'perturbations around baseline plus global exploration; fixed same seed')

def main():
    global OUTPUT, N_TR
    p=argparse.ArgumentParser();p.add_argument('--target-count',type=int,default=51)
    p.add_argument('--input-npz',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=OUTPUT)
    p.add_argument('--n-tr',type=int,default=N_TR)
    p.add_argument('--candidate-seed-base',type=int,default=20261020)
    p.add_argument('--random-state-base',type=int,default=1000)
    a=p.parse_args()
    OUTPUT=a.output_dir
    N_TR=a.n_tr
    if N_TR <= 0:
        raise ValueError('--n-tr must be positive')
    if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    status=OUTPUT/'progress.json'
    progress=json.loads(status.read_text()) if status.exists() else dict(next_candidate=1,accepted=0,rejected=[])
    data=np.load(a.input_npz,allow_pickle=True)
    labels=np.asarray(data['roi_labels'],object)
    left,right=base.default_motor_rois(labels);motor=np.asarray(left+right,np.int32)
    sc=[base.sanitize_matrix(base.finite_nanmean(data['SC'][:,i],axis=0)) for i in range(2)]
    fc=[base.sanitize_matrix(base.finite_nanmean(data['FC'][:,i],axis=0)) for i in range(2)]
    config=argparse.Namespace(n_calls=90,n_initial_points=14,fit_n_tr=N_TR,dt=.001,tr=.700,
        burn_s=30.,fixed_sigma=.001,fixed_i0=.382,bold_input='S_E',
        fit_seed_base=2000,random_state_base=a.random_state_base,
        subject_sides=['R','R','L','R','L','L','R'])
    checkpoint=OUTPUT/'current_candidate.json'
    while progress['accepted']<a.target_count:
        candidate=progress['next_candidate']
        rng=np.random.default_rng(a.candidate_seed_base+candidate)
        seed=int(rng.integers(1000,2000000000))
        args=robust.model_args(config,candidate);args.seed=seed
        record=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(
            candidate_id=candidate,seed=seed,candidate_seed_base=a.candidate_seed_base,
            random_state_base=a.random_state_base,neural_dt=.001,hemo_dt=.01,
            tr_s=config.tr,fixed_i0=config.fixed_i0,burn_s=config.burn_s,
            fit_n_tr=N_TR,validation_n_tr=N_TR,
            results={},attempt=0,baseline_method='90 evaluations fixed single seed',
            followup_method=FOLLOWUP_METHOD,
            selection='both full-duration FC r > .15; SE increases and DFA decreases',
            metric_sampling=METRIC_SPEC)
        if record['candidate_id']!=candidate:raise ValueError('checkpoint mismatch')
        if (record.get('tr_s') != config.tr or record.get('fixed_i0') != config.fixed_i0
                or record.get('fit_n_tr') != N_TR or record.get('validation_n_tr') != N_TR
                or record.get('candidate_seed_base',a.candidate_seed_base) != a.candidate_seed_base
                or record.get('random_state_base',a.random_state_base) != a.random_state_base):
            raise ValueError('Checkpoint TR/I0/duration differs from current configuration; use a new directory')
        # A checkpoint screened with an older metric definition can retain its
        # FC-valid baseline, but every follow-up attempt must be re-screened.
        if record.get('metric_sampling') != METRIC_SPEC:
            record['metric_sampling']=METRIC_SPEC
            record['attempt']=0
            robust.atomic_json(checkpoint,record)
        record['followup_method']=FOLLOWUP_METHOD
        def fit(session_i,n_calls,attempt,x0=None):
            args.n_tr=N_TR; trace=[]
            def objective(x):
                r,_=bw.simulate_eval(sc[session_i],fc[session_i],x,args,seed)
                loss=1-r if np.isfinite(r) else 3.
                trace.append(dict(params=list(map(float,x)),fc_r=float(r),loss=float(loss)))
                if len(trace)%15==0:
                    print(f'candidate {candidate} session {session_i} attempt {attempt}: {len(trace)}/{n_calls}',flush=True)
                return loss
            space=bw.param_space(args)[1]
            kwargs={}
            if x0 is not None:
                # All four fitted parameters have log-uniform search spaces.
                # Perturb in log space so the initialization is symmetric in
                # relative, rather than absolute, parameter change.
                init_rng=np.random.default_rng(args.random_state+session_i*100+attempt*1000+7919)
                center=np.log(np.asarray(x0,float))
                low=np.log(np.asarray([dim.low for dim in space],float))
                high=np.log(np.asarray([dim.high for dim in space],float))
                starts=np.exp(np.clip(center+init_rng.normal(
                    0.,.05*(high-low),size=(4,len(space))),low,high))
                kwargs['x0']=starts.tolist()
            forest_minimize(objective,space,n_calls=n_calls,
                n_initial_points=14 if session_i==0 else 8,
                random_state=args.random_state+session_i*100+attempt*1000,**kwargs)
            best=min(trace,key=lambda r:r['loss']);args.n_tr=N_TR
            r,summary=bw.simulate_eval(sc[session_i],fc[session_i],best['params'],args,seed)
            best['fc_r_fit']=best['fc_r'];best['fc_r']=float(r);best['summary']=summary.tolist()
            return best
        if 'baseline' not in record['results']:
            record['results']['baseline']=fit(0,90,0)
            robust.atomic_json(checkpoint,record)
        baseline=record['results']['baseline']
        accepted=False;reason='baseline_FC'
        if np.isfinite(baseline['fc_r']) and baseline['fc_r']>.15:
            def neural_metrics(session_i,fitrec):
                values=simulate_neural(jnp.asarray(base.prepare_sc(sc[session_i]),jnp.float32),
                    jnp.asarray(fitrec['params'],jnp.float32),jnp.asarray(motor),jax.random.PRNGKey(seed),
                    int(round((config.burn_s+N_TR*config.tr)/config.dt)),
                    int(round(config.burn_s/config.dt)),config.dt,config.fixed_sigma,config.fixed_i0)
                signal=np.asarray(jax.device_get(values[0]),np.float32)
                del values
                with tempfile.TemporaryDirectory(prefix='multirate_metrics_') as temp:
                    state=Path(temp)/'states.npz';metrics=Path(temp)/'metrics.json'
                    np.savez_compressed(state,S_E=signal[None,...])
                    subprocess.run([EEGPY,str(Path(__file__).with_name('neural_metrics.py')),
                                    str(state),str(metrics)],check=True,cwd=REPO)
                    return json.loads(metrics.read_text())[0]

            # The baseline is identical for every follow-up attempt.  Cache its
            # Cache the post-initialisation neural simulation and native-1-kHz SE/DFA.
            if 'screen_metrics' not in baseline:
                baseline['screen_metrics']=neural_metrics(0,baseline)
                robust.atomic_json(checkpoint,record)
            for attempt in range(record['attempt'],8):
                post=fit(1,35,attempt,baseline['params'])
                record['attempt']=attempt+1
                robust.atomic_json(checkpoint,record)
                reason='followup_FC'
                if not np.isfinite(post['fc_r']) or post['fc_r']<=.15:
                    print(f'candidate {candidate} post {attempt}: FC {post["fc_r"]:.4f}',flush=True)
                    continue
                m=[baseline['screen_metrics'],neural_metrics(1,post)]
                delta_se=m[1]['SE']-m[0]['SE'];delta_dfa=m[1]['DFA']-m[0]['DFA']
                print(f'candidate {candidate} post {attempt}: FC {baseline["fc_r"]:.4f}/{post["fc_r"]:.4f}, deltaSE {delta_se:+.5f}, deltaDFA {delta_dfa:+.5f}',flush=True)
                reason='SE_DFA_direction'
                if delta_se>0 and delta_dfa<0 and np.isfinite(delta_se+delta_dfa):
                    number=progress['accepted']+1
                    record['selected_id']=number;record['results']['followup6w']=post
                    for session,metrics_row in zip(('baseline','followup6w'),m):
                        record['results'][session].update(metrics_row)
                    robust.atomic_json(OUTPUT/f'selected_{number:03d}.json',record)
                    progress['accepted']=number;accepted=True
                    print(f'ACCEPT {number}/{a.target_count}: candidate {candidate} seed {seed}',flush=True)
                    break
        if not accepted:
            progress['rejected'].append(dict(candidate_id=candidate,seed=seed,reason=reason,
                baseline_fc=baseline['fc_r']))
            print(f'REJECT candidate {candidate}: {reason}',flush=True)
        progress['next_candidate']=candidate+1
        robust.atomic_json(status,progress)
        checkpoint.unlink(missing_ok=True)
        if progress['accepted']==10:
            print('Reached 10 accepted fits; continuing to target',flush=True)
    print(f"Completed {progress['accepted']} accepted fits",flush=True)

if __name__=='__main__':main()
