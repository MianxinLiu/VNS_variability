"""Isolated 1-ms neural / 10-ms hemodynamic Euler integration.

Ten post-update S_E values are averaged for each hemodynamic update.
Noise, neural equations, BW constants, initial states and FC calculation
match the existing model. No feedback inhibition control is applied.
"""
from functools import partial
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"core"))
import jax
import jax.numpy as jnp
import numpy as np
import dmf_rww_ei_jax_gpu_bw_fc_only as original
import dmf_rww_ei_jax_gpu_fc_only as base

param_space = original.param_space

@partial(jax.jit, static_argnames=('total_steps','burn_steps','steps_per_tr','n_tr','summary_start'))
def simulate(sc, params, key, total_steps, burn_steps, steps_per_tr, n_tr, summary_start):
    n=sc.shape[0]; dt=.001; hemo_dt=.01
    g,wee,wei,wie=params
    ke,ki=jax.random.split(key)
    ne=jax.random.normal(ke,(total_steps,n),dtype=jnp.float32)
    ni=jax.random.normal(ki,(total_steps,n),dtype=jnp.float32)
    def step(carry, xs):
        se,si,s,f,v,q,drive,sums=carry
        t,noise_e,noise_i=xs
        ie=.382+wee*.15*se+g*.15*(sc@se)-wie*si
        ii=.7*.382+wei*.15*se-si
        re=original.h_transfer(ie,310.,125.,.16)
        ri=original.h_transfer(ii,615.,177.,.087)
        se=jnp.clip(se+(-se/.1+(1-se)*.641*re)*dt+.001*jnp.sqrt(dt)*noise_e,0,1)
        si=jnp.clip(si+(-si/.01+ri)*dt+.001*jnp.sqrt(dt)*noise_i,0,1)
        drive=drive+se
        def hemo(state):
            s,f,v,q,drive=state
            fs=jnp.maximum(f,1e-6);vs=jnp.maximum(v,1e-6)
            extraction=fs*(1-(1-.34)**(1/fs))/.34
            ns=s+(drive/10-.65*s-.41*(fs-1))*hemo_dt
            nf=jnp.maximum(f+s*hemo_dt,1e-6)
            nv=jnp.maximum(v+(fs-vs**(1/.32))/.98*hemo_dt,1e-6)
            nq=jnp.maximum(q+(extraction-q/vs**(1-1/.32))/.98*hemo_dt,1e-6)
            return ns,nf,nv,nq,jnp.zeros_like(drive)
        s,f,v,q,drive=jax.lax.cond((t+1)%10==0,hemo,lambda x:x,(s,f,v,q,drive))
        values=jnp.stack((se,si,ie,ii,re,ri))
        sums=sums+jnp.where(t>=summary_start,values,0.)
        bold=.02*(7*(1-q)+2*(1-q/v)+2*(1-v))
        return (se,si,s,f,v,q,drive,sums),bold
    zero=jnp.zeros(n,dtype=jnp.float32);one=jnp.ones(n,dtype=jnp.float32)
    init=(one*.001,one*.001,zero,one,one,one,zero,jnp.zeros((6,n),dtype=jnp.float32))
    final,hist=jax.lax.scan(step,init,(jnp.arange(total_steps),ne,ni))
    means=final[-1]/(total_steps-summary_start)
    summary=jnp.concatenate((means,jnp.stack((means[0]/(means[1]+1e-12),
        means[2]/(means[3]+1e-12),means[4]/(means[5]+1e-12)))))
    return hist[burn_steps::steps_per_tr][:n_tr].T,summary

def simulate_eval(sc, fc_emp, params, args, seed):
    if args.dt != .001 or args.fixed_sigma != .001 or args.bold_input != 'S_E':
        raise ValueError('This isolated pilot fixes dt=.001, sigma=.001, BOLD input=S_E')
    stride=int(round(args.tr/.001));burn=int(round(args.burn_s/.001))
    bold,summary=simulate(jnp.asarray(base.prepare_sc(sc),jnp.float32),
        jnp.asarray(params,jnp.float32),jax.random.PRNGKey(seed),burn+args.n_tr*stride,
        burn,stride,args.n_tr,int(round(args.summary_start_s/.001)))
    ts=np.asarray(jax.device_get(bold))
    return float(base.corr_upper(fc_emp,base.fc_from_ts(ts))),np.asarray(jax.device_get(summary))
