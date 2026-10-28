"""Reproduce the manuscript main model figure from the 50-fit tables."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import wilcoxon,spearmanr
import plot_model_helpers as base
HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'data'/'model'
OUT=HERE.parent/'figures'

def signed(values,alternative='two-sided'):
    values=np.asarray(values)
    return 1. if np.all(values==0) else wilcoxon(values,alternative=alternative).pvalue

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();data_dir=args.data_dir
    metrics=pd.read_csv(data_dir/'all_seed_metrics.csv')
    mse=pd.read_csv(data_dir/'mse_sessions.csv')
    metrics=metrics.merge(mse[['candidate_id','seed','session','MSE']],
                          on=['candidate_id','seed','session'],how='left',validate='one_to_one')
    if metrics.MSE.isna().any():
        raise ValueError('Missing matched-seed MSE results for the requested fits.')
    currents=pd.read_csv(data_dir/'current_terms.csv');params=pd.read_csv(data_dir/'parameters.csv')
    stability=pd.read_csv(data_dir/'stability.csv')
    if metrics.candidate_id.nunique()!=50 or 66 in set(metrics.candidate_id):
        raise ValueError('Expected the 50-fit main ensemble with candidate 66 excluded')
    base.CANDIDATES=tuple(sorted(metrics.candidate_id.unique()))
    plt.rcParams.update({'font.size':10,'axes.titlesize':13,'axes.labelsize':13,
                         'xtick.labelsize':11,'ytick.labelsize':11})
    fig=plt.figure(figsize=(19,14.8),facecolor='white')
    g=fig.add_gridspec(3,12,left=.075,right=.975,top=.945,bottom=.07,
        height_ratios=[1,1.15,1.15],hspace=.65,wspace=1.65)
    ax=fig.add_subplot(g[0,:3]);base.all_seed_bar(ax,metrics,'fc_r','FC similarity','FC r',zoom=True)
    fc_values=metrics['fc_r'].to_numpy(float)
    fc_values=fc_values[np.isfinite(fc_values)]
    if fc_values.size:
        fc_span=max(float(np.ptp(fc_values)),.01)
        ax.set_ylim(float(fc_values.min()-.08*fc_span),
                    float(fc_values.max()+.08*fc_span))
    ax.set_title('FC similarity');base.label(ax,'A')
    bg=g[0,3:].subgridspec(1,3,wspace=.48)
    for i,m in enumerate(('SE','DFA','MSE')):
        ax=fig.add_subplot(bg[0,i]);pair=base.paired_ladder(ax,metrics,m,'Spectral exponent' if m=='SE' else m,m)
        if m=='MSE':
            mse_p=signed(pair.followup6w-pair.baseline)
            ax.set_title(f'MSE\nmean Δ={(pair.followup6w-pair.baseline).mean():+.3f}\np={mse_p:.2g}')
            mse_ax=ax
        if i==0:base.label(ax,'B')
    ax=fig.add_subplot(g[1,:6]);parameter_ax=ax;order=['G','w_EE','w_EI','w_IE']
    groups=[params.loc[params.parameter==m,'delta'].to_numpy() for m in order]
    ps=[signed(x) for x in groups]
    base.horizontal_delta_boxes(ax,groups,[r'$G$',r'$w_{EE}$',r'$w_{EI}$',r'$w_{IE}$'],['#4E79A7','#59A14F','#8F6BB1','#6B7280'],
        'Fitted parameter changes','6 weeks − baseline',[base.sigstar(p) for p in ps],ps,
        annotation_x=.98,annotation_yoffset=.43)
    ax.set_ylim(-.5,3.8);base.label(ax,'C')
    ax.set_xlim(max(-1.8,ax.get_xlim()[0]),ax.get_xlim()[1])
    association_row=g[2,:].subgridspec(1,4,width_ratios=[1.85,1,1,1],wspace=.53)
    ax=fig.add_subplot(g[1,6:]);specs=[('local_E',r'$w_{EE}J_{\mathrm{NMDA}} S^E$'),('network_E',r'$GJ_{\mathrm{NMDA}}(C\cdot S^E)$'),
        ('inhibition_magnitude',r'$w_{IE}S^I$'),('simulated_I_E',r'Net $I^E$')]
    pairs=currents.pivot(index='candidate_id',columns='session',values=[s[0] for s in specs])
    groups=[(pairs[(s[0],'followup6w')]-pairs[(s[0],'baseline')]).to_numpy() for s in specs];ps=[signed(x) for x in groups]
    base.horizontal_delta_boxes(ax,groups,[s[1] for s in specs],['#4E79A7','#59A14F','#8F6BB1','#6B7280'],
        r'Motor $I^E$ current terms','6 weeks − baseline',[base.sigstar(p) for p in ps],ps,
        annotation_yoffset=.43)
    ax.set_ylim(-.5,3.8);base.label(ax,'D')

    ax=fig.add_subplot(association_row[0,0])
    stability_ax=ax
    stability_values=[]
    stability_annotations=[]
    for i,metric in enumerate(('stability_margin','local_motor_stability_margin')):
        pair=stability.pivot(index='candidate_id',columns='session',values=metric)
        pair=pair[['baseline','followup6w']].dropna()
        positions=np.array([0.,.5])+1.5*i
        stability_values.extend(pair.to_numpy().ravel())
        jitter=np.random.default_rng(531+i).uniform(-.055,.055,len(pair))
        for offset,values in zip(jitter,pair.to_numpy()):
            ax.plot(positions+offset,values,'o-',color=base.LIGHT,
                    alpha=.8,lw=.85,ms=2.5,zorder=1)
        boxes=ax.boxplot([pair.baseline,pair.followup6w],positions=positions,
            widths=.30,patch_artist=True,showfliers=False,
            medianprops=dict(color=base.DARK,lw=1.7),
            whiskerprops=dict(color=base.DARK,lw=1.1),
            capprops=dict(color=base.DARK,lw=1.1))
        for box,color in zip(boxes['boxes'],[base.BASE,base.POST]):
            box.set(facecolor=color,edgecolor=color,alpha=.48,zorder=2)
        delta=pair.followup6w-pair.baseline
        pv=signed(delta,'two-sided')
        p_label=f'p={pv:.3f}' if pv>=.001 else 'p<0.001'
        stability_annotations.append((positions.mean(),f'Δ={delta.mean():+.2f}{base.sigstar(pv)}\n{p_label}'))
    low,high=min(stability_values),max(stability_values)
    span=high-low
    for x,label in stability_annotations:
        ax.text(x,high+.07*span,label,ha='center',va='bottom',fontsize=12)
    ax.set(xlim=(-.35,2.35),ylim=(low-.09*span,high+.48*span),
           xticks=[.25,1.75],xticklabels=['Whole cortex','Motor local'],
           ylabel='Stability',title='Stability')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=base.BASE,alpha=.48,label='Baseline'),
                       Patch(facecolor=base.POST,alpha=.48,label='6 weeks')],
              loc='upper center',ncol=2,frameon=False,fontsize=13,
              handlelength=1,columnspacing=.9)
    ax.spines[['top','right']].set_visible(False)
    base.label(ax,'E')

    metric_pair=metrics.pivot(index='candidate_id',columns='session',values=['SE','DFA','MSE'])
    margin_pair=stability.pivot(index='candidate_id',columns='session',
                                values='local_motor_stability_margin')
    delta_margin=margin_pair.followup6w-margin_pair.baseline
    validation=[
        (metric_pair[('SE','followup6w')]-metric_pair[('SE','baseline')],
         'Δ motor local stability','Δ spectral exponent',
         'Stability change vs. SE','F'),
        (metric_pair[('DFA','followup6w')]-metric_pair[('DFA','baseline')],
         'Δ motor local stability','Δ DFA',
         'Stability change vs. DFA',None),
        (metric_pair[('MSE','followup6w')]-metric_pair[('MSE','baseline')],
         'Δ motor local stability','Δ MSE',
         'Stability change vs. MSE',None)]
    association_stats=[]
    for i,(outcome,xlabel,ylabel,title,panel) in enumerate(validation):
        ax=fig.add_subplot(association_row[0,i+1])
        paired=pd.concat([delta_margin.rename('stability'),outcome.rename('metric')],axis=1).dropna()
        x,y=paired.stability,paired.metric
        rho,pvalue=spearmanr(x,y)
        association_stats.append(dict(metric=['SE','DFA','MSE'][i],n=len(paired),
            spearman_rho=rho,p_two_sided=pvalue))
        ax.scatter(x,y,s=42,facecolor='white',edgecolor='#4E79A7',linewidth=1.3)
        if np.ptp(x)>0:
            coef=np.polyfit(x,y,1)
            line=np.linspace(x.min(),x.max(),100)
            ax.plot(line,np.polyval(coef,line),color=base.DARK,linewidth=1.7)
        ax.axhline(0,color='#A8A8A8',linewidth=.8);ax.axvline(0,color='#A8A8A8',linewidth=.8)
        ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
        p_label=f'p={pvalue:.3f}' if pvalue>=.001 else 'p<0.001'
        ax.text(.5,1.025,f'Spearman ρ={rho:.2f}, {p_label}',transform=ax.transAxes,
                ha='center',va='bottom',fontsize=12,fontweight='bold' if pvalue<.05 else 'normal')
        ax.spines[['top','right']].set_visible(False)
        if panel:base.label(ax,panel)

    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(15);ax.yaxis.label.set_fontsize(15)
        ax.xaxis.labelpad=9;ax.yaxis.labelpad=9
        ax.tick_params(labelsize=13,length=4,pad=5)
        if len(ax.get_xticks())>6:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(ax.get_title(),fontsize=15,pad=34)
        ax.grid(False)
        for text in ax.texts:
            if text.get_text() in list('ABCDEFGHIJKLMN'):
                text.set_fontsize(19)
                text.set_position((-.12,1.15))
            else:
                text.set_fontsize(12)
    stability_ax.tick_params(labelsize=15)
    # Anchor the first title line at the same figure height for A and B.
    for top_ax in fig.axes[:4]:
        top_ax.set_title(top_ax.get_title(), fontsize=15, y=1.20, pad=0,
                         verticalalignment='top', multialignment='center')
    stability_ax.yaxis.label.set_fontsize(17)
    stability_ax.title.set_fontsize(17)
    for text in stability_ax.texts:
        if text.get_text()!='E':
            text.set_fontsize(14)
    parameter_pos=parameter_ax.get_position()
    parameter_ax.set_position([parameter_pos.x0,parameter_pos.y0,
                               parameter_pos.width-.025,parameter_pos.height])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    stem='figure5_model_results'
    pd.DataFrame(association_stats).to_csv(
        args.output_dir/f'{stem}_stability_metric_correlations.csv',index=False)
    fig.savefig(args.output_dir/f'{stem}.png',dpi=240)
    fig.savefig(args.output_dir/f'{stem}.pdf')
    plt.close(fig)
    print(args.output_dir/f'{stem}.png',flush=True)


if __name__=="__main__": main()
