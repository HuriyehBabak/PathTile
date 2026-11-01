"""Rebuild audited summaries and figure. Input: extracted job results directory."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
a=parser.parse_args(); out=a.output; (out/'figs').mkdir(parents=True,exist_ok=True)
stages=['transfer_gpu_ms','model_fwd_gpu_ms','loss_gpu_ms','backward_gpu_ms','clip_grad_gpu_ms','optimizer_step_gpu_ms']
rows=[]; manifests={}; checks=[]
for trial in (1,2,3):
 for backend in ('pytorch','warp'):
  p=a.root/f'trial{trial}'/backend
  m=json.loads((p/'training_manifest.json').read_text()); manifests[trial,backend]=m
  e=pd.read_csv(p/'training_epoch_system_summary.csv'); s=pd.read_csv(p/'training_batch_stage_times.csv'); t=pd.read_csv(p/'training_final_test_metrics.csv'); h=pd.read_csv(p/'training_history.csv'); c=pd.read_csv(p/'correctness.csv')
  assert m['protocol']=='temporal_logmse_trainmedian_v2'
  assert m['config']['seed']==trial-1 and m['backend']==backend
  assert m['config']['epochs']==5 and m['config']['batch_size']==16384 and m['train_batches_per_epoch']==68
  assert len(t)==1 and t.iloc[0].train_backend==backend and t.iloc[0].seed==trial-1
  assert e.epoch.tolist()==[2,3,4,5] and h.epoch.tolist()==[1,2,3,4,5]
  assert s.groupby('epoch').size().to_dict()==dict.fromkeys([2,3,4,5],68)
  assert not s.duplicated(['epoch','batch_id']).any()
  assert all(sorted(g.batch_id)==list(range(1,69)) for _,g in s.groupby('epoch'))
  assert set(x for x in s if x.endswith('_gpu_ms'))==set(stages)
  assert np.isfinite(s[stages].to_numpy()).all() and (s[stages]>=0).all().all()
  assert np.isfinite(h[['train_log_mse','val_global_rmsle']].to_numpy()).all()
  assert (e.train_wall_time_s>0).all() and np.isfinite(e.train_wall_time_s).all()
  assert c[['fwd_allclose','bwd_input_allclose','bwd_weight_allclose']].eq(True).all().all() and len(c)==5
  for name,digest in m['source_sha256'].items():
   assert hashlib.sha256((p/'source_snapshot'/name).read_bytes()).hexdigest()==digest
  r=dict(trial=trial,seed=trial-1,backend=backend,epoch_wall_s=e.train_wall_time_s.mean(),gpu_stages_ms=s[stages].sum(axis=1).mean(),forward_ms=s.model_fwd_gpu_ms.mean(),backward_ms=s.backward_gpu_ms.mean(),other_gpu_ms=s[[x for x in stages if x not in ['model_fwd_gpu_ms','backward_gpu_ms']]].sum(axis=1).mean(),throughput_samples_s=68*16384/e.train_wall_time_s.mean(),peak_memory_gib=e.peak_memory_gib.max(),util_gpu_pct=e.util_gpu_avg_pct.mean(),test_rmsle=t.iloc[0].test_rmsle,val_rmsle=h.iloc[-1].val_global_rmsle,zero_rmsle=t.iloc[0].zero_baseline_rmsle,persistence_rmsle=t.iloc[0].persistence_baseline_rmsle)
  assert r['test_rmsle']<r['persistence_rmsle'] and t.iloc[0].test_samples==487539
  rows.append(r)
 for key in ('config','initial_model_sha256','source_sha256','splits','scaler','launch'):
  assert manifests[trial,'warp'][key]==manifests[trial,'pytorch'][key],key
for m in manifests.values():
 for key in ('splits','scaler','source_sha256','launch'):
  assert m[key]==manifests[1,'warp'][key],key
assert len({manifests[t,'warp']['initial_model_sha256'] for t in (1,2,3)})==3
f=pd.DataFrame(rows); f.to_csv(out/'per_trial.csv',index=False)
metrics=['epoch_wall_s','gpu_stages_ms','forward_ms','backward_ms','other_gpu_ms','throughput_samples_s','peak_memory_gib','util_gpu_pct','test_rmsle','val_rmsle']
agg=f.groupby('backend')[metrics].agg(['mean','std']); agg.columns=['_'.join(c) for c in agg.columns]; agg.to_csv(out/'backend_summary.csv')
paired=[]
for trial in (1,2,3):
 g=f[f.trial==trial].set_index('backend'); p,w=g.loc['pytorch'],g.loc['warp']
 r={'trial':trial,'seed':trial-1}
 for key in ('epoch_wall_s','gpu_stages_ms','forward_ms','backward_ms'):
  r[key+'_speedup']=p[key]/w[key]
 r['rmsle_abs_difference']=abs(p.test_rmsle-w.test_rmsle)
 r['epoch_reduction_pct']=100*(1-w.epoch_wall_s/p.epoch_wall_s)
 paired.append(r)
paired=pd.DataFrame(paired); paired.to_csv(out/'paired_results.csv',index=False)
summary={k+'_geomean':float(np.exp(np.log(paired[k]).mean())) for k in paired if k.endswith('_speedup')}
summary['epoch_reduction_from_geomean_pct']=100*(1-1/summary['epoch_wall_s_speedup_geomean'])
summary['max_rmsle_abs_difference']=float(paired.rmsle_abs_difference.max())
(out/'paper_summary.json').write_text(json.dumps(summary,indent=2))
(out/'audit.txt').write_text('PASS: six runs; seeds 0/1/2; paired initial model/configuration; same source hashes and verified snapshots; identical split hashes/scalers/launches across six runs; 68 unique batches in each epoch 2-5; finite nonnegative stage timings; all 30 recorded correctness cases pass; finite training histories; all six test RMSLE scores below persistence.\nPrediction arrays/checkpoints and raw dataset files were excluded from the submitted archive, so reported RMSLE was read from metrics CSVs, not recomputed from predictions. Data-file hashes were compared across manifests, not against raw data bytes.\n')
print(f.to_string(index=False)); print(agg.to_string()); print(json.dumps(summary,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(10,3.65),gridspec_kw={'width_ratios':[1,1.15,1.25]})
colors=['#737F8C','#007D8A']; labels=['PyTorch','Warp']; backends=['pytorch','warp']
ax=axes[0]; means=[agg.loc[b,'epoch_wall_s_mean'] for b in backends]; sd=[agg.loc[b,'epoch_wall_s_std'] for b in backends]
ax.bar(labels,means,color=colors,width=.57,yerr=sd,capsize=4)
for i,v in enumerate(means): ax.text(i,v+.65,f'{v:.2f}',ha='center')
ax.set_ylim(0,max(means)*1.22); ax.set_ylabel('Seconds / training epoch'); ax.set_title('(a) Epoch wall time',loc='left',fontweight='bold')
ax=axes[1]; bottom=np.zeros(2)
for col,label,color in [('forward_ms','Model forward','#327AAB'),('backward_ms','Aggregate backward','#E6A23D'),('other_gpu_ms','Other GPU stages','#A8AFB8')]:
 v=np.array([agg.loc[b,col+'_mean'] for b in backends]); ax.bar(labels,v,bottom=bottom,label=label,color=color,width=.57); bottom+=v
ax.errorbar([0,1],bottom,yerr=[agg.loc[b,'gpu_stages_ms_std'] for b in backends],fmt='none',color='black',capsize=4)
for i,v in enumerate(bottom): ax.text(i,v+8,f'{v:.1f}',ha='center')
ax.set_ylim(0,max(bottom)*1.22); ax.set_ylabel('Milliseconds / batch'); ax.set_title('(b) Recorded GPU stages',loc='left',fontweight='bold')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),frameon=False,fontsize=8,ncol=1)
ax=axes[2]; keys=['epoch_wall_s','gpu_stages_ms','forward_ms','backward_ms']; v=[summary[k+'_speedup_geomean'] for k in keys]
ax.bar(['Epoch','GPU\nstages','Forward','Backward'],v,color='#007D8A',width=.65)
for i,x in enumerate(v): ax.text(i,x+.05,f'{x:.2f}×',ha='center',fontsize=9)
ax.axhline(1,color='#555555',ls='--',lw=.8); ax.set_ylim(0,max(v)*1.22); ax.set_ylabel('Paired speedup over PyTorch'); ax.set_title('(c) Geometric-mean speedups',loc='left',fontweight='bold'); ax.tick_params(axis='x',labelsize=8)
for ax in axes: ax.set_axisbelow(True); ax.yaxis.grid(True,alpha=.2)
fig.subplots_adjust(left=.065,right=.99,top=.88,bottom=.29,wspace=.48)
fig.savefig(out/'figs/fig_end_to_end_a6000.pdf',bbox_inches='tight')
fig.savefig(out/'figs/fig_end_to_end_a6000.png',dpi=200,bbox_inches='tight')
