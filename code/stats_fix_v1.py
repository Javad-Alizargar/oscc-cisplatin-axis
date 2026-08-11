"""Correct malignant-atlas inference to the patient/sample level."""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

SEED=20260812
rng=np.random.default_rng(SEED)
BASE=Path(os.environ.get('MALIGNANT_RESULTS_DIR',Path(__file__).resolve().parents[1]/'step5_malignant_v1/results/malignant_out_v1'))
OUT=Path(__file__).resolve().parent/'results_v1'
FIG=OUT/'figures'; TAB=OUT/'tables'; META=OUT/'metadata'
for p in (FIG,TAB,META): p.mkdir(parents=True,exist_ok=True)

pb=pd.read_csv(BASE/'tables/sample_program_scores_v1.tsv',sep='\t')
comp=pd.read_csv(BASE/'tables/state_composition_v1.tsv',sep='\t')
coords=pd.read_csv(BASE/'tables/malignant_coordinates_v1.tsv.gz',sep='\t')
programs=[c for c in pb if c not in ['dataset','sample_id','axis_z']]
states=[c for c in comp if c not in ['dataset','sample_id']]

def ranks(x): return stats.rankdata(np.asarray(x,float))
def corr(x,y):
 x=np.asarray(x,float); y=np.asarray(y,float)
 return np.corrcoef(x,y)[0,1] if np.std(x)>0 and np.std(y)>0 else np.nan
def stratified_vectors(frame,xcol,ycol):
 xs=[]; ys=[]; blocks=[]
 for _,g in frame.groupby('dataset',sort=True):
  x=ranks(g[xcol]); y=ranks(g[ycol]); xs.extend(x-x.mean()); ys.extend(y-y.mean()); blocks.append(len(g))
 return np.asarray(xs),np.asarray(ys),blocks
def stratified_test(frame,xcol,ycol,B=50000):
 x,y,blocks=stratified_vectors(frame,xcol,ycol); est=corr(x,y); null=np.empty(B)
 for b in range(B):
  yp=[]; k=0
  for n in blocks: yp.extend(rng.permutation(y[k:k+n])); k+=n
  null[b]=corr(x,np.asarray(yp))
 p=(1+np.sum(np.abs(null)>=abs(est)))/(B+1)
 boots=[]
 groups=[g for _,g in frame.groupby('dataset',sort=True)]
 for _ in range(10000):
  z=[]
  for g in groups: z.append(g.iloc[rng.integers(0,len(g),len(g))])
  z=pd.concat(z,ignore_index=True); bx,by,_=stratified_vectors(z,xcol,ycol); r=corr(bx,by)
  if np.isfinite(r): boots.append(r)
 lo,hi=np.quantile(boots,[.025,.975])
 return est,lo,hi,p
def bh(p):
 p=np.asarray(p,float); order=np.argsort(p); q=np.empty(len(p)); running=1.
 for rank,idx in reversed(list(enumerate(order,1))): running=min(running,p[idx]*len(p)/rank); q[idx]=running
 return q

# Program inference: pooled descriptive, each cohort, and cohort-stratified test.
pa=[]
for name in programs:
 row={'program':name,'pooled_rho':stats.spearmanr(pb[name],pb.axis_z).statistic}
 signs=[]
 for ds,g in pb.groupby('dataset'):
  r=stats.spearmanr(g[name],g.axis_z).statistic; row[f'{ds}_rho']=r; signs.append(np.sign(r))
 est,lo,hi,p=stratified_test(pb,name,'axis_z'); row.update(stratified_rho=est,ci_low=lo,ci_high=hi,p_perm=p,direction_concordant=len(set(signs))==1)
 pa.append(row)
pa=pd.DataFrame(pa); pa['p_bh']=bh(pa.p_perm); pa=pa.sort_values('stratified_rho')
pa.to_csv(TAB/'program_stratified_inference_v1.tsv',sep='\t',index=False)

# State abundance inference. One row per sample; no cell-level confidence intervals.
sc=comp.merge(pb[['dataset','sample_id','axis_z']],on=['dataset','sample_id'],validate='one_to_one')
sa=[]
for st in states:
 row={'state':st,'n_cells':int((coords.state.astype(str)==str(st)).sum()),'n_samples_present':int((sc[st]>0).sum())}
 signs=[]
 for ds,g in sc.groupby('dataset'):
  r=stats.spearmanr(g[st],g.axis_z).statistic; row[f'{ds}_rho']=r; signs.append(np.sign(r))
 est,lo,hi,p=stratified_test(sc,st,'axis_z'); row.update(stratified_rho=est,ci_low=lo,ci_high=hi,p_perm=p,direction_concordant=len(set(signs))==1)
 sa.append(row)
sa=pd.DataFrame(sa); sa['p_bh']=bh(sa.p_perm); sa=sa.sort_values('stratified_rho')
sa.to_csv(TAB/'state_abundance_stratified_inference_v1.tsv',sep='\t',index=False)

# Corrected six-panel figure with descriptive cell panels and inferential sample panels clearly separated.
sns.set_theme(style='white',font_scale=.85)
def panel(ax,l,t): ax.text(-.12,1.07,l,transform=ax.transAxes,fontweight='bold',fontsize=14,va='top'); ax.set_title(t,loc='left',fontweight='bold',fontsize=10)
fig,axs=plt.subplots(2,3,figsize=(14.2,9.2),constrained_layout=True)
ax=axs[0,0];panel(ax,'a','Malignant-state manifold (descriptive)');sns.scatterplot(data=coords,x='UMAP1',y='UMAP2',hue=coords.state.astype(str),s=8,linewidth=0,palette='tab10',ax=ax);ax.legend(title='State',ncol=2,fontsize=7,frameon=False)
ax=axs[0,1];panel(ax,'b','Resistance-axis localization (descriptive)');q=ax.scatter(coords.UMAP1,coords.UMAP2,c=coords.axis_z,s=8,cmap='coolwarm',vmin=-2,vmax=2,linewidth=0);fig.colorbar(q,ax=ax,label='Resistance-axis z');ax.set(xlabel='UMAP1',ylabel='UMAP2')
ax=axs[0,2];panel(ax,'c','Patient-level malignant-state composition');comp.set_index(['dataset','sample_id'])[states].plot.bar(stacked=True,colormap='tab20',width=.86,ax=ax);ax.set(ylabel='Cell fraction',xlabel='Patient/sample');ax.tick_params(axis='x',rotation=70,labelsize=6);ax.legend(title='State',ncol=2,fontsize=6,frameon=False)
ax=axs[1,0];panel(ax,'d','Cohort-stratified program associations');z=pa.sort_values('stratified_rho');y=np.arange(len(z));ax.errorbar(z.stratified_rho,y,xerr=np.vstack([z.stratified_rho-z.ci_low,z.ci_high-z.stratified_rho]),fmt='o',color='#0072B2',ecolor='#777777',capsize=3);ax.axvline(0,c='k',ls='--',lw=.8);ax.set_yticks(y,z.program,fontsize=7);ax.set(xlabel='Stratified Spearman rho (95% bootstrap CI)',ylabel='');
for i,r in enumerate(z.itertuples()):
 if r.p_bh<.05: ax.text(.98,i,'BH q<0.05',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=6,color='#D55E00')
ax=axs[1,1];panel(ax,'e','Cross-dataset direction audit');hm=pa.set_index('program')[['GSE172577_rho','GSE215403_rho','stratified_rho']].rename(columns=lambda x:x.replace('_rho',''));sns.heatmap(hm,cmap='vlag',center=0,vmin=-1,vmax=1,annot=True,fmt='.2f',annot_kws={'fontsize':6},ax=ax,cbar_kws={'label':'Spearman rho'});ax.set(xlabel='',ylabel='')
ax=axs[1,2];panel(ax,'f','State abundance versus resistance');z=sa.sort_values('stratified_rho');y=np.arange(len(z));ax.errorbar(z.stratified_rho,y,xerr=np.vstack([z.stratified_rho-z.ci_low,z.ci_high-z.stratified_rho]),fmt='o',color='#009E73',ecolor='#777777',capsize=3);ax.axvline(0,c='k',ls='--',lw=.8);ax.set_yticks(y,[f"State {r.state} (n={r.n_cells})" for r in z.itertuples()],fontsize=7);ax.set(xlabel='Stratified Spearman rho (95% bootstrap CI)',ylabel='')
fig.suptitle('Patient-level validation of malignant states and cisplatin-resistance programs',fontweight='bold',fontsize=15)
fig.savefig(FIG/'Figure_malignant_v2.png',dpi=300,bbox_inches='tight',facecolor='white');fig.savefig(FIG/'Figure_malignant_v2.pdf',bbox_inches='tight',facecolor='white');fig.savefig(FIG/'Figure_malignant_v2.tif',dpi=600,bbox_inches='tight',facecolor='white',pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)

summary={'n_samples':int(len(pb)),'n_cells':int(len(coords)),'programs_bh_lt_0_05':pa.loc[pa.p_bh<.05,'program'].tolist(),'programs_direction_concordant':pa.loc[pa.direction_concordant,'program'].tolist(),'state_tests_bh_lt_0_05':sa.loc[sa.p_bh<.05,'state'].astype(str).tolist(),'interpretation':'Cell-level panels are descriptive. All association estimates, confidence intervals, permutation P values, and multiplicity correction use the patient/sample as the inferential unit and stratify by dataset.'}
(META/'summary_v1.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
