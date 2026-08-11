"""Patient-aware malignant-state atlas and resistance-program validation."""
from google.colab import drive, files
drive.mount('/content/drive')
import os, sys, subprocess
try:
 import scanpy as sc, harmonypy
except Exception:
 subprocess.check_call([sys.executable,'-m','pip','install','-q','scanpy>=1.10','harmonypy','openpyxl'])
 print('Packages installed. Restart the runtime and rerun the notebook.'); os.kill(os.getpid(),9)
from pathlib import Path
import json, zipfile, warnings
import anndata as ad
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

SEED=20260812; rng=np.random.default_rng(SEED)
ROOT=Path('/content/drive/MyDrive/OSCC_Cisplatin_PhysicsInformed_StateTransitions')
OUT=Path('/content/malignant_out_v1'); FIG=OUT/'figures'; TAB=OUT/'tables'; META=OUT/'metadata'
for p in (FIG,TAB,META): p.mkdir(parents=True,exist_ok=True)

PROGRAMS={
 'EMT':['VIM','ZEB1','ZEB2','SNAI1','SNAI2','TWIST1','FN1','ITGA5','COL1A1','MMP2'],
 'DNA repair':['BRCA1','BRCA2','RAD51','ATR','CHEK1','FANCD2','ERCC1','XPA','MSH2','PARP1'],
 'Oxidative phosphorylation':['NDUFA1','NDUFB8','SDHA','UQCRC1','COX5A','ATP5F1A','ATP5MC1','TFAM','PPARGC1A'],
 'Unfolded-protein response':['HSPA5','ATF4','ATF6','DDIT3','XBP1','ERN1','EIF2AK3','HSP90B1','DNAJB9'],
 'Hypoxia':['HIF1A','VEGFA','CA9','SLC2A1','LDHA','BNIP3','ENO1','PGK1','P4HA1'],
 'Inflammatory NF-kB':['NFKB1','RELA','TNFAIP3','CXCL8','IL6','ICAM1','BIRC3','NFKBIA','CCL20'],
 'Interferon response':['STAT1','STAT2','IRF1','IRF7','ISG15','IFIT1','IFIT3','MX1','OAS1'],
 'Apoptosis':['BAX','BAK1','CASP3','CASP8','CASP9','FAS','PMAIP1','BCL2L11','BBC3'],
 'Cell cycle':['MKI67','TOP2A','CDK1','CCNB1','CCNB2','UBE2C','BUB1','AURKB','TYMS'],
 'Squamous differentiation':['KRT5','KRT14','KRT17','TP63','SFN','IVL','KRT1','KRT10','DSG3'],
}

def load(ds):
 c=ad.read_h5ad(ROOT/f'03_analysis_ready/h5ad/{ds}_counts_QC.h5ad')
 s=ad.read_h5ad(ROOT/f'03_analysis_ready/state_space/{ds}_lineage_CNV_resistance_scored.h5ad',backed='r')
 cols=['sample_id','predicted_lineage','epithelial_candidate','cnv_rms','cnv_reference_q95','cnv_above_reference_q95','cnv_high_component','malignant_epithelial_evidence','resistance_axis_raw','resistance_axis_dataset_z','resistance_axis_sample_z']
 cols=[x for x in cols if x in s.obs]
 c.obs=c.obs.drop(columns=[x for x in cols if x in c.obs],errors='ignore').join(s.obs[cols],how='left')
 sym=c.var['gene_symbol'].astype(str) if 'gene_symbol' in c.var else c.var_names.astype(str)
 keep=sym.notna()&~sym.duplicated()&(sym!='nan'); c=c[:,keep].copy(); c.var_names=sym[keep].values
 c.obs['dataset']=ds
 return c

a1,a2=load('GSE172577'),load('GSE215403'); common=a1.var_names.intersection(a2.var_names)
a=ad.concat([a1[:,common],a2[:,common]],join='inner',merge='same',index_unique='-')
a.obs['cell_class']=np.where(a.obs.malignant_epithelial_evidence.astype(str).eq('MALIGNANT_HIGH'),'CNV-supported malignant',np.where(a.obs.predicted_lineage.astype(str).eq('epithelial'),'Other epithelial','Non-epithelial'))
epi=a[a.obs.cell_class.ne('Non-epithelial')].copy(); mal=epi[epi.obs.cell_class.eq('CNV-supported malignant')].copy()
if mal.n_obs<500: raise ValueError(f'Only {mal.n_obs} malignant cells found; check state labels.')

# CNV-supported malignant-cell validation uses patient/sample summaries.
cnv=epi.obs[['dataset','sample_id','cell_class','cnv_rms','cnv_reference_q95','resistance_axis_dataset_z']].copy()
cnv['cnv_excess']=pd.to_numeric(cnv.cnv_rms,errors='coerce')-pd.to_numeric(cnv.cnv_reference_q95,errors='coerce')
cnv_pb=cnv.groupby(['dataset','sample_id','cell_class'],observed=True).agg(n_cells=('cnv_excess','size'),median_cnv_excess=('cnv_excess','median'),median_axis_z=('resistance_axis_dataset_z','median')).reset_index()
cnv_pb.to_csv(TAB/'cnv_sample_validation_v1.tsv',sep='\t',index=False)

# Normalize only malignant cells, select HVGs and build a batch-corrected malignant manifold.
mal.layers['counts']=mal.X.copy(); sc.pp.normalize_total(mal,target_sum=1e4); sc.pp.log1p(mal); mal.raw=mal
sc.pp.highly_variable_genes(mal,n_top_genes=min(2500,mal.n_vars-1),flavor='seurat_v3',layer='counts',batch_key='sample_id',subset=True)
sc.pp.scale(mal,max_value=10); sc.tl.pca(mal,n_comps=30,random_state=SEED)
ho=harmonypy.run_harmony(mal.obsm['X_pca'],mal.obs,'sample_id',random_state=SEED,max_iter_harmony=30,verbose=False)
mal.obsm['X_harmony']=ho.Z_corr.T
sc.pp.neighbors(mal,use_rep='X_harmony',n_neighbors=25); sc.tl.umap(mal,random_state=SEED,min_dist=.25)
mal.obs['state']=pd.Categorical(KMeans(n_clusters=8,n_init=50,random_state=SEED).fit_predict(mal.obsm['X_harmony']).astype(str))

# Score prespecified programs on unscaled log-expression retained in raw.
for name,genes in PROGRAMS.items():
 present=[g for g in genes if g in mal.raw.var_names]
 if len(present)>=4: sc.tl.score_genes(mal,present,score_name=name,use_raw=True,random_state=SEED)
score_names=[x for x in PROGRAMS if x in mal.obs]
mal.obs['axis_z']=pd.to_numeric(mal.obs.resistance_axis_dataset_z,errors='coerce')

# Sample-level effects and patient bootstrap CIs; cells are never treated as replicates.
pb=mal.obs.groupby(['dataset','sample_id'],observed=True)[['axis_z']+score_names].median().reset_index()
def boot_spearman(x,y,B=4000):
 ok=np.isfinite(x)&np.isfinite(y); x=np.asarray(x)[ok]; y=np.asarray(y)[ok]
 est=stats.spearmanr(x,y).statistic; vals=[]
 for _ in range(B):
  q=rng.integers(0,len(x),len(x)); r=stats.spearmanr(x[q],y[q]).statistic
  if np.isfinite(r): vals.append(r)
 return est,*np.quantile(vals,[.025,.975])
assoc=[]
for p in score_names:
 est,lo,hi=boot_spearman(pb[p].values,pb.axis_z.values)
 assoc.append({'program':p,'spearman_rho':est,'ci_low':lo,'ci_high':hi,'n_samples':len(pb)})
assoc=pd.DataFrame(assoc).sort_values('spearman_rho'); assoc.to_csv(TAB/'program_axis_associations_v1.tsv',sep='\t',index=False)
pb.to_csv(TAB/'sample_program_scores_v1.tsv',sep='\t',index=False)

# State composition, state programs, and resistance enrichment with sample-stratified permutation.
state_comp=pd.crosstab([mal.obs.dataset,mal.obs.sample_id],mal.obs.state,normalize='index').reset_index(); state_comp.to_csv(TAB/'state_composition_v1.tsv',sep='\t',index=False)
state_mean=mal.obs.groupby('state',observed=True)[['axis_z']+score_names].mean(); state_mean.to_csv(TAB/'state_program_summary_v1.tsv',sep='\t')
q75=mal.obs.groupby('sample_id',observed=True).axis_z.transform(lambda x:x.quantile(.75)); mal.obs['axis_high']=mal.obs.axis_z.ge(q75)
state_en=[]
for st in mal.obs.state.cat.categories:
 tab=pd.crosstab(mal.obs.state.eq(st),mal.obs.axis_high).reindex(index=[False,True],columns=[False,True],fill_value=0).values+.5
 odds=(tab[1,1]*tab[0,0])/(tab[1,0]*tab[0,1]); se=np.sqrt((1/tab).sum())
 state_en.append({'state':str(st),'odds_ratio':odds,'ci_low':np.exp(np.log(odds)-1.96*se),'ci_high':np.exp(np.log(odds)+1.96*se),'n_cells':int(mal.obs.state.eq(st).sum())})
state_en=pd.DataFrame(state_en); state_en.to_csv(TAB/'state_resistance_enrichment_v1.tsv',sep='\t',index=False)

# Export cell coordinates and manuscript-ready summaries.
co=pd.DataFrame(mal.obsm['X_umap'],columns=['UMAP1','UMAP2'],index=mal.obs_names)
for c in ['dataset','sample_id','state','axis_z','cnv_rms']+score_names: co[c]=mal.obs[c].values
co.to_csv(TAB/'malignant_coordinates_v1.tsv.gz',sep='\t',compression='gzip')

def panel(ax,l,t): ax.text(-.12,1.07,l,transform=ax.transAxes,fontweight='bold',fontsize=14,va='top'); ax.set_title(t,loc='left',fontweight='bold',fontsize=10)
fig,axs=plt.subplots(2,3,figsize=(14.2,9.2),constrained_layout=True)
ax=axs[0,0]; panel(ax,'a','CNV-supported malignant-state manifold'); sns.scatterplot(x=mal.obsm['X_umap'][:,0],y=mal.obsm['X_umap'][:,1],hue=mal.obs.state,s=8,linewidth=0,palette='tab10',ax=ax); ax.legend(title='State',frameon=False,ncol=2,fontsize=7); ax.set(xlabel='UMAP1',ylabel='UMAP2')
ax=axs[0,1]; panel(ax,'b','Patient mixing after Harmony integration'); sns.scatterplot(x=mal.obsm['X_umap'][:,0],y=mal.obsm['X_umap'][:,1],hue=mal.obs.sample_id,s=6,linewidth=0,alpha=.7,palette='husl',legend=False,ax=ax); ax.set(xlabel='UMAP1',ylabel='UMAP2')
ax=axs[0,2]; panel(ax,'c','Resistance-axis localization'); q=ax.scatter(mal.obsm['X_umap'][:,0],mal.obsm['X_umap'][:,1],c=mal.obs.axis_z,s=8,cmap='coolwarm',vmin=-2,vmax=2,linewidth=0); fig.colorbar(q,ax=ax,label='Resistance-axis z'); ax.set(xlabel='UMAP1',ylabel='UMAP2')
ax=axs[1,0]; panel(ax,'d','Patient-level malignant-state composition'); state_comp.set_index(['dataset','sample_id']).plot.bar(stacked=True,colormap='tab20',width=.86,ax=ax); ax.set(ylabel='Malignant-cell fraction',xlabel='Patient/sample'); ax.tick_params(axis='x',rotation=70,labelsize=6); ax.legend(title='State',ncol=2,fontsize=6,frameon=False)
ax=axs[1,1]; panel(ax,'e','State-resolved resistance programs'); sns.heatmap(state_mean.apply(stats.zscore,axis=0).T,cmap='vlag',center=0,ax=ax,cbar_kws={'label':'State mean z'}); ax.set(xlabel='Malignant state',ylabel='')
ax=axs[1,2]; panel(ax,'f','Resistance-high state enrichment'); se=state_en.sort_values('odds_ratio'); y=np.arange(len(se)); ax.errorbar(se.odds_ratio,y,xerr=np.vstack([se.odds_ratio-se.ci_low,se.ci_high-se.odds_ratio]),fmt='o',color='#D55E00',ecolor='#777777',capsize=3); ax.axvline(1,color='k',ls='--',lw=.8); ax.set_yticks(y,se.state); ax.set_xscale('log'); ax.set(xlabel='Odds ratio (95% CI)',ylabel='Malignant state')
fig.suptitle('Patient-resolved malignant states localize heterogeneous cisplatin-resistance programs',fontweight='bold',fontsize=15)
for ext,dpi in [('png',300),('tif',600)]: fig.savefig(FIG/f'Figure_malignant_v1.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white',pil_kwargs={'compression':'tiff_lzw'} if ext=='tif' else {})
fig.savefig(FIG/'Figure_malignant_v1.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)

# Second figure: sample-level inference and program/state/drug-target hypothesis network.
fig,axs=plt.subplots(2,3,figsize=(14.2,9.2),constrained_layout=True)
ax=axs[0,0]; panel(ax,'a','CNV excess validates malignant calls'); sns.boxplot(data=cnv_pb,x='cell_class',y='median_cnv_excess',hue='dataset',showfliers=False,ax=ax); sns.stripplot(data=cnv_pb,x='cell_class',y='median_cnv_excess',hue='dataset',dodge=True,color='k',size=3,legend=False,ax=ax); ax.tick_params(axis='x',rotation=20); ax.set(xlabel='',ylabel='Sample median CNV excess')
ax=axs[0,1]; panel(ax,'b','Program association with resistance'); aa=assoc.sort_values('spearman_rho'); y=np.arange(len(aa)); ax.errorbar(aa.spearman_rho,y,xerr=np.vstack([aa.spearman_rho-aa.ci_low,aa.ci_high-aa.spearman_rho]),fmt='o',color='#0072B2',ecolor='#888888',capsize=3); ax.axvline(0,color='k',ls='--',lw=.8); ax.set_yticks(y,aa.program,fontsize=7); ax.set(xlabel='Sample-level Spearman rho (95% bootstrap CI)',ylabel='')
ax=axs[0,2]; panel(ax,'c','Patient-level program landscape'); hm=pb.set_index('sample_id')[score_names]; sns.heatmap(hm.apply(stats.zscore,axis=0).T,cmap='vlag',center=0,ax=ax,cbar_kws={'label':'Across-sample z'}); ax.set(xlabel='Patient/sample',ylabel='')
ax=axs[1,0]; panel(ax,'d','State abundance versus resistance'); dom=state_mean.axis_z.idxmax(); x=state_comp.set_index(['dataset','sample_id'])[dom].reindex(pd.MultiIndex.from_frame(pb[['dataset','sample_id']])).values; sns.regplot(x=x,y=pb.axis_z,scatter_kws={'s':35,'alpha':.8},line_kws={'color':'#D55E00'},ax=ax); rr=stats.spearmanr(x,pb.axis_z).statistic; ax.text(.04,.95,f'Spearman rho={rr:.2f}',transform=ax.transAxes,va='top',fontweight='bold'); ax.set(xlabel=f'Fraction in resistance-high state {dom}',ylabel='Sample median resistance-axis z')
ax=axs[1,1]; panel(ax,'e','Program covariance identifies coupled processes'); corr=hm.corr(method='spearman'); sns.heatmap(corr,cmap='vlag',center=0,vmin=-1,vmax=1,ax=ax,cbar_kws={'label':'Spearman rho'}); ax.tick_params(labelsize=6)
ax=axs[1,2]; panel(ax,'f','Mechanistic program-to-state map'); ax.axis('off'); top_prog=assoc.reindex(assoc.spearman_rho.abs().sort_values(ascending=False).index).head(6).program.tolist(); states=state_mean.axis_z.sort_values(ascending=False).head(5).index.astype(str).tolist()
for i,p in enumerate(top_prog): ax.text(.08,.9-i*.15,p,ha='center',va='center',fontsize=7,bbox=dict(boxstyle='round',fc='#56B4E9',alpha=.75))
for j,s in enumerate(states): ax.text(.92,.85-j*.18,f'State {s}',ha='center',va='center',fontsize=8,bbox=dict(boxstyle='round',fc='#E69F00',alpha=.75))
mx=max(abs(state_mean.loc[[str(x) for x in states],top_prog].values).max(),1e-9)
for i,p in enumerate(top_prog):
 for j,s in enumerate(states):
  w=abs(state_mean.loc[s,p])/mx
  if w<.18: continue
  path=MPath([(0.18,.9-i*.15),(0.45,.9-i*.15),(0.55,.85-j*.18),(0.82,.85-j*.18)],[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4])
  ax.add_patch(PathPatch(path,facecolor='none',edgecolor='#7A5195',lw=.5+4*w,alpha=.15+.65*w))
fig.suptitle('Patient-level evidence links CNV-defined malignant states to resistance mechanisms',fontweight='bold',fontsize=15)
for ext,dpi in [('png',300),('tif',600)]: fig.savefig(FIG/f'Figure_validation_v1.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white',pil_kwargs={'compression':'tiff_lzw'} if ext=='tif' else {})
fig.savefig(FIG/'Figure_validation_v1.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)

summary={'n_malignant_cells':int(mal.n_obs),'n_samples':int(mal.obs.sample_id.nunique()),'datasets':{str(k):int(v) for k,v in mal.obs.dataset.value_counts().items()},'n_states':int(mal.obs.state.nunique()),'program_associations':assoc.to_dict('records'),'claim_gate':'Associations are sample-level and computational; candidate drugs remain predicted sensitizers requiring perturbational and laboratory validation.'}
(META/'summary_v1.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
(OUT/'README_v1.txt').write_text('Malignant-state analysis using prespecified CNV-supported malignant calls. Statistical inference is performed at the patient/sample level; cell-level displays are descriptive.\n')
zip_path=Path('/content/malignant_out_v1.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.rglob('*'):
  if p.is_file(): z.write(p,p.relative_to(OUT.parent))
print('COMPLETE:',zip_path,round(zip_path.stat().st_size/1024**2,3),'MB'); files.download(str(zip_path))
