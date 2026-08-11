from google.colab import drive,files
drive.mount('/content/drive')
import os,subprocess,sys
try: import scvi,scanpy as sc
except Exception:
 subprocess.check_call([sys.executable,'-m','pip','install','-q','scvi-tools==1.3.3','scanpy>=1.10','scikit-misc'])
 print('Installation complete. Runtime will restart; rerun this cell after reconnection.');os.kill(os.getpid(),9)
from pathlib import Path
import gc
import json,zipfile
import anndata as ad,numpy as np,pandas as pd
from scipy import sparse
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt,seaborn as sns
SEED=20260811;scvi.settings.seed=SEED
ROOT=Path('/content/drive/MyDrive/OSCC_Cisplatin_PhysicsInformed_StateTransitions');OUT=Path('/content/atlas_out_v1');FIG=OUT/'figures';TAB=OUT/'tables';META=OUT/'metadata'
for p in [FIG,TAB,META]:p.mkdir(parents=True,exist_ok=True)
def load(ds):
 c=ad.read_h5ad(ROOT/f'03_analysis_ready/h5ad/{ds}_counts_QC.h5ad');s=ad.read_h5ad(ROOT/f'03_analysis_ready/state_space/{ds}_lineage_CNV_resistance_scored.h5ad',backed='r')
 cols=['predicted_lineage','malignant_epithelial_evidence','resistance_axis_dataset_z','resistance_axis_sample_z'];c.obs=c.obs.join(s.obs[cols],how='left')
 sym=c.var.gene_symbol.astype(str);keep=sym.notna()&~sym.duplicated()&(sym!='nan');c=c[:,keep].copy();c.var_names=sym[keep].values;c.obs['dataset']=ds;return c
a1,a2=load('GSE172577'),load('GSE215403');common=a1.var_names.intersection(a2.var_names)
a=ad.concat([a1[:,common],a2[:,common]],join='inner',merge='same',index_unique='-');del a1,a2;gc.collect()
a.layers['counts']=a.X.copy();sc.pp.highly_variable_genes(a,n_top_genes=3000,flavor='seurat_v3',batch_key='sample_id',layer='counts',subset=True)
# Preserve counts in the sparse layer; normalize X without zero-centering or densification.
sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a);sc.tl.pca(a,n_comps=30,zero_center=False,random_state=SEED)
scvi.model.SCVI.setup_anndata(a,layer='counts',batch_key='sample_id',categorical_covariate_keys=['dataset'])
model=scvi.model.SCVI(a,n_latent=20,n_layers=2,n_hidden=128,gene_likelihood='nb');model.train(max_epochs=150,early_stopping=True,check_val_every_n_epoch=5)
a.obsm['X_scVI']=model.get_latent_representation();sc.pp.neighbors(a,use_rep='X_scVI',n_neighbors=30);sc.tl.umap(a,random_state=SEED)
a.obs['scvi_cluster']=pd.Categorical(KMeans(n_clusters=14,n_init=30,random_state=SEED).fit_predict(a.obsm['X_scVI']).astype(str))
def entropy(rep,labels,k=30,n=15000):
 rng=np.random.default_rng(SEED);ix=rng.choice(len(rep),min(n,len(rep)),False);nn=NearestNeighbors(n_neighbors=k+1).fit(rep).kneighbors(rep[ix],return_distance=False)[:,1:];lab=np.asarray(labels);out=[]
 for q in nn:
  p=pd.Series(lab[q]).value_counts(normalize=True).values;out.append(-(p*np.log(p+1e-12)).sum()/np.log(max(len(np.unique(lab)),2)))
 return np.mean(out)
rng=np.random.default_rng(SEED);ix=rng.choice(a.n_obs,min(15000,a.n_obs),False);metrics=[]
for name,rep in [('Sparse PCA',a.obsm['X_pca']),('scVI',a.obsm['X_scVI'])]:
 metrics.extend([{'representation':name,'metric':'Dataset mixing entropy','value':entropy(rep,a.obs.dataset)},{'representation':name,'metric':'Lineage silhouette','value':silhouette_score(rep[ix],a.obs.predicted_lineage.astype(str).values[ix])}])
pd.DataFrame(metrics).to_csv(TAB/'integration_metrics_v1.tsv',sep='\t',index=False)
co=pd.DataFrame(a.obsm['X_umap'],columns=['UMAP1','UMAP2']);co['cell_id']=a.obs_names
for c in ['dataset','sample_id','predicted_lineage','malignant_epithelial_evidence','resistance_axis_dataset_z','scvi_cluster']:co[c]=a.obs[c].values
co.to_csv(TAB/'atlas_coordinates_v1.tsv.gz',sep='\t',index=False,compression='gzip');comp=pd.crosstab(a.obs.sample_id,a.obs.predicted_lineage,normalize='index');comp.to_csv(TAB/'sample_composition_v1.tsv',sep='\t')
markers=['EPCAM','KRT8','KRT18','PTPRC','CD3D','CD79A','LST1','COL1A1','DCN','PECAM1','VWF','RGS5','ACTA2'];markers=[g for g in markers if g in a.var_names]
avg=[]
for lin in sorted(a.obs.predicted_lineage.astype(str).unique()):
 m=a.obs.predicted_lineage.astype(str).values==lin;x=a[m,markers].X;x=x.toarray() if sparse.issparse(x) else np.asarray(x)
 for j,g in enumerate(markers):avg.append({'lineage':lin,'gene':g,'mean':x[:,j].mean(),'fraction':(x[:,j]>0).mean()})
pd.DataFrame(avg).to_csv(TAB/'marker_summary_v1.tsv',sep='\t',index=False)
def panel(ax,l,t):ax.text(-.12,1.07,l,transform=ax.transAxes,fontweight='bold',fontsize=14,va='top');ax.set_title(t,loc='left',fontweight='bold',fontsize=10)
fig,axs=plt.subplots(2,3,figsize=(14.1,9.1),constrained_layout=True)
for ax,l,title,hue,pal in [(axs[0,0],'a','scVI atlas by lineage',a.obs.predicted_lineage,'tab10'),(axs[0,1],'b','Cohort mixing in shared latent space',a.obs.dataset,['#0072B2','#D55E00'])]:
 panel(ax,l,title);sns.scatterplot(x=a.obsm['X_umap'][:,0],y=a.obsm['X_umap'][:,1],hue=hue,s=2,linewidth=0,alpha=.55,ax=ax,palette=pal);ax.legend(markerscale=4,fontsize=7,frameon=False);ax.set(xlabel='UMAP1',ylabel='UMAP2')
ax=axs[0,2];panel(ax,'c','Resistance-axis localization in malignant cells');m=a.obs.malignant_epithelial_evidence.astype(str).eq('MALIGNANT_HIGH').values;q=ax.scatter(a.obsm['X_umap'][m,0],a.obsm['X_umap'][m,1],c=a.obs.resistance_axis_dataset_z.values[m],s=5,cmap='coolwarm',vmin=-2,vmax=2);fig.colorbar(q,ax=ax,label='Axis z');ax.set(xlabel='UMAP1',ylabel='UMAP2');ax.text(.02,.02,f'n={m.sum():,}',transform=ax.transAxes)
ax=axs[1,0];panel(ax,'d','Patient/sample cellular composition');comp.plot.bar(stacked=True,ax=ax,colormap='tab20',width=.85);ax.set(ylabel='Cell fraction',xlabel='Sample');ax.tick_params(axis='x',rotation=70,labelsize=6);ax.legend(fontsize=6,frameon=False,ncol=2)
ax=axs[1,1];panel(ax,'e','Lineage-marker expression');pv=pd.DataFrame(avg).pivot(index='gene',columns='lineage',values='mean');sns.heatmap(pv,cmap='viridis',ax=ax,cbar_kws={'label':'Mean log expression'});ax.set(xlabel='',ylabel='')
ax=axs[1,2];panel(ax,'f','Quantitative integration diagnostics');sns.barplot(data=pd.DataFrame(metrics),x='metric',y='value',hue='representation',ax=ax,palette=['#999999','#009E73']);ax.set(xlabel='',ylabel='Metric value');ax.tick_params(axis='x',rotation=20);ax.legend(frameon=False)
fig.suptitle('A patient-resolved single-cell atlas of oral squamous cell carcinoma',fontsize=15,fontweight='bold')
fig.savefig(FIG/'Figure_atlas_v1.png',dpi=300,bbox_inches='tight');fig.savefig(FIG/'Figure_atlas_v1.tif',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'});fig.savefig(FIG/'Figure_atlas_v1.pdf',bbox_inches='tight');plt.close(fig)
summary={'n_cells':a.n_obs,'n_samples':a.obs.sample_id.nunique(),'n_common_genes':len(common),'n_hvg':a.n_vars,'datasets':a.obs.dataset.value_counts().to_dict(),'lineages':a.obs.predicted_lineage.value_counts().to_dict(),'metrics':metrics};(META/'summary_v1.json').write_text(json.dumps(summary,indent=2)+'\n')
zip_path=Path('/content/atlas_out_v1.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(OUT.parent))
print('COMPLETE:',zip_path,zip_path.stat().st_size/1024**2,'MB');files.download(str(zip_path))
