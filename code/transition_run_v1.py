"""Origin-held-out optimal-transport analysis of GSE117872 OSCC resistance states."""
from google.colab import drive, files
drive.mount('/content/drive')

import subprocess,sys
subprocess.check_call([sys.executable,'-m','pip','install','-q','POT==0.9.5','umap-learn>=0.5.6'])

from pathlib import Path
import hashlib,json,math,zipfile
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse,stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import umap
import ot
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch,Rectangle

SEED=20260811; rng=np.random.default_rng(SEED)
ROOT=Path('/content/drive/MyDrive/OSCC_Cisplatin_PhysicsInformed_StateTransitions')
OUT=Path('/content/transition_out_v1'); FIG=OUT/'figures';TAB=OUT/'tables';META=OUT/'metadata'
for p in [FIG,TAB,META]:p.mkdir(parents=True,exist_ok=True)
H5=ROOT/'03_analysis_ready/h5ad/GSE117872_log1pTPM_all_matched_cells.h5ad'
a=ad.read_h5ad(H5)
required={'origin','analysis_condition'}
if not required.issubset(a.obs):raise ValueError(f'Missing columns: {required-set(a.obs.columns)}')
a=a[a.obs.origin.isin(['HN120','HN137','HN148'])].copy()
a.obs['condition']=a.obs.analysis_condition.astype(str).map({
 'SENSITIVE_OR_PARENTAL':'Sensitive/parental','RESISTANT':'Resistant','DRUG_HOLIDAY_SEPARATE':'Drug holiday'})
if a.obs.condition.isna().any():raise ValueError('Unmapped analysis_condition values')

X=a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
genes=np.asarray(a.var_names.astype(str)); variances=X.var(axis=0)
hvg=np.argsort(variances)[-2500:]; Xh=X[:,hvg]; gh=genes[hvg]
Xs=StandardScaler().fit_transform(Xh); pca=PCA(n_components=30,random_state=SEED); Z=pca.fit_transform(Xs)
emb=umap.UMAP(n_neighbors=30,min_dist=.25,metric='euclidean',random_state=SEED).fit_transform(Z)
states=KMeans(n_clusters=6,n_init=50,random_state=SEED).fit_predict(Z)
a.obs['state']=['State '+str(x+1) for x in states]
coords=pd.DataFrame({'cell_id':a.obs_names,'UMAP1':emb[:,0],'UMAP2':emb[:,1],'origin':a.obs.origin.astype(str).values,'condition':a.obs.condition.values,'state':a.obs.state.values})
coords.to_csv(TAB/'cell_coordinates_v1.tsv.gz',sep='\t',index=False,compression='gzip')

def coupling(source_idx,target_idx):
    si=np.asarray(source_idx);ti=np.asarray(target_idx)
    # Keep all cells; groups are small. Scale cost by its positive median for stable epsilon.
    C=ot.dist(Z[si],Z[ti],metric='sqeuclidean');scale=np.median(C[C>0]);C=C/max(scale,1e-8)
    aa=np.full(len(si),1/len(si));bb=np.full(len(ti),1/len(ti))
    G=ot.sinkhorn(aa,bb,C,reg=.08,numItermax=5000,stopThr=1e-9,warn=False)
    return si,ti,G,C

flows=[];couplings={}
for origin in ['HN120','HN137']:
 for src,tgt,label in [('Sensitive/parental','Resistant','S_to_R'),('Resistant','Drug holiday','R_to_H')]:
  si=np.where((a.obs.origin.values==origin)&(a.obs.condition.values==src))[0]
  ti=np.where((a.obs.origin.values==origin)&(a.obs.condition.values==tgt))[0]
  if min(len(si),len(ti))<20:continue
  si,ti,G,C=coupling(si,ti);couplings[(origin,label)]=(si,ti,G,C)
  for s in range(6):
   for t in range(6):
    mass=G[np.ix_(states[si]==s,states[ti]==t)].sum()
    flows.append({'origin':origin,'transition':label,'source_state':f'State {s+1}','target_state':f'State {t+1}','mass':mass})
pd.DataFrame(flows).to_csv(TAB/'optimal_transport_state_flows_v1.tsv',sep='\t',index=False)

def rbf_mmd(x,y):
    x=np.asarray(x);y=np.asarray(y);z=np.vstack([x,y]);d=ot.dist(z,z)
    med=np.median(d[d>0]);gamma=1/max(2*med,1e-8)
    kxx=np.exp(-gamma*ot.dist(x,x));kyy=np.exp(-gamma*ot.dist(y,y));kxy=np.exp(-gamma*ot.dist(x,y))
    return float(kxx.mean()+kyy.mean()-2*kxy.mean())

# Origin-held-out transport-vector prediction and classifier generalization.
validation=[];predicted={}
for train,test in [('HN120','HN137'),('HN137','HN120')]:
 si,ti,G,_=couplings[(train,'S_to_R')]
 displacement={}
 for s in range(6):
  src=np.where(states[si]==s)[0];tgt=np.where(states[ti]==s)[0]
  if len(src) and len(tgt):
   block=G[np.ix_(src,tgt)];w=block/block.sum() if block.sum()>0 else block
   displacement[s]=(w.sum(0)[:,None]*Z[ti[tgt]]).sum(0)-(w.sum(1)[:,None]*Z[si[src]]).sum(0)
  else: displacement[s]=Z[ti].mean(0)-Z[si].mean(0)
 test_s=np.where((a.obs.origin.values==test)&(a.obs.condition.values=='Sensitive/parental'))[0]
 test_r=np.where((a.obs.origin.values==test)&(a.obs.condition.values=='Resistant'))[0]
 pred=np.vstack([Z[i]+displacement[states[i]] for i in test_s]);predicted[(train,test)]=pred
 validation.append({'train_origin':train,'test_origin':test,'metric':'MMD baseline-sensitive vs observed-resistant','value':rbf_mmd(Z[test_s],Z[test_r])})
 validation.append({'train_origin':train,'test_origin':test,'metric':'MMD transported vs observed-resistant','value':rbf_mmd(pred,Z[test_r])})
 tr=np.where((a.obs.origin.values==train)&a.obs.condition.isin(['Sensitive/parental','Resistant']).values)[0]
 te=np.where((a.obs.origin.values==test)&a.obs.condition.isin(['Sensitive/parental','Resistant']).values)[0]
 ytr=(a.obs.condition.values[tr]=='Resistant').astype(int);yte=(a.obs.condition.values[te]=='Resistant').astype(int)
 clf=LogisticRegression(C=.1,max_iter=5000,class_weight='balanced',random_state=SEED).fit(Z[tr],ytr)
 auc=roc_auc_score(yte,clf.predict_proba(Z[te])[:,1]);validation.append({'train_origin':train,'test_origin':test,'metric':'Held-out-origin resistance AUC','value':auc})
pd.DataFrame(validation).to_csv(TAB/'heldout_validation_v1.tsv',sep='\t',index=False)

# Origin-specific gene shifts and drug-holiday directional reversal.
effects=[];reversal=[]
for origin in ['HN120','HN137']:
 idx={c:np.where((a.obs.origin.values==origin)&(a.obs.condition.values==c))[0] for c in ['Sensitive/parental','Resistant','Drug holiday']}
 er=X[idx['Resistant']].mean(0)-X[idx['Sensitive/parental']].mean(0)
 eh=X[idx['Drug holiday']].mean(0)-X[idx['Resistant']].mean(0)
 for g,x,y in zip(genes,er,eh):effects.append({'origin':origin,'gene':g,'resistant_minus_sensitive':x,'holiday_minus_resistant':y})
 reversal.append({'origin':origin,'cosine_holiday_vs_resistance':float(np.dot(er,eh)/(np.linalg.norm(er)*np.linalg.norm(eh))),
                  'spearman_holiday_vs_resistance':stats.spearmanr(er,eh).statistic})
pd.DataFrame(effects).to_csv(TAB/'gene_shift_by_origin_v1.tsv.gz',sep='\t',index=False,compression='gzip')
pd.DataFrame(reversal).to_csv(TAB/'drug_holiday_reversal_v1.tsv',sep='\t',index=False)

wide=pd.DataFrame(effects).pivot(index='gene',columns='origin',values='resistant_minus_sensitive').dropna()
rho=stats.spearmanr(wide.HN120,wide.HN137).statistic

def panel(ax,letter,title):ax.text(-.12,1.07,letter,transform=ax.transAxes,fontweight='bold',fontsize=14,va='top');ax.set_title(title,loc='left',fontweight='bold',fontsize=10)
colors={'Sensitive/parental':'#0072B2','Resistant':'#D55E00','Drug holiday':'#009E73'}
fig,axs=plt.subplots(2,3,figsize=(14.1,9.1),constrained_layout=True)
ax=axs[0,0];panel(ax,'a','Three-state single-cell landscape')
for c in colors:
 m=a.obs.condition.values==c;ax.scatter(emb[m,0],emb[m,1],s=8,alpha=.65,c=colors[c],label=f'{c} (n={m.sum()})',rasterized=True)
ax.legend(frameon=False,fontsize=7);ax.set_xlabel('UMAP1');ax.set_ylabel('UMAP2')
ax=axs[0,1];panel(ax,'b','Origin-resolved reproducibility')
marks={'HN120':'o','HN137':'^','HN148':'s'}
for o,mk in marks.items():
 m=a.obs.origin.values==o;ax.scatter(emb[m,0],emb[m,1],s=9,alpha=.55,marker=mk,label=f'{o} (n={m.sum()})',rasterized=True)
ax.legend(frameon=False,fontsize=8);ax.set_xlabel('UMAP1');ax.set_ylabel('UMAP2')

# Alluvial transport mass, averaged across origins.
ax=axs[0,2];panel(ax,'c','Optimal-transport flow between cell states');ax.axis('off')
fl=pd.DataFrame(flows);fl=fl[fl.transition.eq('S_to_R')].groupby(['source_state','target_state']).mass.mean().reset_index()
left={f'State {i+1}':.88-i*.14 for i in range(6)};right=left.copy();mx=fl.mass.max()
for s,y in left.items():ax.add_patch(Rectangle((.03,y-.035),.09,.07,color='#56B4E9'));ax.text(.025,y,s,ha='right',va='center',fontsize=7)
for t,y in right.items():ax.add_patch(Rectangle((.88,y-.035),.09,.07,color='#E69F00'));ax.text(.975,y,t,ha='left',va='center',fontsize=7)
for r in fl.nlargest(18,'mass').itertuples():
 y0=left[r.source_state];y1=right[r.target_state];verts=[(.12,y0),(.42,y0),(.58,y1),(.88,y1)];path=MplPath(verts,[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4]);ax.add_patch(PathPatch(path,lw=.5+7*r.mass/mx,alpha=.35,color='#7A5195',fill=False))
ax.text(.075,.98,'Sensitive',ha='center',fontweight='bold');ax.text(.925,.98,'Resistant',ha='center',fontweight='bold')

ax=axs[1,0];panel(ax,'d','Independent-origin gene-shift concordance')
ax.hexbin(wide.HN120,wide.HN137,gridsize=55,bins='log',cmap='viridis',mincnt=1);ax.axhline(0,c='k',lw=.6);ax.axvline(0,c='k',lw=.6)
score=np.abs(wide.HN120)+np.abs(wide.HN137)
for g in score.nlargest(12).index:ax.text(wide.loc[g,'HN120'],wide.loc[g,'HN137'],g,fontsize=6)
ax.text(.03,.96,f'Spearman rho={rho:.3f}',transform=ax.transAxes,va='top',fontweight='bold');ax.set_xlabel('HN120 resistant − sensitive');ax.set_ylabel('HN137 resistant − sensitive')

ax=axs[1,1];panel(ax,'e','Origin-held-out transport validation')
vd=pd.DataFrame(validation);tests=['HN137','HN120'];x=np.arange(2);w=.25
base=[vd[(vd.test_origin==t)&vd.metric.str.startswith('MMD baseline')].value.iloc[0] for t in tests]
pred=[vd[(vd.test_origin==t)&vd.metric.str.startswith('MMD transported')].value.iloc[0] for t in tests]
auc=[vd[(vd.test_origin==t)&vd.metric.str.startswith('Held')].value.iloc[0] for t in tests]
ax.bar(x-w,base,w,label='Baseline MMD',color='#999999');ax.bar(x,pred,w,label='Transported MMD',color='#0072B2');ax.bar(x+w,auc,w,label='Held-out AUC',color='#D55E00')
ax.set_xticks(x,[f'Test {t}' for t in tests]);ax.set_ylabel('Metric value');ax.legend(frameon=False,fontsize=7)

ax=axs[1,2];panel(ax,'f','Drug-holiday reversal of resistance shifts')
rv=pd.DataFrame(reversal);yy=np.arange(len(rv));ax.barh(yy-.17,rv.cosine_holiday_vs_resistance,.32,label='Cosine',color='#009E73');ax.barh(yy+.17,rv.spearman_holiday_vs_resistance,.32,label='Spearman',color='#CC79A7');ax.axvline(0,c='k',lw=.7);ax.set_yticks(yy,rv.origin);ax.set_xlabel('Directional agreement (negative = reversal)');ax.legend(frameon=False,fontsize=8)
fig.suptitle('Origin-held-out optimal-transport analysis of OSCC cisplatin-resistance states',fontweight='bold',fontsize=15)
fig.savefig(FIG/'Figure_transition_v1.png',dpi=300,bbox_inches='tight',facecolor='white');fig.savefig(FIG/'Figure_transition_v1.tif',dpi=600,bbox_inches='tight',facecolor='white',pil_kwargs={'compression':'tiff_lzw'});fig.savefig(FIG/'Figure_transition_v1.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)

summary={'seed':SEED,'n_cells':a.n_obs,'n_genes':a.n_vars,'hvg':len(hvg),'origins':a.obs.origin.value_counts().to_dict(),'conditions':a.obs.condition.value_counts().to_dict(),'gene_shift_spearman':rho,'no_biological_replicate_warning':'HN120 and HN137 are the two paired origins; cell-level metrics are descriptive and origin-held-out, not n=cell inference.'}
(META/'analysis_summary_v1.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'README_v1.txt').write_text('Origin-held-out GSE117872 optimal-transport analysis. Results are hypothesis-generating; two paired origins do not support population-level efficacy claims.\n')
zip_path=Path('/content/transition_out_v1.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(OUT.parent))
print('COMPLETE:',zip_path,zip_path.stat().st_size/1024**2,'MB');files.download(str(zip_path))
