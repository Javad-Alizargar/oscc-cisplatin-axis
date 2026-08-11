"""Pathway and latent-program decomposition of origin-specific OSCC resistance."""
from google.colab import drive,files
drive.mount('/content/drive')
from pathlib import Path
import json,zipfile,requests
import anndata as ad
import numpy as np,pandas as pd
from scipy import sparse,stats
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

SEED=20260811;rng=np.random.default_rng(SEED)
ROOT=Path('/content/drive/MyDrive/OSCC_Cisplatin_PhysicsInformed_StateTransitions');OUT=Path('/content/program_out_v1');FIG=OUT/'figures';TAB=OUT/'tables';META=OUT/'metadata'
for p in [FIG,TAB,META]:p.mkdir(parents=True,exist_ok=True)
a=ad.read_h5ad(ROOT/'03_analysis_ready/h5ad/GSE117872_log1pTPM_all_matched_cells.h5ad')
a=a[a.obs.origin.isin(['HN120','HN137'])].copy();a.obs['condition']=a.obs.analysis_condition.astype(str).map({'SENSITIVE_OR_PARENTAL':'Sensitive','RESISTANT':'Resistant','DRUG_HOLIDAY_SEPARATE':'Holiday'})
X=a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X);genes=np.asarray(a.var_names.astype(str));gidx={g:i for i,g in enumerate(genes)}

def library(name):
 u='https://maayanlab.cloud/Enrichr/geneSetLibrary';r=requests.get(u,params={'mode':'text','libraryName':name},timeout=120);r.raise_for_status();out={}
 for line in r.text.splitlines():
  x=line.split('\t');
  if len(x)>=3:out[x[0]]=set(x[2:])
 return out
hall=library('MSigDB_Hallmark_2020');react=library('Reactome_2022')

# Standardized single-cell pathway scores, retaining pathways with adequate coverage.
Z=StandardScaler().fit_transform(X);sets={}
for source,lib in [('Hallmark',hall),('Reactome',react)]:
 for name,gs in lib.items():
  ix=[gidx[g] for g in gs if g in gidx]
  if 10<=len(ix)<=400:sets[(source,name)]=ix
P=np.column_stack([Z[:,ix].mean(1) for ix in sets.values()]);pnames=[f'{s}|{n}' for s,n in sets]

effects=[]
for origin in ['HN120','HN137']:
 for transition,c1,c0 in [('Resistance','Resistant','Sensitive'),('Holiday reversal','Holiday','Resistant')]:
  i1=np.where((a.obs.origin.values==origin)&(a.obs.condition.values==c1))[0];i0=np.where((a.obs.origin.values==origin)&(a.obs.condition.values==c0))[0]
  e=P[i1].mean(0)-P[i0].mean(0)
  for n,v in zip(pnames,e):effects.append({'origin':origin,'transition':transition,'pathway':n,'effect':v})
eff=pd.DataFrame(effects);eff.to_csv(TAB/'pathway_effects_v1.tsv',sep='\t',index=False)

# Non-negative matrix factorization programs with seed stability.
var=X.var(0);ix=np.argsort(var)[-3000:];Xn=X[:,ix];gn=genes[ix]
Ws=[];Hs=[]
for seed in [SEED,SEED+1,SEED+2,SEED+3,SEED+4]:
 m=NMF(n_components=12,init='nndsvda',random_state=seed,max_iter=1000,alpha_W=.01,l1_ratio=.1)
 Ws.append(m.fit_transform(np.maximum(Xn,0)));Hs.append(m.components_)
W=Ws[0];H=Hs[0];Wz=StandardScaler().fit_transform(W)
top=[]
for k in range(12):
 for rank,j in enumerate(np.argsort(H[k])[-30:][::-1],1):top.append({'program':f'P{k+1}','rank':rank,'gene':gn[j],'loading':H[k,j]})
pd.DataFrame(top).to_csv(TAB/'nmf_program_genes_v1.tsv',sep='\t',index=False)
pm=[]
for o in ['HN120','HN137']:
 for c in ['Sensitive','Resistant','Holiday']:
  m=(a.obs.origin.values==o)&(a.obs.condition.values==c)
  for k,v in enumerate(Wz[m].mean(0),1):pm.append({'origin':o,'condition':c,'program':f'P{k}','mean_z':v})
pm=pd.DataFrame(pm);pm.to_csv(TAB/'nmf_program_activity_v1.tsv',sep='\t',index=False)

# Held-out-origin classification using pathway activities.
cv=[]
for tr,te in [('HN120','HN137'),('HN137','HN120')]:
 itr=np.where((a.obs.origin.values==tr)&a.obs.condition.isin(['Sensitive','Resistant']).values)[0];ite=np.where((a.obs.origin.values==te)&a.obs.condition.isin(['Sensitive','Resistant']).values)[0]
 ytr=(a.obs.condition.values[itr]=='Resistant').astype(int);yte=(a.obs.condition.values[ite]=='Resistant').astype(int)
 for space,M in [('Pathway',P),('NMF program',Wz)]:
  clf=LogisticRegression(C=.1,max_iter=5000,class_weight='balanced',random_state=SEED).fit(M[itr],ytr);pr=clf.predict_proba(M[ite])[:,1]
  cv.append({'train_origin':tr,'test_origin':te,'space':space,'auc':roc_auc_score(yte,pr)})
pd.DataFrame(cv).to_csv(TAB/'heldout_program_validation_v1.tsv',sep='\t',index=False)

wide=eff[eff.transition.eq('Resistance')].pivot(index='pathway',columns='origin',values='effect').dropna();rho=stats.spearmanr(wide.HN120,wide.HN137).statistic
rev=eff[eff.transition.eq('Holiday reversal')].pivot(index='pathway',columns='origin',values='effect').dropna()
resmean=wide.mean(1);revmean=rev.mean(1);revrho=stats.spearmanr(resmean,revmean).statistic

def panel(ax,l,t):ax.text(-.12,1.07,l,transform=ax.transAxes,fontweight='bold',fontsize=14,va='top');ax.set_title(t,loc='left',fontweight='bold',fontsize=10)
fig,axs=plt.subplots(2,3,figsize=(14.1,9.1),constrained_layout=True)
ax=axs[0,0];panel(ax,'a','Pathway-level cross-origin concordance');ax.hexbin(wide.HN120,wide.HN137,gridsize=40,bins='log',cmap='magma',mincnt=1);ax.axhline(0,c='k',lw=.6);ax.axvline(0,c='k',lw=.6);ax.text(.03,.96,f'Spearman rho={rho:.3f}',transform=ax.transAxes,va='top',fontweight='bold');ax.set_xlabel('HN120 resistance effect');ax.set_ylabel('HN137 resistance effect')

ax=axs[0,1];panel(ax,'b','Most reproducible Hallmark programs');hallidx=[x for x in wide.index if x.startswith('Hallmark|')];score=wide.loc[hallidx].abs().mean(1)*(np.sign(wide.loc[hallidx].HN120)==np.sign(wide.loc[hallidx].HN137));sel=score.nlargest(18).index;hm=wide.loc[sel].rename(index=lambda x:x.split('|',1)[1].replace('HALLMARK_','').replace('_',' '));sns.heatmap(hm,cmap='vlag',center=0,ax=ax,cbar_kws={'label':'Resistance effect'});ax.set_xlabel('Origin');ax.set_ylabel('')

ax=axs[0,2];panel(ax,'c','Resistance and drug-holiday pathway vectors');ax.hexbin(resmean,revmean,gridsize=40,bins='log',cmap='viridis',mincnt=1);ax.axhline(0,c='k',lw=.6);ax.axvline(0,c='k',lw=.6);ax.text(.03,.96,f'Spearman rho={revrho:.3f}',transform=ax.transAxes,va='top',fontweight='bold');ax.set_xlabel('Mean resistance effect');ax.set_ylabel('Mean holiday − resistant effect')

ax=axs[1,0];panel(ax,'d','Latent program activity by origin and condition');mat=pm.assign(group=pm.origin+' '+pm.condition).pivot(index='program',columns='group',values='mean_z');sns.heatmap(mat,cmap='vlag',center=0,ax=ax,cbar_kws={'label':'Mean activity z'});ax.set_xlabel('');ax.set_ylabel('')

ax=axs[1,1];panel(ax,'e','Program-to-pathway mechanistic map');ax.axis('off')
# Link each NMF program to Hallmark pathways by top-gene Jaccard overlap.
edges=[]
tg=pd.DataFrame(top).groupby('program').gene.apply(set)
for pr,gs in tg.items():
 for name,hgs in hall.items():
  j=len(gs&hgs)/max(len(gs|hgs),1)
  if j>0:edges.append((pr,name.replace('HALLMARK_','').replace('_',' '),j))
edges=sorted(edges,key=lambda x:x[2],reverse=True)[:24];progs=sorted(set(x[0] for x in edges));paths=list(dict.fromkeys(x[1] for x in edges))[:12]
for i,p in enumerate(progs):ax.text(.08,.94-i*.075,p,ha='center',va='center',bbox=dict(boxstyle='round',fc='#56B4E9',alpha=.7),fontsize=8)
for i,p in enumerate(paths):ax.text(.92,.94-i*.075,p,ha='center',va='center',bbox=dict(boxstyle='round',fc='#E69F00',alpha=.7),fontsize=6)
for pr,pa,w in edges:
 if pa not in paths:continue
 y0=.94-progs.index(pr)*.075;y1=.94-paths.index(pa)*.075;ax.plot([.14,.86],[y0,y1],color='#7A5195',alpha=.25+.7*w/max(x[2] for x in edges),lw=.5+8*w/max(x[2] for x in edges))

ax=axs[1,2];panel(ax,'f','Held-out-origin program generalization');cvd=pd.DataFrame(cv);sns.barplot(data=cvd,x='test_origin',y='auc',hue='space',palette=['#0072B2','#009E73'],ax=ax);ax.axhline(.5,c='k',ls='--',lw=.8);ax.set_ylim(0,1);ax.set_xlabel('Held-out origin');ax.set_ylabel('Resistance AUC');ax.legend(frameon=False,fontsize=8)
fig.suptitle('Origin-specific and drug-holiday-reversible programs in OSCC cisplatin resistance',fontweight='bold',fontsize=15)
for ext,dpi in [('png',300),('tif',600)]:fig.savefig(FIG/f'Figure_programs_v1.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white',pil_kwargs={'compression':'tiff_lzw'} if ext=='tif' else {})
fig.savefig(FIG/'Figure_programs_v1.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)

summary={'pathway_cross_origin_spearman':rho,'holiday_vs_resistance_pathway_spearman':revrho,'heldout':cv,'n_hallmark':len(hall),'n_reactome':len(react),'interpretation_gate':'Only pathway/program convergence with cross-origin support should enter the main manuscript.'}
(META/'analysis_summary_v1.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'README_v1.txt').write_text('Pathway and NMF analysis of origin-specific resistance and drug-holiday reversal. Cell-level classification is descriptive; biological replication remains two paired origins.\n')
zip_path=Path('/content/program_out_v1.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(OUT.parent))
print('COMPLETE:',zip_path,zip_path.stat().st_size/1024**2,'MB');files.download(str(zip_path))
