"""State-5-specific LINCS query with orthogonal PRISM and frozen-axis validation."""
from google.colab import drive,files
drive.mount('/content/drive')
import os,sys,subprocess
try: import scanpy,h5py
except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q','scanpy>=1.10','h5py','openpyxl'])
from pathlib import Path
import json,zipfile,re,gc
import anndata as ad,numpy as np,pandas as pd,h5py
from scipy import sparse,stats
import matplotlib.pyplot as plt,seaborn as sns
from matplotlib.patches import FancyBboxPatch

SEED=20260812;rng=np.random.default_rng(SEED)
ROOT=Path('/content/drive/MyDrive/OSCC_Cisplatin_PhysicsInformed_StateTransitions');OUT=Path('/content/drug_link_out_v1');FIG=OUT/'figures';TAB=OUT/'tables';META=OUT/'metadata'
for p in (FIG,TAB,META):p.mkdir(parents=True,exist_ok=True)
STATE_FILE=Path('/content/malignant_coordinates_v1.tsv.gz')
if not STATE_FILE.exists():
 candidates=list(ROOT.rglob('malignant_coordinates_v1.tsv.gz'))
 if candidates:STATE_FILE=candidates[0]
if not STATE_FILE.exists():raise FileNotFoundError('Upload malignant_coordinates_v1.tsv.gz from malignant_out_v1/tables to /content before running.')
coords=pd.read_csv(STATE_FILE,sep='\t',index_col=0);coords.index=coords.index.astype(str).str.replace(r'-[01]$','',regex=True)

def load(ds):
 a=ad.read_h5ad(ROOT/f'03_analysis_ready/h5ad/{ds}_counts_QC.h5ad');sym=a.var.gene_symbol.astype(str) if 'gene_symbol' in a.var else a.var_names.astype(str);keep=sym.notna()&~sym.duplicated()&(sym!='nan');a=a[:,keep].copy();a.var_names=sym[keep].values
 m=coords[coords.dataset.eq(ds)].copy(); common=a.obs_names.intersection(m.index);a=a[common].copy();a.obs=a.obs.join(m[['sample_id','state','axis_z']],how='left',rsuffix='_state');a.obs['dataset']=ds;return a
a1,a2=load('GSE172577'),load('GSE215403');common=a1.var_names.intersection(a2.var_names);a=ad.concat([a1[:,common],a2[:,common]],join='inner',index_unique='-');del a1,a2;gc.collect()
X=a.X.tocsr() if sparse.issparse(a.X) else sparse.csr_matrix(a.X);lib=np.asarray(X.sum(1)).ravel();
# Within-sample state-5 versus other-malignant logCPM contrasts.
effects=[]
for (ds,sid),idx in a.obs.groupby(['dataset','sample_id']).indices.items():
 idx=np.asarray(idx);is5=a.obs.iloc[idx].state.astype(str).eq('5').values
 if is5.sum()<5 or (~is5).sum()<20:continue
 def pb(q):return np.log2(1+np.asarray(X[idx[q]].sum(0)).ravel()/max(lib[idx[q]].sum(),1)*1e6)
 e=pb(is5)-pb(~is5);effects.append(pd.Series(e,index=a.var_names,name=f'{ds}::{sid}'))
E=pd.DataFrame(effects);med=E.median();cons=np.maximum((E>0).mean(),(E<0).mean());sig=pd.DataFrame({'gene':med.index,'median_log2cpm_difference':med.values,'sign_consistency':cons.values,'n_samples':len(E)})
sig=sig.sort_values('median_log2cpm_difference');sig['selected']='';sig.loc[sig.tail(150).index,'selected']='UP';sig.loc[sig.head(150).index,'selected']='DOWN';sig.to_csv(TAB/'state5_signature_v1.tsv',sep='\t',index=False);E.T.to_csv(TAB/'state5_sample_contrasts_v1.tsv.gz',sep='\t',compression='gzip')
up=sig.loc[sig.selected.eq('UP'),'gene'].tolist();down=sig.loc[sig.selected.eq('DOWN'),'gene'].tolist()

RAW=ROOT/'01_raw_data/LINCS_GSE70138';GCTX=ROOT/'03_analysis_ready/drug_reversal/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx'
gene_info=pd.read_csv(RAW/'GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz',sep='\t',low_memory=False);gid=next(c for c in ['pr_gene_id','gene_id'] if c in gene_info);gsym=next(c for c in ['pr_gene_symbol','gene_symbol'] if c in gene_info);gene_info[gsym]=gene_info[gsym].astype(str).str.upper()
siginfo=pd.read_csv(RAW/'GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz',sep='\t',low_memory=False);metrics=pd.read_csv(RAW/'GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz',sep='\t',low_memory=False)
with h5py.File(GCTX,'r') as h:
 M=h['/0/DATA/0/matrix'];rowids=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in h['/0/META/ROW/id'][:]]);colids=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in h['/0/META/COL/id'][:]])
 order={str(x):i for i,x in enumerate(rowids)};gi=gene_info.set_index(gid.astype(str) if False else gid)
 mapidx=dict(zip(gene_info[gsym],gene_info[gid].astype(str).map(order)))
 # Freeze the query after identifier mapping; subsequent sensitivity analyses
 # use exactly this mapped universe and cannot silently reintroduce absent genes.
 up=[g for g in up if g.upper() in mapidx and pd.notna(mapidx[g.upper()])]
 down=[g for g in down if g.upper() in mapidx and pd.notna(mapidx[g.upper()])]
 ui=[mapidx[g.upper()] for g in up];di=[mapidx[g.upper()] for g in down]
 ui=np.array(ui,int);di=np.array(di,int)
 if min(len(ui),len(di))<75:raise ValueError(f'LINCS mapping insufficient: UP={len(ui)}, DOWN={len(di)}')
 # GCTX is signatures x genes. Read selected gene columns, limiting memory to ~150 MB.
 A=np.asarray(M[:,np.sort(np.r_[ui,di])],dtype=np.float32);sel=np.sort(np.r_[ui,di]);pos={x:i for i,x in enumerate(sel)}
 score=A[:,[pos[x] for x in di]].mean(1)-A[:,[pos[x] for x in ui]].mean(1)
ls=pd.DataFrame({'sig_id':colids,'state5_reversal':score});ls=ls.merge(siginfo,on='sig_id',how='left').merge(metrics[['sig_id','tas','distil_nsample']],on='sig_id',how='left');ls.to_csv(TAB/'state5_lincs_signature_scores_v1.tsv.gz',sep='\t',index=False,compression='gzip')
q=ls[ls.pert_type.eq('trt_cp') & ls.tas.ge(.1) & ls.distil_nsample.ge(2)].copy();q['condition']=q.cell_id.astype(str)+'|'+q.pert_idose.astype(str)+'|'+q.pert_itime.astype(str)
cond=q.groupby(['pert_id','pert_iname','cell_id','condition'],as_index=False).agg(reversal=('state5_reversal','median'),tas=('tas','median'))
cell=cond.groupby(['pert_id','pert_iname','cell_id'],as_index=False).agg(reversal=('reversal','median'),n_conditions=('condition','nunique'))
rank=cell.groupby(['pert_id','pert_iname'],as_index=False).agg(state5_reversal=('reversal','median'),q25=('reversal',lambda x:x.quantile(.25)),q75=('reversal',lambda x:x.quantile(.75)),n_lincs_cell_lines=('cell_id','nunique'),fraction_positive=('reversal',lambda x:(x>0).mean()))
rank['state5_rank']=rank.state5_reversal.rank(ascending=False,method='min');rank['state5_percentile']=rank.state5_reversal.rank(pct=True);rank=rank.sort_values('state5_rank')

# Leave-one-sample-out query stability using the same 300 selected genes and patient contrasts.
sub=sig.set_index('gene').loc[up+down];los=[]
for omit in E.index:
 eff=E.drop(index=omit).median();u=eff.loc[up].sort_values(ascending=False).head(150).index;d=eff.loc[down].sort_values().head(150).index
 # Fixed selected universe: signed, magnitude-weighted cosine-like reversal.
 wi=np.r_[np.abs(eff.loc[d].values),np.abs(eff.loc[u].values)];ii=np.r_[[pos[mapidx[g.upper()]] for g in d],[pos[mapidx[g.upper()]] for g in u]];sgn=np.r_[np.ones(len(d)),-np.ones(len(u))]
 s=(A[:,ii]*(wi*sgn)).sum(1)/(np.sum(wi)+1e-9);tmp=pd.DataFrame({'sig_id':colids,'s':s}).merge(siginfo,on='sig_id').merge(metrics[['sig_id','tas','distil_nsample']],on='sig_id');tmp=tmp[tmp.pert_type.eq('trt_cp')&tmp.tas.ge(.1)&tmp.distil_nsample.ge(2)];rr=tmp.groupby('pert_id').s.median().rank(pct=True)
 los.append(pd.DataFrame({'pert_id':rr.index,'omitted_sample':omit,'percentile':rr.values}))
lopo=pd.concat(los);stab=lopo.groupby('pert_id').percentile.agg(['median','min']).reset_index().rename(columns={'median':'LOSO_median_percentile','min':'LOSO_min_percentile'});rank=rank.merge(stab,on='pert_id',how='left')

# Orthogonal evidence: old frozen-axis ranking and PRISM single-dose validation.
old=pd.read_csv(ROOT/'03_analysis_ready/drug_reversal/tables/integrated_LINCS_PRISM_drug_candidates.tsv.gz',sep='\t',low_memory=False).rename(columns={'pert_core':'pert_id','lincs_rank':'frozen_axis_rank','lincs_median_reversal':'frozen_axis_reversal'})
keep=['pert_id','frozen_axis_rank','frozen_axis_reversal','LOPO_candidate_set_min_percentile','evidence_tier','prism_uadt_n_cell_lines','prism_uadt_median_LFC','prism_pan_median_LFC']
rank=rank.merge(old[keep].drop_duplicates('pert_id'),on='pert_id',how='left');rank['PRISM_supported']=rank.prism_uadt_n_cell_lines.fillna(0).ge(8)&rank.prism_uadt_median_LFC.fillna(0).lt(0)
rank['state5_robust']=rank.n_lincs_cell_lines.ge(5)&rank.fraction_positive.ge(.7)&rank.LOSO_min_percentile.ge(.75)
rank['frozen_axis_top10pct']=rank.frozen_axis_rank.le(old.frozen_axis_rank.quantile(.1))
rank['evidence_count']=rank[['state5_robust','PRISM_supported','frozen_axis_top10pct']].sum(1)
rank['priority_class']=np.select([rank.state5_robust&rank.PRISM_supported&rank.frozen_axis_top10pct,rank.state5_robust&rank.PRISM_supported,rank.state5_robust&rank.frozen_axis_top10pct,rank.state5_robust],['A_three-way','B_state5_PRISM','B_dual_transcriptomic','C_state5_only'],'D_unconfirmed')
rank=rank.sort_values(['evidence_count','state5_rank'],ascending=[False,True]);rank.to_csv(TAB/'state5_integrated_drug_ranking_v1.tsv',sep='\t',index=False);lopo.to_csv(TAB/'state5_leave_one_sample_out_v1.tsv.gz',sep='\t',index=False,compression='gzip')

def panel(ax,l,t):ax.text(-.12,1.07,l,transform=ax.transAxes,fontweight='bold',fontsize=14,va='top');ax.set_title(t,loc='left',fontweight='bold',fontsize=10)
sns.set_theme(style='white',font_scale=.85);fig,axs=plt.subplots(2,3,figsize=(14.2,9.2),constrained_layout=True)
ax=axs[0,0];panel(ax,'a','State-5 signature reproducibility');show=sig[sig.selected.ne('')].sort_values('median_log2cpm_difference');ax.scatter(show.median_log2cpm_difference,show.sign_consistency,c=np.where(show.selected.eq('UP'),'#D55E00','#0072B2'),s=12,alpha=.75);ax.axvline(0,c='k',lw=.7);ax.set(xlabel='Median within-patient log2CPM difference',ylabel='Sign consistency')
ax=axs[0,1];panel(ax,'b','State-5 reversal across LINCS cell lines');top=rank.head(15).pert_id;hh=cell[cell.pert_id.isin(top)].pivot(index='pert_iname',columns='cell_id',values='reversal');hh=hh.loc[rank[rank.pert_id.isin(top)].pert_iname];sns.heatmap(hh,cmap='vlag',center=0,ax=ax,cbar_kws={'label':'Reversal score'});ax.set(xlabel='LINCS cell line',ylabel='')
ax=axs[0,2];panel(ax,'c','Cross-screen concordance');z=rank.dropna(subset=['frozen_axis_rank']);ax.scatter(z.state5_rank,z.frozen_axis_rank,c=np.where(z.PRISM_supported,'#D55E00','#999999'),s=18,alpha=.7);ax.set_xscale('log');ax.set_yscale('log');ax.set(xlabel='State-5 LINCS rank',ylabel='Frozen-axis LINCS rank');r=stats.spearmanr(z.state5_rank,z.frozen_axis_rank).statistic;ax.text(.04,.95,f'Spearman rho={r:.2f}',transform=ax.transAxes,va='top',fontweight='bold')
ax=axs[1,0];panel(ax,'d','Leave-one-sample-out stability');top=rank.head(15);zz=lopo[lopo.pert_id.isin(top.pert_id)].merge(top[['pert_id','pert_iname']],on='pert_id');sns.boxplot(data=zz,y='pert_iname',x='percentile',order=top.pert_iname,showfliers=False,color='#88CCEE',ax=ax);ax.axvline(.75,c='#D55E00',ls='--');ax.set(xlabel='Ranking percentile across omissions',ylabel='')
ax=axs[1,1];panel(ax,'e','PRISM upper-aerodigestive depletion');z=rank[rank.prism_uadt_n_cell_lines.fillna(0).ge(8)].sort_values('state5_rank').head(15).sort_values('prism_uadt_median_LFC');ax.barh(z.pert_iname,z.prism_uadt_median_LFC,color=np.where(z.prism_uadt_median_LFC<0,'#009E73','#CC6677'));ax.axvline(0,c='k',lw=.7);ax.set(xlabel='Median single-dose LFC (lower = depletion)',ylabel='')
ax=axs[1,2];panel(ax,'f','Independent evidence convergence');top=rank.head(15);em=top.set_index('pert_iname')[['state5_robust','PRISM_supported','frozen_axis_top10pct']].astype(float);sns.heatmap(em,cmap=sns.color_palette(['#F2F2F2','#228833'],as_cmap=True),vmin=0,vmax=1,cbar=False,linewidths=.5,ax=ax);ax.set_xticklabels(['State-5\nrobust','PRISM\ndepletion','Frozen-axis\ntop decile'],rotation=0);ax.set(ylabel='')
fig.suptitle('State-specific and orthogonal prioritization of predicted cisplatin sensitizers',fontweight='bold',fontsize=15)
for ext,dpi in [('png',300),('tif',600)]:fig.savefig(FIG/f'Figure_drug_link_v1.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white',pil_kwargs={'compression':'tiff_lzw'} if ext=='tif' else {})
fig.savefig(FIG/'Figure_drug_link_v1.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)
summary={'state5_samples_used':len(E),'signature_up':len(up),'signature_down':len(down),'lincs_up_mapped':len(ui),'lincs_down_mapped':len(di),'n_compounds':len(rank),'priority_counts':rank.priority_class.value_counts().to_dict(),'top_candidates':rank.head(20)[['pert_iname','priority_class','state5_rank','state5_reversal','LOSO_min_percentile','PRISM_supported','frozen_axis_rank']].to_dict('records'),'claim_gate':'Predicted sensitizers for laboratory testing; no efficacy, synergy, or clinical benefit is established.'}
(META/'summary_v1.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
zip_path=Path('/content/drug_link_out_v1.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(OUT.parent))
print('COMPLETE:',zip_path,round(zip_path.stat().st_size/1024**2,3),'MB');files.download(str(zip_path))
