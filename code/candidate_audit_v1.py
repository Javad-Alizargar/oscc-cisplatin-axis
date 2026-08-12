from pathlib import Path
import json
import pandas as pd,numpy as np
import matplotlib.pyplot as plt,seaborn as sns
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
ROOT=Path(__file__).resolve().parent.parent;STEP=Path(__file__).resolve().parent;OUT=STEP/'results_v1';TAB=OUT/'tables';FIG=OUT/'figures';META=OUT/'metadata'
rank=pd.read_csv(ROOT/'step6_drug_link_v1/results/drug_link_out_v1/tables/state5_integrated_drug_ranking_v1.tsv',sep='\t')
names=['bisindolylmaleimide','CPI-613','picolinic-acid','moxonidine','teriflunomide','rottlerin','trifluoperazine','selumetinib','trametinib','gemcitabine']
d=rank[rank.pert_iname.isin(names)].copy()
audit={
'bisindolylmaleimide':dict(identity='Unresolved BRD-K31342827 screening reagent',mechanism='Do not assign: class members differ in PKC and transporter activity',direct_oscc='None located',combination='No compound-specific cisplatin evidence located',negative='Exact structure and target unresolved',maximum_claim='Three-screen computational hit requiring identity resolution',pmids='26657401|17575116'),
'CPI-613':dict(identity='Devimistat; lipoate analogue',mechanism='Mitochondrial pyruvate- and α-ketoglutarate-dehydrogenase perturbation',direct_oscc='No direct OSCC validation located',combination='Preclinical synergy with gemcitabine/cisplatin in biliary cancer',negative='Phase III pancreatic and AML trials did not establish benefit',maximum_claim='Leading dual-transcriptomic predicted sensitizer',pmids='37115501|39088774|40454396'),
'picolinic-acid':dict(identity='Picolinic acid',mechanism='Metal-chelating metabolite; mechanism not established here',direct_oscc='None located',combination='None located',negative='Weak PRISM depletion and no frozen-axis top-decile support',maximum_claim='Secondary state-5/PRISM hypothesis',pmids=''),
'moxonidine':dict(identity='Moxonidine',mechanism='Imidazoline/adrenergic receptor agonist',direct_oscc='None located',combination='None located',negative='No PRISM evidence',maximum_claim='Dual-transcriptomic hypothesis only',pmids=''),
'teriflunomide':dict(identity='Teriflunomide',mechanism='Dihydroorotate dehydrogenase inhibitor',direct_oscc='None located in this audit',combination='No direct cisplatin-resensitization evidence verified',negative='No PRISM evidence',maximum_claim='Dual-transcriptomic hypothesis only',pmids=''),
'rottlerin':dict(identity='Rottlerin',mechanism='Pleiotropic mitochondrial/kinase probe; not a selective PKCδ inhibitor',direct_oscc='None verified',combination='None verified',negative='Mechanistically promiscuous; no PRISM evidence',maximum_claim='Low-specificity transcriptomic hypothesis',pmids=''),
'trifluoperazine':dict(identity='Trifluoperazine',mechanism='Dopamine-receptor antagonist with calmodulin-related effects',direct_oscc='None verified',combination='No direct OSCC cisplatin evidence verified',negative='No PRISM evidence',maximum_claim='Dual-transcriptomic hypothesis only',pmids=''),
'selumetinib':dict(identity='Selumetinib',mechanism='MEK1/2 inhibitor',direct_oscc='No state-specific validation',combination='Mechanistic relevance possible, not validated by this study',negative='No matched PRISM support in current screen',maximum_claim='Dual-transcriptomic hypothesis only',pmids=''),
'trametinib':dict(identity='Trametinib',mechanism='MEK1/2 inhibitor',direct_oscc='No state-specific validation',combination='Mechanistic relevance possible, not validated by this study',negative='No matched PRISM support in current screen',maximum_claim='Dual-transcriptomic hypothesis only',pmids=''),
'gemcitabine':dict(identity='Gemcitabine',mechanism='Antimetabolite/nucleoside analogue',direct_oscc='Not a state-specific OSCC sensitization result',combination='Established chemotherapy; state-5 LINCS-only signal here',negative='Absent from independent frozen-axis table and current PRISM match',maximum_claim='State-5 reversal control/hypothesis',pmids='')}
for c in ['identity','mechanism','direct_oscc','combination','negative','maximum_claim','pmids']:d[c]=d.pert_iname.map(lambda x:audit[x][c])
d['identity_resolved']=~d.pert_iname.eq('bisindolylmaleimide');d['direct_OSCC_cisplatin_evidence']=False;d['PRISM_depletion']=d.PRISM_supported;d['dual_transcriptomic']=d.frozen_axis_top10pct;d['state5_robust']=d.state5_robust
d=d.sort_values(['evidence_count','state5_rank'],ascending=[False,True]);d.to_csv(TAB/'candidate_evidence_audit_v1.tsv',sep='\t',index=False)
refs=pd.DataFrame([
[1,37115501,'10.1158/1078-0432.CCR-22-3505','Devimistat with gemcitabine and cisplatin in biliary tract cancer','CPI-613 combination precedent; different cancer'],[2,39088774,'10.1200/JCO.23.02659','AVENGER 500 phase III study','Negative phase III evidence for devimistat'],[3,40454396,'','ARMADA phase III trial','Negative/limiting phase III evidence for devimistat'],[4,26657401,'10.1371/journal.pone.0144667','PKC-independent transporter effects of Ro 31-8220','Chemical-class caution; not identity proof'],[5,17575116,'10.1158/1535-7163.MCT-06-0811','ABCG2 inhibition by bisindolylmaleimides','Chemical-class transporter evidence; not identity proof'],[6,29523854,'','Cisplatin oxidative stress and carbon metabolism in HNSCC','HNSCC metabolic rationale'],[7,31534516,'','Mitochondrial metabolism and cisplatin resistance in tongue SCC','OSCC/TSCC metabolic rationale'],[8,38164660,'10.1002/hed.27620','ATP7B and extracellular-vesicle cisplatin resistance in HNSCC','Direct disease-context resistance biology']
],columns=['citation_order','pmid','doi','title','supported_use']);refs.to_csv(TAB/'verified_candidate_references_v1.tsv',sep='\t',index=False)

sns.set_theme(style='white',font_scale=.86);fig,axs=plt.subplots(2,3,figsize=(14.2,9.2),constrained_layout=True)
def panel(ax,l,t):ax.text(-.12,1.07,l,transform=ax.transAxes,fontweight='bold',fontsize=14,va='top');ax.set_title(t,loc='left',fontweight='bold',fontsize=10)
ax=axs[0,0];panel(ax,'a','Evidence-gated candidate funnel');ax.axis('off');items=[('1,750','LINCS compounds'),('328','Robust state-5'),('43','Orthogonal support'),('2','PRISM support'),('1','Three-screen hit')];
for i,(n,t) in enumerate(items):
 y=.92-i*.18;w=.92-i*.14;ax.add_patch(plt.Rectangle(((1-w)/2,y-.11),w,.13,color=sns.color_palette('Blues',6)[i+1],alpha=.9));ax.text(.5,y-.045,f'{n}  {t}',ha='center',va='center',color='white' if i>1 else 'black',fontweight='bold',fontsize=8)
ax=axs[0,1];panel(ax,'b','Quantitative evidence for shortlisted drugs');show=d.head(10);x=np.arange(len(show));ax.scatter(x,show.state5_percentile,label='State-5 percentile',s=35);ax.scatter(x,1-show.frozen_axis_rank/1750,label='Frozen-axis percentile',s=35);ax.scatter(x,np.where(show.PRISM_supported,np.clip(-show.prism_uadt_median_LFC,0,1),np.nan),label='PRISM depletion',s=35);ax.set_xticks(x,show.pert_iname,rotation=65,ha='right',fontsize=6);ax.set_ylim(0,1.05);ax.set_ylabel('Evidence metric (0–1)');ax.legend(frameon=False,fontsize=6)
ax=axs[0,2];panel(ax,'c','Independent evidence matrix');em=show.set_index('pert_iname')[['identity_resolved','state5_robust','dual_transcriptomic','PRISM_depletion','direct_OSCC_cisplatin_evidence']].astype(float);sns.heatmap(em,cmap=sns.color_palette(['#F2F2F2','#228833'],as_cmap=True),vmin=0,vmax=1,cbar=False,linewidths=.5,ax=ax);ax.set_xticklabels(['Identity','State-5','Frozen axis','PRISM','Direct OSCC'],rotation=25,ha='right');ax.set_ylabel('')
ax=axs[1,0];panel(ax,'d','Cross-cell-line state-5 reversal');z=show.sort_values('state5_reversal');y=np.arange(len(z));ax.hlines(y,z.q25,z.q75,color='#999999');ax.scatter(z.state5_reversal,y,c=np.where(z.PRISM_supported,'#D55E00','#0072B2'),s=30);ax.axvline(0,c='k',lw=.7);ax.set_yticks(y,z.pert_iname,fontsize=7);ax.set_xlabel('LINCS median reversal (IQR)')
ax=axs[1,1];panel(ax,'e','Program–mechanism–candidate evidence network');ax.axis('off')
# Three-level alluvial layout. Links denote verified mechanism/rationale, not efficacy.
program_nodes=[('Mitochondrial\nmetabolism',.88),('UPR / stress',.66),('Drug transport',.44),('MAPK signaling',.22)]
mechanism_nodes=[('PDH / TCA',.88),('PKC / ABCG2*',.62),('DHODH',.42),('MEK1/2',.20)]
drug_nodes=[('CPI-613',.90),('Bisindolylmaleimide*',.70),('Teriflunomide',.49),('Selumetinib',.28),('Trametinib',.10)]
for label,y0 in program_nodes:ax.text(.06,y0,label,ha='center',va='center',fontsize=7.5,fontweight='bold',bbox=dict(boxstyle='round,pad=.25',fc='#56B4E9',ec='none',alpha=.85))
for label,y0 in mechanism_nodes:ax.text(.50,y0,label,ha='center',va='center',fontsize=7.3,fontweight='bold',bbox=dict(boxstyle='round,pad=.25',fc='#CC79A7',ec='none',alpha=.78))
for label,y0 in drug_nodes:ax.text(.94,y0,label,ha='center',va='center',fontsize=7.2,fontweight='bold',bbox=dict(boxstyle='round,pad=.25',fc='#E69F00',ec='none',alpha=.82))
links1=[(.88,.88,.95),(.66,.88,.45),(.66,.62,.35),(.44,.62,.85),(.66,.42,.35),(.22,.20,.95)]
links2=[(.88,.90,.95),(.62,.70,.85),(.42,.49,.72),(.20,.28,.80),(.20,.10,.80)]
def curve(x0,y0,x1,y1,w,color):
 path=MPath([(x0,y0),(x0+.14,y0),(x1-.14,y1),(x1,y1)],[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4]);ax.add_patch(PathPatch(path,facecolor='none',edgecolor=color,lw=1.5+5*w,alpha=.18+.48*w,capstyle='round'))
for y0,y1,w in links1:curve(.16,y0,.40,y1,w,'#7A5195')
for y0,y1,w in links2:curve(.60,y0,.84,y1,w,'#7A5195')
ax.text(.06,.98,'Resistance program',ha='center',fontsize=7,color='#555555');ax.text(.50,.98,'Putative mechanism',ha='center',fontsize=7,color='#555555');ax.text(.94,.98,'Predicted candidate',ha='center',fontsize=7,color='#555555')
ax.text(.50,.015,'*BRD-K31342827 exact structure unresolved; class-level links only',ha='center',fontsize=6.2,color='#555555')

ax=axs[1,2];panel(ax,'f','Ranked multi-source candidate evidence')
# Dense row-wise display: stacked bars are screen contributions; diamonds show
# leave-one-sample-out stability without occupying a separate empty axis region.
plot=d.copy();plot['frozen_pct']=1-plot.frozen_axis_rank.fillna(1750)/1750;plot['prism_scaled']=np.where(plot.PRISM_supported,np.clip(-plot.prism_uadt_median_LFC,0,1),0)
plot['priority_total']=plot.state5_percentile+plot.frozen_pct+plot.prism_scaled
plot=plot.sort_values('priority_total',ascending=True);y=np.arange(len(plot));left=np.zeros(len(plot))
for col,label,color in [('state5_percentile','State-5 reversal','#0072B2'),('frozen_pct','Frozen-axis reversal','#E69F00'),('prism_scaled','PRISM depletion','#009E73')]:
 vals=plot[col].to_numpy();ax.barh(y,vals,left=left,height=.62,color=color,label=label,edgecolor='white',linewidth=.3);left+=vals
# Stability is displayed on the same 0–3 scale, positioned in a narrow right column.
stability_x=2.75+.22*(plot.LOSO_min_percentile.to_numpy()-.75)/.25
ax.scatter(stability_x,y,marker='D',s=20,c='#6A3D9A',label='LOSO stability',zorder=4)
for pos,r in enumerate(plot.itertuples()):
 if not r.identity_resolved:ax.scatter(3.08,pos,marker='X',s=32,c='#D55E00',linewidth=1.2,zorder=5)
ax.axvline(2.68,c='#999999',lw=.7,ls='--');ax.text(2.86,len(plot)-.05,'Stability',ha='center',va='bottom',fontsize=6.5,color='#6A3D9A',fontweight='bold')
ax.set_yticks(y,[x.replace('bisindolylmaleimide','bisindolylmaleimide*') for x in plot.pert_iname],fontsize=7.1);ax.set_xlim(0,3.18);ax.set_xlabel('Cumulative normalized evidence');ax.set_ylabel('')
ax.set_xticks([0,1,2,3]);ax.grid(axis='x',color='#DDDDDD',lw=.5);ax.legend(frameon=False,fontsize=6.2,ncol=2,loc='lower right')
ax.text(.01,.01,'*Exact identity unresolved; orange X at right',transform=ax.transAxes,fontsize=5.8,color='#555555')
fig.suptitle('Evidence audit separates computational convergence from validated sensitization',fontweight='bold',fontsize=15)
fig.savefig(FIG/'Figure_candidate_audit_v1.png',dpi=300,bbox_inches='tight');fig.savefig(FIG/'Figure_candidate_audit_v1.pdf',bbox_inches='tight');fig.savefig(FIG/'Figure_candidate_audit_v1.tif',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
summary={'recommended_lead':'CPI-613 (devimistat): strongest state-specific and frozen-axis reversal, but no direct OSCC or current PRISM validation','identity_hold':'BRD-K31342827: do not assign a specific bisindolylmaleimide structure or mechanism','secondary':'picolinic acid: weak state-5/PRISM hypothesis','directly_validated_sensitizers':0,'claim':'Computationally prioritized candidates for experimental cisplatin-combination testing.'};(META/'summary_v1.json').write_text(json.dumps(summary,indent=2)+'\n')
