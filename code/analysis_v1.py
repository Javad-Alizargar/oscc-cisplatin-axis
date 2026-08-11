"""Colab-ready JDS strengthening analysis for the OSCC cisplatin project.

This script reads the frozen analysis-ready objects in Google Drive, recomputes
publication summaries at the patient/sample level, and exports five six-panel
figures plus one dataset table. It never interprets cross-sectional maps as
directional transitions and never merges expression matrices across studies.
"""
from pathlib import Path
import hashlib, json, math, re, shutil, sys, subprocess, warnings

def ensure(packages):
    missing=[]
    for import_name,pip_name in packages:
        try: __import__(import_name)
        except ImportError: missing.append(pip_name)
    if missing: subprocess.check_call([sys.executable,"-m","pip","install","-q",*missing])

ensure([("pandas","pandas"),("numpy","numpy"),("scipy","scipy"),("matplotlib","matplotlib"),
        ("seaborn","seaborn"),("anndata","anndata"),("networkx","networkx"),("lifelines","lifelines")])
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, anndata as ad, networkx as nx
from scipy import stats, sparse
from matplotlib.patches import FancyBboxPatch, Circle

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,
                     "axes.labelsize":9,"figure.titlesize":14,"pdf.fonttype":42,"ps.fonttype":42})
SEED=20260810
rng=np.random.default_rng(SEED)
ROOT=Path("/content/drive/MyDrive/OSCC_Cisplatin_PhysicsInformed_StateTransitions")
if not ROOT.exists():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception: pass
assert ROOT.exists(), f"Project not found: {ROOT}"
OUT=Path("/content/jds_out_v6")
FIG=OUT/"figures"; TAB=OUT/"tables"; SUP=OUT/"supplement"
for p in (FIG,TAB,SUP): p.mkdir(parents=True,exist_ok=True)

def read_table(rel):
    p=ROOT/rel
    if not p.exists(): raise FileNotFoundError(p)
    return pd.read_csv(p,sep="\t" if ".tsv" in p.name else ",",compression="infer")

def clean_gene(value):
    text=str(value).strip().upper()
    return text if text and text not in {"NAN","NA","NONE"} and re.fullmatch(r"[A-Z0-9][A-Z0-9._-]*",text) else ""

def feature_mapping(obj,tcga=False):
    raw=obj.var["gene_name"].astype(str) if tcga else obj.var_names.astype(str)
    symbols=np.asarray([clean_gene(x) for x in raw],dtype=object)
    mapping={}
    for i,g in enumerate(symbols):
        if g: mapping.setdefault(g,[]).append(i)
    return symbols,mapping

def inspect_symbols(path,tcga=False):
    a=ad.read_h5ad(path,backed="r")
    try: return set(feature_mapping(a,tcga)[1])
    finally: a.file.close()

def exact_common_universe_scores(path,tcga,layer,fixed_up,fixed_down):
    """Reproduce Block-13 within-sample ranks and common-universe score."""
    a=ad.read_h5ad(path)
    try:
        symbols,mapping=feature_mapping(a,tcga)
        matrix=a.layers[layer] if layer else a.X
        if sparse.issparse(matrix): matrix=matrix.toarray()
        matrix=np.asarray(matrix,dtype=np.float32)
        valid=np.flatnonzero(np.asarray([bool(x) for x in symbols]))
        valid_pos={int(original):pos for pos,original in enumerate(valid)}
        values=matrix[:,valid]
        ranks=np.empty_like(values,dtype=np.float32)
        for i in range(values.shape[0]): ranks[i]=(stats.rankdata(values[i],method="average")/values.shape[1]).astype(np.float32)
        def gene_mean(g):
            pos=[valid_pos[i] for i in mapping[g] if i in valid_pos]
            return ranks[:,pos].mean(axis=1)
        up=np.column_stack([gene_mean(g) for g in fixed_up])
        down=np.column_stack([gene_mean(g) for g in fixed_down])
        score=up.mean(axis=1,dtype=np.float64)-down.mean(axis=1,dtype=np.float64)
        return pd.DataFrame({"sample_id":a.obs_names.astype(str),"common_universe_score":score})
    finally: del a

def panel(ax, letter, title):
    ax.text(-.08,1.08,letter,transform=ax.transAxes,fontsize=13,fontweight="bold",va="top")
    ax.set_title(title,fontweight="bold",pad=8)

def save(fig,name):
    fig.savefig(FIG/f"{name}.png",dpi=300,bbox_inches="tight",facecolor="white")
    fig.savefig(FIG/f"{name}.tif",dpi=600,bbox_inches="tight",facecolor="white",pil_kwargs={"compression":"tiff_lzw"})
    fig.savefig(FIG/f"{name}.pdf",bbox_inches="tight",facecolor="white")
    plt.close(fig)

def cliff_delta(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    return (np.greater(x[:,None],y).sum()-np.less(x[:,None],y).sum())/(len(x)*len(y))

def bootstrap_delta(x,y,B=5000):
    x=np.asarray(x,float); y=np.asarray(y,float)
    vals=np.empty(B)
    for b in range(B): vals[b]=cliff_delta(rng.choice(x,len(x),True),rng.choice(y,len(y),True))
    return np.quantile(vals,[.025,.975])

# ---------- Frozen inputs ----------
sig=read_table("03_analysis_ready/signatures/cisplatin_resistance_consensus_signature.tsv.gz")
robust=read_table("03_analysis_ready/robustness_enrichment_figures/tables/robust_LINCS_candidates.tsv")
prism=read_table("03_analysis_ready/robustness_enrichment_figures/tables/PRISM_supported_candidates.tsv")
integ=read_table("03_analysis_ready/robustness_enrichment_figures/tables/integrated_candidate_robustness_annotations.tsv.gz")
nulls=read_table("03_analysis_ready/clinical_signature_specificity_audit/tables/expression_matched_random_signature_null_replicates.tsv.gz")
nullsum=read_table("03_analysis_ready/clinical_signature_specificity_audit/tables/expression_matched_random_signature_null_summary.tsv")
direction=read_table("03_analysis_ready/state_direction_null_audit/tables/state_direction_permutation_null_summary.tsv")
cox=read_table("03_analysis_ready/independent_clinical_validation/tables/overall_and_cause_specific_Cox_models.tsv")
tcga=read_table("03_analysis_ready/independent_clinical_validation/tables/TCGA_strict_oral_fixed_resistance_scores.tsv.gz")
tcga_exp=read_table("03_analysis_ready/independent_clinical_validation/tables/TCGA_expanded_oral_fixed_resistance_scores.tsv.gz")
gse=read_table("03_analysis_ready/independent_clinical_validation/tables/GSE41613_fixed_resistance_scores.tsv.gz")

# Reconstruct the exact Block-13 common-platform score used by the
# expression-matched random-signature audit. The full-platform score retained
# in the clinical tables is a related descriptive score but must not be mixed
# with this null analysis.
strict_h5=ROOT/"03_analysis_ready/h5ad/TCGA_HNSC_strict_oral_STAR_counts_TPM.h5ad"
expanded_h5=ROOT/"03_analysis_ready/h5ad/TCGA_HNSC_expanded_oral_STAR_counts_TPM.h5ad"
gse_h5=ROOT/"03_analysis_ready/h5ad/GSE41613_gene_level_submitted_expression.h5ad"
discovery={clean_gene(x) for x in sig["gene_symbol"]};discovery.discard("")
common=discovery & inspect_symbols(strict_h5,True) & inspect_symbols(expanded_h5,True) & inspect_symbols(gse_h5,False)
selected=sig["selected_for_LINCS_query"].astype(str).str.lower().isin(["true","1"])
fixed_up=[clean_gene(x) for x in sig.loc[selected & sig["consensus_signed_rank_score"].gt(0),"gene_symbol"] if clean_gene(x) in common]
fixed_down=[clean_gene(x) for x in sig.loc[selected & sig["consensus_signed_rank_score"].lt(0),"gene_symbol"] if clean_gene(x) in common]
if (len(common),len(fixed_up),len(fixed_down)) != (15038,140,131):
    raise ValueError(f"Unexpected common signature dimensions: common={len(common)}, UP={len(fixed_up)}, DOWN={len(fixed_down)}")
strict_common=exact_common_universe_scores(strict_h5,True,"TPM",fixed_up,fixed_down)
expanded_common=exact_common_universe_scores(expanded_h5,True,"TPM",fixed_up,fixed_down)
gse_common=exact_common_universe_scores(gse_h5,False,None,fixed_up,fixed_down)
def attach_common(meta,scores):
    out=meta.copy();out["sample_id"]=out["sample_id"].astype(str)
    out=out.merge(scores,on="sample_id",how="left",validate="one_to_one")
    if out["common_universe_score"].isna().any(): raise ValueError("Common-universe score alignment failed")
    return out
tcga=attach_common(tcga,strict_common);tcga_exp=attach_common(tcga_exp,expanded_common);gse=attach_common(gse,gse_common)

# ---------- Table 1 ----------
dataset_rows=[
 ["GSE117872","scRNA-seq","2 paired origins; 1,302 cells","Discovery","Resistant-parental paired effects","Hypothesis-generating; cells not independent"],
 ["GSE168424","Microarray","3 SAS monolayer + 3 SAS sphere","Independent discovery","Sphere-monolayer effects","Stem-like contrast; small n"],
 ["GSE103322","Smart-seq2","18 patients; 5,902 cells","Single-cell mapping","Frozen-axis projection","140 conservative malignant anchors"],
 ["GSE172577","10x scRNA-seq","6 samples; 58,999 QC cells","Single-cell mapping","Frozen-axis projection","1,372 conservative malignant anchors"],
 ["GSE215403","10x scRNA-seq","12 samples; 47,094 QC cells","Single-cell mapping","Frozen-axis projection","2,032 conservative malignant anchors"],
 ["TCGA-HNSC strict oral","RNA-seq","99 cases; 102 samples","Primary clinical projection","Stage and overall survival","Strict oral-site adjudication"],
 ["GSE41613","Microarray","97 HPV-negative OSCC patients","Primary clinical projection","Stage and overall survival","Independent primary cohort"],
 ["TCGA-HNSC expanded","RNA-seq","308 cases; 338 samples","Sensitivity analysis","Stage and paired tissue","Overlaps strict TCGA"],
 ["LINCS GSE70138","L1000","118,050 signatures","Drug hypothesis generation","Signature reversal","Not efficacy or resensitization"],
 ["PRISM 23Q2","Single-dose screen","Compound-cell-line measurements","Orthogonal support","Depletion","Not dose response or synergy"],
]
table1=pd.DataFrame(dataset_rows,columns=["Dataset","Platform","Independent units","Role","Endpoint","Interpretive boundary"])
table1.to_csv(TAB/"Table_1_dataset_and_validation_architecture.tsv",sep="\t",index=False)

# ---------- Figure 1: design and evidence hierarchy ----------
fig,axs=plt.subplots(2,3,figsize=(14,8.5),constrained_layout=True)
colors=["#4477AA","#66CCEE","#228833","#CCBB44","#EE6677"]
ax=axs[0,0]; panel(ax,"a","Leakage-resistant analysis sequence"); ax.axis("off")
steps=[("Discovery\n2 datasets",.01),("Freeze\n300 genes",.21),("Single-cell\n3 cohorts",.41),("Drug\nhypotheses\n2 resources",.61),("Clinical\ntests\n2 cohorts",.81)]
for i,(txt,x) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((x,.34),.14,.32,boxstyle="round,pad=.015",fc=colors[i],alpha=.18,ec=colors[i],lw=1.5))
    ax.text(x+.07,.50,txt,ha="center",va="center",fontsize=7,fontweight="bold")
    if i<4: ax.annotate("",(x+.195,.50),(x+.155,.50),arrowprops=dict(arrowstyle="->",lw=1.2))
ax.set_xlim(0,1);ax.set_ylim(0,1)
ax=axs[0,1];panel(ax,"b","Independent units, not cell counts")
labs=["Discovery\norigins/replicates","Single-cell\npatients/samples","Clinical\npatients","Drug\nsignatures"]
vals=[5,36,196,118050]; disp=np.log10(vals)
ax.bar(labs,disp,color=colors[:4]);ax.set_ylabel("log10 independent units/signatures")
for i,(v,d) in enumerate(zip(vals,disp)): ax.text(i,d+.08,f"{v:,}",ha="center",fontsize=8)
ax=axs[0,2];panel(ax,"c","Frozen-axis construction gate")
tiers=sig["signature_tier"].fillna("Unclassified").value_counts().head(5)
ax.barh(tiers.index[::-1],tiers.values[::-1],color=sns.color_palette("colorblind",len(tiers)));ax.set_xlabel("Genes")
ax=axs[1,0];panel(ax,"d","Single-cell mapping cohorts")
names=["GSE103322","GSE172577","GSE215403"]; cells=[140,1372,2032]
ax.bar(names,cells,color=["#009E73","#56B4E9","#0072B2"]);ax.set_ylabel("Conservative malignant anchors")
for i,v in enumerate(cells): ax.text(i,v+45,f"{v:,}",ha="center")
ax=axs[1,1];panel(ax,"e","Prespecified clinical hierarchy");ax.axis("off")
items=[("Primary","Mean stage Cliff delta:\nstrict TCGA + GSE41613","#228833"),("Secondary","Cohort-specific matched-null tests","#CCBB44"),("Sensitivity","Expanded TCGA; paired tissue","#EE6677"),("Negative","Overall-survival meta-analysis","#777777")]
for i,(a,b,c) in enumerate(items):
    y=.83-i*.22;ax.add_patch(FancyBboxPatch((.05,y-.10),.90,.16,boxstyle="round,pad=.02",fc=c,alpha=.14,ec=c));ax.text(.10,y,a,fontweight="bold",color=c);ax.text(.34,y,b,va="center")
ax=axs[1,2];panel(ax,"f","Final claim gate");ax.axis("off")
claims=[("Supported","Cross-dataset axis; stage enrichment","#228833"),("Mapping only","Single-cell heterogeneity","#4477AA"),("Not supported","Direction, survival, drug convergence","#D55E00"),("Not tested","Efficacy, synergy, resensitization","#777777")]
for i,(a,b,c) in enumerate(claims):
    y=.83-i*.22;ax.text(.04,y,"●",color=c,fontsize=18,va="center");ax.text(.12,y,a,fontweight="bold",color=c);ax.text(.12,y-.08,b,fontsize=8)
fig.suptitle("Study architecture and evidence boundaries",fontweight="bold");save(fig,"Figure_1_study_architecture")

# ---------- Figure 2: axis construction ----------
fig,axs=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
ax=axs[0,0];panel(ax,"a","Independent discovery effects")
q=sig.dropna(subset=["GSE117872_signed_effect_rank","GSE168424_signed_effect_rank"])
sel=q["selected_for_LINCS_query"].astype(str).str.lower().isin(["true","1"])
ax.scatter(q.loc[~sel,"GSE117872_signed_effect_rank"],q.loc[~sel,"GSE168424_signed_effect_rank"],s=3,c="#BBBBBB",alpha=.25,rasterized=True)
ax.scatter(q.loc[sel,"GSE117872_signed_effect_rank"],q.loc[sel,"GSE168424_signed_effect_rank"],s=10,c=np.where(q.loc[sel,"consensus_signed_rank_score"].gt(0),"#D55E00","#0072B2"),alpha=.8)
rho=stats.spearmanr(q["GSE117872_signed_effect_rank"],q["GSE168424_signed_effect_rank"],nan_policy="omit").statistic
ax.text(.03,.96,f"Spearman ρ={rho:.3f}",transform=ax.transAxes,va="top");ax.axhline(0,c="k",lw=.5);ax.axvline(0,c="k",lw=.5);ax.set_xlabel("GSE117872 signed rank");ax.set_ylabel("GSE168424 signed rank")
ax=axs[0,1];panel(ax,"b","Consensus eligibility tiers")
tiers=sig["signature_tier"].fillna("Unclassified").value_counts();ax.barh(tiers.index,tiers.values,color=sns.color_palette("muted",len(tiers)));ax.invert_yaxis();ax.set_xlabel("Genes")
ax=axs[0,2];panel(ax,"c","Independent-unit bootstrap stability")
st=sig.loc[sig["direction_concordant_across_discovery_datasets"].astype(str).str.lower().isin(["true","1"]),"bootstrap_direction_stability"].dropna()
ax.hist(st,bins=np.linspace(.5,1,26),color="#4477AA",alpha=.85);ax.axvline(.8,ls="--",c="#D55E00",label="Eligibility threshold");ax.set_xlabel("Direction stability");ax.set_ylabel("Genes");ax.legend(frameon=False)
ax=axs[1,0];panel(ax,"d","Frozen query extremes")
f=sig[sel].copy().sort_values("consensus_signed_rank_score");show=pd.concat([f.head(8),f.tail(8)])
ax.barh(show["gene_symbol"],show["consensus_signed_rank_score"],color=np.where(show["consensus_signed_rank_score"]>0,"#D55E00","#0072B2"));ax.axvline(0,c="k",lw=.7);ax.set_xlabel("Consensus signed-rank score")
ax=axs[1,1];panel(ax,"e","Effect concordance of frozen genes")
up=f[f["consensus_signed_rank_score"].gt(0)];dn=f[f["consensus_signed_rank_score"].lt(0)]
for d,c,l in [(up,"#D55E00","UP"),(dn,"#0072B2","DOWN")]: ax.scatter(d["GSE117872_effect_resistant_minus_sensitive"],d["GSE168424_effect_sphere_minus_SAS"],s=14,alpha=.55,c=c,label=l)
ax.axhline(0,c="k",lw=.5);ax.axvline(0,c="k",lw=.5);ax.set_xlabel("GSE117872 effect");ax.set_ylabel("GSE168424 effect");ax.legend(frameon=False)
ax=axs[1,2];panel(ax,"f","Selection audit")
counts=pd.Series({"Common genes":len(sig),"Direction-concordant":int(sig["direction_concordant_across_discovery_datasets"].astype(str).str.lower().isin(["true","1"]).sum()),"Stable concordant":int(((sig["bootstrap_direction_stability"]>=.8)&sig["direction_concordant_across_discovery_datasets"].astype(str).str.lower().isin(["true","1"])).sum()),"Frozen query":len(f)})
ax.plot(range(len(counts)),counts.values,marker="o",lw=2,color="#009E73");ax.set_xticks(range(len(counts)),counts.index,rotation=25,ha="right");ax.set_ylabel("Genes");ax.set_yscale("log")
for i,v in enumerate(counts):ax.text(i,v*1.15,f"{v:,}",ha="center",fontsize=8)
fig.suptitle("Cross-dataset construction of the fixed cisplatin-resistance-associated axis",fontweight="bold");save(fig,"Figure_2_axis_construction")

# ---------- Figure 3: non-directional single-cell mapping ----------
state_specs=[("GSE103322","patient_id","03_analysis_ready/directed_state_potential/GSE103322_malignant_core_directed_state.h5ad"),("GSE172577","sample_id","03_analysis_ready/directed_state_potential/GSE172577_malignant_core_directed_state.h5ad"),("GSE215403","sample_id","03_analysis_ready/directed_state_potential/GSE215403_malignant_core_directed_state.h5ad")]
states=[]
fig,axs=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
for j,(name,sample_col,rel) in enumerate(state_specs):
    a=ad.read_h5ad(ROOT/rel,backed="r"); coords=np.asarray(a.obsm["X_state_pca"][:,:2]); obs=a.obs[[sample_col,"resistance_axis_sample_z"]].copy();obs["dataset"]=name;obs["sample"]=obs[sample_col].astype(str);states.append(obs[["dataset","sample","resistance_axis_sample_z"]]);a.file.close()
    ax=axs[0,j];panel(ax,chr(97+j),f"{name}: separate patient-balanced map")
    sc=ax.scatter(coords[:,0],coords[:,1],c=obs["resistance_axis_sample_z"],s=7,cmap="coolwarm",vmin=-2,vmax=2,alpha=.7,rasterized=True);ax.set_xlabel("PC1");ax.set_ylabel("PC2");ax.text(.02,.02,f"n={len(obs):,}",transform=ax.transAxes)
allstate=pd.concat(states,ignore_index=True)
ax=axs[1,0];panel(ax,"d","Patient/sample-level axis distribution")
sm=allstate.groupby(["dataset","sample"],as_index=False)["resistance_axis_sample_z"].median()
sns.stripplot(data=sm,x="dataset",y="resistance_axis_sample_z",hue="dataset",palette="colorblind",legend=False,ax=ax,jitter=.15);ax.axhline(0,c="k",lw=.7,ls="--");ax.set_ylabel("Median within-sample z score");ax.set_xlabel("")
ax=axs[1,1];panel(ax,"e","Within-sample heterogeneity")
het=allstate.groupby(["dataset","sample"])["resistance_axis_sample_z"].agg(lambda x:np.quantile(x,.9)-np.quantile(x,.1)).reset_index(name="P90-P10")
sns.boxplot(data=het,x="dataset",y="P90-P10",hue="dataset",palette="colorblind",legend=False,ax=ax);sns.stripplot(data=het,x="dataset",y="P90-P10",color="black",size=3,ax=ax);ax.set_xlabel("");ax.set_ylabel("Within-sample P90-P10")
ax=axs[1,2];panel(ax,"f","Corrective direction-null audit")
ds=direction[(direction["alpha"].isin([0,.5]))&(direction["metric"].eq("median_of_patient_median_drift"))].copy(); y=np.arange(len(ds));
# The permutation interval describes the null distribution and is not a CI
# centered on the observation. Draw it independently so observations outside
# the interval do not create invalid negative error-bar lengths.
ax.hlines(y,ds["null_q025"],ds["null_q975"],color="#999999",lw=3,label="Permutation-null 95% interval")
ax.scatter(ds["null_median"],y,marker="|",s=90,color="black",label="Null median",zorder=3)
ax.scatter(ds["observed"],y,s=34,color="#D55E00",label="Observed",zorder=4)
ax.axvline(0,c="k",lw=.7);ax.set_yticks(y,[f"{r.dataset}; α={r.alpha:g}" for r in ds.itertuples()]);ax.set_xlabel("Direction statistic");ax.legend(frameon=False,fontsize=7,loc="upper left");ax.text(.98,.98,"0/3 datasets passed corrected tests",transform=ax.transAxes,ha="right",va="top",color="#D55E00",fontweight="bold",fontsize=8)
sm.to_csv(TAB/"single_cell_sample_medians.tsv",sep="\t",index=False);het.to_csv(TAB/"single_cell_within_sample_heterogeneity.tsv",sep="\t",index=False)
fig.suptitle("Non-directional single-cell projection and corrective null analysis",fontweight="bold");save(fig,"Figure_3_single_cell_mapping")

# ---------- Figure 4: drug hypotheses ----------
fig,axs=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
ax=axs[0,0];panel(ax,"a","Robust LINCS reversal hypotheses")
r=robust.sort_values("lincs_median_reversal").tail(12);ax.barh(r["pert_iname"],r["lincs_median_reversal"],color="#4477AA");ax.set_xlabel("Median LINCS reversal score")
ax=axs[0,1];panel(ax,"b","Cross-context robustness")
rr=robust.set_index("pert_iname");metrics=["lincs_fraction_cell_lines_positive","QC_top100_fraction","query_deletion_top100_frequency","LOPO_candidate_set_min_percentile"]
z=rr[metrics].copy();z.columns=["Cell-line\npositive","QC\nretention","Query-deletion\nretention","Leave-one-origin\nminimum"]
sns.heatmap(z,cmap="YlGnBu",vmin=0,vmax=1,ax=ax,cbar_kws={"label":"Robustness fraction/percentile"});ax.set_xlabel("");ax.set_ylabel("")
ax=axs[0,2];panel(ax,"c","Compound-to-mechanism network")
G=nx.Graph(); top=robust.head(10)
for row in top.itertuples():
    drug=str(row.pert_iname); moas=[x.strip() for x in str(row.moa).split("|") if x.strip() and x!="nan"][:2] or ["Unannotated"]
    G.add_node(drug,kind="drug")
    for m in moas:G.add_node(m,kind="moa");G.add_edge(drug,m)
drugs=[n for n,d in G.nodes(data=True) if d["kind"]=="drug"];moas=[n for n,d in G.nodes(data=True) if d["kind"]=="moa"]
pos={n:(0,i/max(1,len(drugs)-1)) for i,n in enumerate(drugs)}|{n:(1,i/max(1,len(moas)-1)) for i,n in enumerate(moas)}
nx.draw_networkx_edges(G,pos,ax=ax,alpha=.35);nx.draw_networkx_nodes(G,pos,nodelist=drugs,node_color="#4477AA",node_size=160,ax=ax);nx.draw_networkx_nodes(G,pos,nodelist=moas,node_color="#EE6677",node_size=190,ax=ax);nx.draw_networkx_labels(G,pos,font_size=6,ax=ax);ax.axis("off")
ax.set_xlim(-.38,1.38);ax.set_ylim(-.08,1.08)
ax=axs[1,0];panel(ax,"d","PRISM single-dose depletion support")
p=prism.sort_values("prism_uadt_median_LFC").head(11);ax.barh(p["pert_iname"],p["prism_uadt_median_LFC"],color="#009E73");ax.axvline(0,c="k",lw=.7);ax.set_xlabel("Median UADT-lineage log-fold change")
ax=axs[1,1];panel(ax,"e","LINCS and PRISM answer different questions")
ax.axis("off");ax.add_patch(Circle((.38,.52),.28,fc="#4477AA",alpha=.25,ec="#4477AA",lw=2));ax.add_patch(Circle((.62,.52),.28,fc="#009E73",alpha=.25,ec="#009E73",lw=2));ax.text(.25,.52,"12 robust\nLINCS",ha="center",va="center",fontweight="bold");ax.text(.75,.52,"11 PRISM\nsupported",ha="center",va="center",fontweight="bold");ax.text(.50,.52,"0\noverlap",ha="center",va="center",fontweight="bold",color="#D55E00",fontsize=12)
ax=axs[1,2];panel(ax,"f","Evidence boundary")
ax.axis("off");lines=[("Supported","Perturbational reversal hypotheses"),("Orthogonal","Single-dose depletion hypotheses"),("Not supported","Convergence, dose response, synergy"),("Not tested","Cisplatin resensitization or efficacy")]
for i,(a,b) in enumerate(lines):y=.83-i*.22;ax.text(.04,y,a,fontweight="bold",color=["#4477AA","#009E73","#D55E00","#777777"][i],fontsize=8);ax.text(.43,y,b,fontsize=8,wrap=True)
robust.to_csv(TAB/"robust_LINCS_candidates.tsv",sep="\t",index=False);prism.to_csv(TAB/"PRISM_supported_candidates.tsv",sep="\t",index=False)
fig.suptitle("Robustness and evidence boundaries of computational drug hypotheses",fontweight="bold");save(fig,"Figure_4_drug_hypotheses")

# ---------- Figure 5: clinical projection ----------
def primary(df):
    if "is_unique_primary_tumor" in df: return df[df["is_unique_primary_tumor"].astype(str).str.lower().isin(["true","1"])]
    return df
tc=primary(tcga); te=primary(tcga_exp); gs=gse.copy()
cohorts=[("Strict TCGA",tc),("GSE41613",gs),("Expanded TCGA",te)]
effects=[]
for name,d in cohorts:
    a=d.loc[d.stage_group.eq("III_IV"),"common_universe_score"].dropna();b=d.loc[d.stage_group.eq("I_II"),"common_universe_score"].dropna()
    if len(a)==0 or len(b)==0: raise ValueError(f"Empty stage group in {name}: advanced={len(a)}, early={len(b)}, labels={sorted(d.stage_group.dropna().astype(str).unique())}")
    ci=bootstrap_delta(a,b);effects.append([name,cliff_delta(a,b),ci[0],ci[1],len(a),len(b)])
effects=pd.DataFrame(effects,columns=["cohort","cliff_delta","ci_low","ci_high","n_advanced","n_early"]);effects.to_csv(TAB/"stage_effects_bootstrap_CI.tsv",sep="\t",index=False)
fig,axs=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
for j,(name,d) in enumerate(cohorts[:2]):
    ax=axs[0,j];panel(ax,chr(97+j),f"{name}: fixed score by stage")
    dd=d[d.stage_group.isin(["I_II","III_IV"])].copy();dd["stage_display"]=dd.stage_group.map({"I_II":"I/II","III_IV":"III/IV"})
    sns.violinplot(data=dd,x="stage_display",y="common_universe_score",inner=None,color="#88CCEE",ax=ax);sns.boxplot(data=dd,x="stage_display",y="common_universe_score",width=.25,showfliers=False,boxprops={"facecolor":"white"},ax=ax);sns.stripplot(data=dd,x="stage_display",y="common_universe_score",color="black",alpha=.35,size=2,ax=ax);ax.set_xlabel("Stage");ax.set_ylabel("Common-universe fixed-axis score")
ax=axs[0,2];panel(ax,"c","Expression-matched signature null")
metric="primary_independent_mean_stage_Cliff_delta";obs=float(nullsum.loc[nullsum.metric.eq(metric),"observed"].iloc[0]);emp=float(nullsum.loc[nullsum.metric.eq(metric),"empirical_p"].iloc[0])
ax.hist(nulls[metric],bins=35,color="#AA99CC",alpha=.85);ax.axvline(obs,c="#D55E00",lw=2,label=f"Observed={obs:.3f}");ax.axvline(nulls[metric].median(),c="k",ls="--",label="Null median");ax.set_xlabel("Mean stage Cliff delta");ax.set_ylabel("Matched random signatures");ax.legend(frameon=False,title=f"Empirical P={emp:.3f}")
ax=axs[1,0];panel(ax,"d","Stage-effect forest plot")
y=np.arange(len(effects));ax.errorbar(effects.cliff_delta,y,xerr=[effects.cliff_delta-effects.ci_low,effects.ci_high-effects.cliff_delta],fmt="o",color="#4477AA",capsize=4);ax.axvline(0,c="k",ls="--",lw=.8);ax.set_yticks(y,effects.cohort);ax.invert_yaxis();ax.set_xlabel("Cliff delta, advanced vs early stage (95% bootstrap CI)")
ax=axs[1,1];panel(ax,"e","Overall-survival effect estimates")
cc=cox[(cox.outcome.str.contains("overall",case=False,na=False))&(cox.covariate.str.contains("score",case=False,na=False))].drop_duplicates("cohort").copy();yy=np.arange(len(cc));ax.errorbar(cc.hazard_ratio,yy,xerr=[cc.hazard_ratio-cc.HR_CI95_low,cc.HR_CI95_high-cc.hazard_ratio],fmt="o",color="#D55E00",capsize=4);ax.axvline(1,c="k",ls="--");ax.set_yticks(yy,cc.cohort);ax.set_xscale("log");ax.set_xlabel("Hazard ratio per 1-SD score (95% CI)")
ax=axs[1,2];panel(ax,"f","Clinical claim audit");ax.axis("off")
claims=[("Supported","Primary cross-cohort stage statistic","#228833"),("Supported","GSE41613; expanded TCGA sensitivity","#228833"),("Not specific","Strict TCGA alone; paired tissue","#D55E00"),("Null","Overall-survival meta-analysis","#777777"),("Not established","Prediction or prognosis","#777777")]
for i,(a,b,c) in enumerate(claims):y=.88-i*.17;ax.text(.03,y,"●",color=c,fontsize=15,va="center");ax.text(.10,y,a,fontweight="bold",color=c,fontsize=8);ax.text(.43,y,b,fontsize=8,wrap=True)
fig.suptitle("Advanced-stage enrichment, specificity nulls, and clinical uncertainty",fontweight="bold");save(fig,"Figure_5_clinical_projection")

# ---------- Reproducibility manifest ----------
manifest=[]
for p in sorted(OUT.rglob("*")):
    if p.is_file(): manifest.append({"relative_path":str(p.relative_to(OUT)),"size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(manifest).to_csv(OUT/"SHA256_manifest.tsv",sep="\t",index=False)
(OUT/"README.txt").write_text("JDS strengthening outputs. All analyses use frozen public-data derivatives. Figures are exported as PNG (review), TIFF 600 dpi (submission), and vector PDF. Directional state-transition claims are not supported and are excluded.\n")
archive=Path("/content/jds_out_v6.zip")
if archive.exists(): archive.unlink()
shutil.make_archive(str(archive.with_suffix("")),"zip",OUT.parent,OUT.name)
print(f"COMPLETE: {archive} {archive.stat().st_size/1024/1024:.3f} MB")
try:
    from google.colab import files
    files.download(str(archive))
except Exception: pass
