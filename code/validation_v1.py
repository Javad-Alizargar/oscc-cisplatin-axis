"""Frozen-axis functional validation for the OSCC cisplatin project.

This Colab script adds orthogonal DepMap 24Q2 CRISPR/expression analyses and
cell-line-level PRISM association tests. It never changes the frozen Block-6
signature or the locked v6 clinical statistics.
"""
from pathlib import Path
import hashlib, json, re, shutil, subprocess, sys, warnings

def ensure(items):
    missing=[]
    for imp,pip in items:
        try: __import__(imp)
        except ImportError: missing.append(pip)
    if missing: subprocess.check_call([sys.executable,"-m","pip","install","-q",*missing])
ensure([("pandas","pandas"),("numpy","numpy"),("scipy","scipy"),("matplotlib","matplotlib"),
        ("seaborn","seaborn"),("requests","requests")])
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, requests
from scipy import stats
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore",category=FutureWarning)
SEED=20260811; rng=np.random.default_rng(SEED)
ROOT=Path("/content/drive/MyDrive/OSCC_Cisplatin_PhysicsInformed_StateTransitions")
if not ROOT.exists():
    from google.colab import drive
    drive.mount("/content/drive")
assert ROOT.exists(),f"Project not found: {ROOT}"
OUT=Path("/content/drug_val_out_v2"); FIG=OUT/"figures"; TAB=OUT/"tables"; META=OUT/"metadata"
for p in (FIG,TAB,META): p.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/"external_validation"/"DepMap24Q2"; CACHE.mkdir(parents=True,exist_ok=True)

DEPMAP={
 "Model.csv":("https://ndownloader.figshare.com/files/46489732",559182),
 "CRISPRGeneEffect.csv":("https://ndownloader.figshare.com/files/46489063",419330454),
 "OmicsExpressionProteinCodingGenesTPMLogp1.csv":("https://ndownloader.figshare.com/files/46490878",460868099),
}
def download(name,url,min_bytes):
    path=CACHE/name
    if path.exists() and path.stat().st_size>=min_bytes: return path
    tmp=path.with_suffix(path.suffix+".part")
    with requests.get(url,stream=True,timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
    if tmp.stat().st_size<min_bytes: raise ValueError(f"Truncated download: {name}")
    tmp.replace(path); return path
paths={n:download(n,*v) for n,v in DEPMAP.items()}

def read_tsv(rel): return pd.read_csv(ROOT/rel,sep="\t",compression="infer",low_memory=False)
def gene(x):
    s=str(x).strip().upper(); return re.sub(r"\s+\([0-9]+\)$","",s)
def bh(p):
    p=np.asarray(p,float); n=len(p); order=np.argsort(np.where(np.isfinite(p),p,np.inf)); q=np.full(n,np.nan); last=1.0
    for rank,i in reversed(list(enumerate(order,1))):
        if np.isfinite(p[i]): last=min(last,p[i]*n/rank);q[i]=last
    return q
def cliff(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)==0 or len(y)==0: return np.nan
    return (np.greater(x[:,None],y).sum()-np.less(x[:,None],y).sum())/(len(x)*len(y))
def boot_spearman(x,y,B=2000):
    x=np.asarray(x,float);y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    if len(x)<6:return np.nan,np.nan,np.nan
    rho=stats.spearmanr(x,y).statistic; vals=[]
    for _ in range(B):
        z=rng.integers(0,len(x),len(x)); r=stats.spearmanr(x[z],y[z]).statistic
        if np.isfinite(r):vals.append(r)
    lo,hi=np.quantile(vals,[.025,.975]);return rho,lo,hi

# Frozen signature and prespecified targets.
sig=read_tsv("03_analysis_ready/signatures/cisplatin_resistance_consensus_signature.tsv.gz")
sel=sig["selected_for_LINCS_query"].astype(str).str.lower().isin(["true","1"])
up=[gene(x) for x in sig.loc[sel & sig["consensus_signed_rank_score"].gt(0),"gene_symbol"]]
down=[gene(x) for x in sig.loc[sel & sig["consensus_signed_rank_score"].lt(0),"gene_symbol"]]
if (len(up),len(down))!=(150,150): raise ValueError(f"Frozen signature changed: {len(up)}/{len(down)}")
targets=["AKT1","AKT2","AKT3","PDPK1","JAK1","JAK2","BRAF","RAF1","TRPV1","ALOX5",
         "MASTL","ENSA","CMTM6","BASP1","SRC","ATP7B"]

# Read only needed columns from large DepMap matrices.
def selected_matrix(path,wanted):
    hdr=pd.read_csv(path,nrows=0).columns.tolist(); idcol=hdr[0]
    mapping={gene(c):c for c in hdr[1:]}; keep=[idcol]+[mapping[g] for g in wanted if g in mapping]
    d=pd.read_csv(path,usecols=keep,low_memory=False); d=d.rename(columns={idcol:"ModelID",**{c:gene(c) for c in keep[1:]}})
    return d,mapping
# Freeze a broad background-gene pool before reading expression. This permits
# signed random-gene-set nulls without loading the complete 460-MB matrix.
expr_header=pd.read_csv(paths["OmicsExpressionProteinCodingGenesTPMLogp1.csv"],nrows=0).columns.tolist()
header_genes=[gene(c) for c in expr_header[1:]]
eligible_bg=sorted(set(header_genes)-set(up)-set(down)-set(targets))
background_genes=list(rng.choice(eligible_bg,min(5000,len(eligible_bg)),replace=False))
expr,expr_map=selected_matrix(paths["OmicsExpressionProteinCodingGenesTPMLogp1.csv"],set(up+down+targets+background_genes))
dep,dep_map=selected_matrix(paths["CRISPRGeneEffect.csv"],set(targets))
model=pd.read_csv(paths["Model.csv"],low_memory=False)
id_candidates=[c for c in model.columns if c.lower() in {"modelid","depmap_id","model_id"}]
if not id_candidates: raise ValueError(f"Model ID column not found: {model.columns.tolist()}")
model=model.rename(columns={id_candidates[0]:"ModelID"}); model["ModelID"]=model["ModelID"].astype(str)
lineage_col=next((c for c in ["OncotreeLineage","OncotreePrimaryDisease","PrimaryDisease","lineage"] if c in model),None)
if lineage_col is None: raise ValueError("No lineage field in Model.csv")
model["uadt"]=model[lineage_col].fillna("").astype(str).str.upper().eq("HEAD AND NECK")
if int(model["uadt"].sum()) < 40:
    raise ValueError(f"Head-and-neck cohort recognition failed: n={int(model['uadt'].sum())}; labels={model[lineage_col].value_counts().head(20).to_dict()}")
name_col=next((c for c in ["CellLineName","CCLEName","ModelID"] if c in model),"ModelID")

# Gene-wise z-score axis avoids cross-platform absolute-expression assumptions.
available_up=[g for g in up if g in expr]; available_down=[g for g in down if g in expr]
if min(len(available_up),len(available_down))<120: raise ValueError("Insufficient frozen-gene mapping to DepMap")
X=expr.drop(columns="ModelID").apply(pd.to_numeric,errors="coerce")
Z=(X-X.mean())/X.std(ddof=0).replace(0,np.nan)
expr["depmap_axis_z"]=Z[available_up].mean(axis=1)-Z[available_down].mean(axis=1)
base=expr[["ModelID","depmap_axis_z"]].merge(model[["ModelID",name_col,lineage_col,"uadt"]],on="ModelID",how="left",validate="one_to_one")
base.to_csv(TAB/"depmap_axis_scores.tsv",sep="\t",index=False)

# Target dependency/selectivity and axis-dependency associations.
dd=base.merge(dep,on="ModelID",how="inner",validate="one_to_one")
rows=[]
for t in targets:
    if t not in dd: continue
    u=pd.to_numeric(dd.loc[dd.uadt,t],errors="coerce").dropna();o=pd.to_numeric(dd.loc[~dd.uadt,t],errors="coerce").dropna()
    sub=dd.loc[dd.uadt,["depmap_axis_z",t]].dropna();rho,lo,hi=boot_spearman(sub.depmap_axis_z,sub[t])
    psel=stats.mannwhitneyu(u,o,alternative="two-sided").pvalue if len(u)>2 and len(o)>2 else np.nan
    pcorr=stats.spearmanr(sub.depmap_axis_z,sub[t]).pvalue if len(sub)>5 else np.nan
    rows.append([t,len(u),len(o),u.median(),o.median(),cliff(u,o),psel,len(sub),rho,lo,hi,pcorr])
td=pd.DataFrame(rows,columns=["target","n_uadt","n_other","uadt_median_gene_effect","other_median_gene_effect","uadt_vs_other_cliff_delta","selectivity_p","n_axis_dependency","axis_dependency_rho","rho_ci_low","rho_ci_high","axis_dependency_p"])
td["selectivity_q"]=bh(td.selectivity_p);td["axis_dependency_q"]=bh(td.axis_dependency_p)

# Random signed-gene-set null for prespecified axis-dependency correlations.
pool=[g for g in background_genes if g in expr]
null_rows=[]; BNULL=500
for b in range(BNULL):
    pick=rng.choice(pool,len(available_up)+len(available_down),replace=False)
    score=Z[list(pick[:len(available_up)])].mean(axis=1)-Z[list(pick[len(available_up):])].mean(axis=1)
    temp=pd.DataFrame({"ModelID":expr.ModelID,"score":score}).merge(dep,on="ModelID").merge(model[["ModelID","uadt"]],on="ModelID")
    temp=temp[temp.uadt]
    for t in targets:
        if t in temp:
            z=temp[["score",t]].dropna(); r=stats.spearmanr(z.score,z[t]).statistic if len(z)>5 else np.nan
            null_rows.append([b,t,r])
null=pd.DataFrame(null_rows,columns=["replicate","target","null_rho"])
td=td.merge(null.groupby("target").null_rho.apply(list).rename("_null"),on="target",how="left")
td["random_signature_empirical_p"]=[(1+np.sum(np.abs(v)>=abs(r)))/(1+len(v)) if isinstance(v,list) and np.isfinite(r) else np.nan for v,r in zip(td._null,td.axis_dependency_rho)]
td=td.drop(columns="_null");td["random_signature_q"]=bh(td.random_signature_empirical_p)
td.to_csv(TAB/"target_dependency_validation.tsv",sep="\t",index=False)
null.to_csv(TAB/"random_signature_dependency_null.tsv.gz",sep="\t",index=False,compression="gzip")

# Cell-line-level PRISM association with frozen DepMap axis.
robust=read_tsv("03_analysis_ready/robustness_enrichment_figures/tables/robust_LINCS_candidates.tsv")
prismtab=read_tsv("03_analysis_ready/robustness_enrichment_figures/tables/PRISM_supported_candidates.tsv")
integ=read_tsv("03_analysis_ready/drug_reversal/tables/integrated_LINCS_PRISM_drug_candidates.tsv.gz")
candidates=set(integ.head(1000).pert_core.astype(str)); name_map=integ.drop_duplicates("pert_core").set_index("pert_core")["pert_iname"].to_dict()
lfc_path=ROOT/"01_raw_data/PRISM_23Q2/Repurposing_Public_23Q2_LFC_COLLAPSED.csv"
cell_path=ROOT/"01_raw_data/PRISM_23Q2/Repurposing_Public_23Q2_Cell_Line_Meta_Data.csv"
cm=pd.read_csv(cell_path,dtype=str)[["row_id","depmap_id","ccle_name"]].drop_duplicates("row_id")
parts=[]
for ch in pd.read_csv(lfc_path,usecols=["row_id","broad_id","LFC"],chunksize=200000,low_memory=False):
    ch["pert_core"]=ch.broad_id.astype(str).str.extract(r"(BRD-[A-Z][0-9]+)",expand=False)
    q=ch[ch.pert_core.isin(candidates)].copy()
    if len(q):parts.append(q)
pv=pd.concat(parts,ignore_index=True);pv.LFC=pd.to_numeric(pv.LFC,errors="coerce");pv=pv.dropna(subset=["LFC"])
pv=pv.merge(cm,on="row_id",how="left").merge(base[["ModelID","depmap_axis_z","uadt"]],left_on="depmap_id",right_on="ModelID",how="inner")
pv=pv[pv.uadt].groupby(["pert_core","depmap_id"],as_index=False).agg(LFC=("LFC","median"),depmap_axis_z=("depmap_axis_z","first"))
rows=[]
for pid,d in pv.groupby("pert_core"):
    if len(d)<8:continue
    rho,lo,hi=boot_spearman(d.depmap_axis_z,d.LFC);p=stats.spearmanr(d.depmap_axis_z,d.LFC).pvalue
    rows.append([pid,name_map.get(pid,pid),len(d),d.LFC.median(),rho,lo,hi,p])
pa=pd.DataFrame(rows,columns=["pert_core","pert_iname","n_uadt_lines","median_LFC","axis_LFC_rho","rho_ci_low","rho_ci_high","p"]);pa["q"]=bh(pa.p)
pa=pa.merge(integ[["pert_core","lincs_rank","lincs_median_reversal"]].drop_duplicates("pert_core"),on="pert_core",how="left")
pa.to_csv(TAB/"prism_axis_response_association.tsv",sep="\t",index=False)

# Integrated evidence table; no claim of efficacy.
drug_targets=pd.concat([robust,prismtab],ignore_index=True).drop_duplicates("pert_core")
support=[]
def truth(v): return str(v).strip().lower() in {"true","1","yes"}
for r in drug_targets.itertuples():
    ts=[gene(x) for x in str(getattr(r,"target","")).split("|") if gene(x) in set(td.target)]
    sub=td[td.target.isin(ts)]
    support.append({"pert_core":r.pert_core,"pert_iname":r.pert_iname,"targets":"|".join(ts),
      "robust_LINCS":truth(getattr(r,"robust_LINCS_candidate",False)),
      "PRISM_list":r.pert_core in set(prismtab.pert_core),
      "any_target_uadt_dependency_q_lt_0_05":bool((sub.selectivity_q<.05).any()),
      "any_target_axis_dependency_q_lt_0_05":bool((sub.axis_dependency_q<.05).any()),
      "interpretation":"computational validation candidate; not efficacy or cisplatin resensitization"})
evidence=pd.DataFrame(support).merge(pa[["pert_core","n_uadt_lines","axis_LFC_rho","q"]],on="pert_core",how="left")
evidence.to_csv(TAB/"candidate_evidence_matrix.tsv",sep="\t",index=False)

# Six-panel validation figure.
sns.set_theme(style="whitegrid",context="paper");plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42})
fig,axs=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
def panel(ax,l,t):ax.text(-.08,1.08,l,transform=ax.transAxes,fontweight="bold",fontsize=13,va="top");ax.set_title(t,fontweight="bold")
ax=axs[0,0];panel(ax,"a","Frozen external-validation design");ax.axis("off")
items=["Frozen\n300-gene axis","DepMap 24Q2\nexpression","CRISPR\ndependency","PRISM\ncell-line response","Nulls + FDR\nclaim gate"]
for i,s in enumerate(items):
    x=.01+i*.20;ax.add_patch(FancyBboxPatch((x,.38),.15,.25,boxstyle="round,pad=.02",fc=sns.color_palette("colorblind")[i],alpha=.18));ax.text(x+.075,.505,s,ha="center",va="center",fontweight="bold",fontsize=7)
ax=axs[0,1];panel(ax,"b","Frozen-axis activity across models");sns.violinplot(data=base.dropna(subset=["uadt"]),x="uadt",y="depmap_axis_z",inner=None,color="#88CCEE",ax=ax);sns.boxplot(data=base.dropna(subset=["uadt"]),x="uadt",y="depmap_axis_z",width=.22,showfliers=False,boxprops={"facecolor":"white"},ax=ax);ax.set_xticks([0,1],["Other","Head and neck"]);ax.set_xlabel("");ax.set_ylabel("DepMap frozen-axis z score")
ax=axs[0,2];panel(ax,"c","Upper-aerodigestive target dependency");show=td.sort_values("uadt_median_gene_effect").head(12);ax.barh(show.target,show.uadt_median_gene_effect,color=np.where(show.selectivity_q<.05,"#D55E00","#4477AA"));ax.axvline(0,c="k",lw=.7);ax.set_xlabel("Median CRISPR gene effect\n(lower = stronger dependency)")
ax=axs[1,0];panel(ax,"d","Axis-linked dependency estimates");show=td.sort_values("axis_dependency_rho");y=np.arange(len(show));ax.errorbar(show.axis_dependency_rho,y,xerr=[show.axis_dependency_rho-show.rho_ci_low,show.rho_ci_high-show.axis_dependency_rho],fmt="o",color="#0072B2",capsize=2);ax.axvline(0,c="k",ls="--",lw=.7);ax.set_yticks(y,show.target);ax.set_xlabel("Spearman rho: axis vs gene effect")
ax=axs[1,1];panel(ax,"e","PRISM response associated with axis");show=pa.sort_values("q").head(12).sort_values("axis_LFC_rho");y=np.arange(len(show));ax.hlines(y,show.rho_ci_low,show.rho_ci_high,color="#999999",lw=1.5);ax.scatter(show.axis_LFC_rho,y,c=np.where(show.q<.05,"#D55E00","#009E73"),s=24,zorder=3);ax.axvline(0,c="k",ls="--",lw=.7);ax.set_yticks(y,show.pert_iname);ax.set_xlabel("Spearman rho: axis vs LFC")
ax=axs[1,2];panel(ax,"f","Evidence convergence gate");cols=["robust_LINCS","PRISM_list","any_target_uadt_dependency_q_lt_0_05","any_target_axis_dependency_q_lt_0_05"];em=evidence.set_index("pert_iname")[cols].astype(float);em=em.loc[em.sum(axis=1).sort_values(ascending=False).head(12).index];sns.heatmap(em,cmap=sns.color_palette(["#F2F2F2","#228833"],as_cmap=True),vmin=0,vmax=1,cbar=False,linewidths=.5,ax=ax);ax.set_xticklabels(["LINCS","PRISM","UADT\ndependency","Axis-linked\ndependency"],rotation=25,ha="right");ax.set_ylabel("")
fig.suptitle("Orthogonal functional validation of frozen cisplatin-resistance hypotheses",fontweight="bold",fontsize=14)
for ext,dpi in [("png",300),("tif",600)]:fig.savefig(FIG/f"Figure_validation_v2.{ext}",dpi=dpi,bbox_inches="tight",facecolor="white")
fig.savefig(FIG/"Figure_validation_v2.pdf",bbox_inches="tight",facecolor="white");plt.close(fig)

# Reproducibility metadata and archive.
source={"DepMap_release":"24Q2","DepMap_DOI":"10.25452/figshare.plus.25880521.v1","DepMap_files":{},"seed":SEED,"frozen_up":len(up),"frozen_down":len(down)}
for n,p in paths.items():source["DepMap_files"][n]={"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"url":DEPMAP[n][0]}
(META/"source_versions.json").write_text(json.dumps(source,indent=2))
manifest=[]
for p in sorted(OUT.rglob("*")):
    if p.is_file():manifest.append({"relative_path":str(p.relative_to(OUT)),"size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(manifest).to_csv(OUT/"SHA256_manifest.tsv",sep="\t",index=False)
(OUT/"README.txt").write_text("Orthogonal computational validation of the frozen OSCC cisplatin-resistance axis. Associations are hypothesis-generating and do not establish efficacy, synergy, or cisplatin resensitization.\n")
archive=Path("/content/drug_val_out_v2.zip")
if archive.exists():archive.unlink()
shutil.make_archive(str(archive.with_suffix("")),"zip",OUT.parent,OUT.name)
print(f"COMPLETE: {archive} {archive.stat().st_size/1024/1024:.3f} MB")
try:
    from google.colab import files
    files.download(str(archive))
except Exception: pass
