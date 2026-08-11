"""Export exact AnnData schemas needed to build the advanced atlas workflow."""
from google.colab import drive, files
drive.mount('/content/drive')

import subprocess, sys
subprocess.check_call([sys.executable,'-m','pip','install','-q','anndata>=0.10','h5py>=3.10'])

from pathlib import Path
import json, zipfile
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path('/content/drive/MyDrive/OSCC_Cisplatin_PhysicsInformed_StateTransitions')
OUT=Path('/content/atlas_schema_out_v1');OUT.mkdir(parents=True,exist_ok=True)
objects={
 'GSE117872':ROOT/'03_analysis_ready/h5ad/GSE117872_log1pTPM_all_matched_cells.h5ad',
 'GSE103322':ROOT/'03_analysis_ready/h5ad/GSE103322_submitted_normalized_expression.h5ad',
 'GSE172577':ROOT/'03_analysis_ready/h5ad/GSE172577_counts_QC.h5ad',
 'GSE215403':ROOT/'03_analysis_ready/h5ad/GSE215403_counts_QC.h5ad',
 'GSE103322_scored':ROOT/'03_analysis_ready/state_space/GSE103322_lineage_CNV_resistance_scored.h5ad',
 'GSE172577_scored':ROOT/'03_analysis_ready/state_space/GSE172577_lineage_CNV_resistance_scored.h5ad',
 'GSE215403_scored':ROOT/'03_analysis_ready/state_space/GSE215403_lineage_CNV_resistance_scored.h5ad',
}

def scalar(v):
    if pd.isna(v):return None
    if isinstance(v,(np.integer,np.floating,np.bool_)):return v.item()
    return str(v)

schemas=[];value_rows=[]
for name,path in objects.items():
    if not path.exists():
        schemas.append({'name':name,'path':str(path),'exists':False});continue
    a=ad.read_h5ad(path,backed='r')
    x=a.X
    schemas.append({
      'name':name,'path':str(path.relative_to(ROOT)),'exists':True,
      'n_obs':a.n_obs,'n_vars':a.n_vars,'X_type':type(x).__name__,'X_dtype':str(x.dtype),
      'obs_columns':'|'.join(map(str,a.obs.columns)),'var_columns':'|'.join(map(str,a.var.columns)),
      'layers':'|'.join(map(str,a.layers.keys())),'obsm':'|'.join(map(str,a.obsm.keys())),
      'obsp':'|'.join(map(str,a.obsp.keys())),'raw_present':a.raw is not None,
      'obs_names_unique':bool(a.obs_names.is_unique),'var_names_unique':bool(a.var_names.is_unique),
    })
    for col in a.obs.columns:
        s=a.obs[col]
        nunique=int(s.nunique(dropna=True))
        if nunique<=100 or any(k in str(col).lower() for k in ['sample','patient','origin','condition','state','type','label','malig','batch','stage']):
            vc=s.astype(str).value_counts(dropna=False).head(40)
            for val,n in vc.items():
                value_rows.append({'object':name,'column':str(col),'n_unique':nunique,'value':str(val)[:300],'n':int(n)})
    del a

pd.DataFrame(schemas).to_csv(OUT/'h5ad_schema_v1.tsv',sep='\t',index=False)
pd.DataFrame(value_rows).to_csv(OUT/'obs_values_v1.tsv',sep='\t',index=False)

meta_files={
 'GSE117872_meta':ROOT/'02_preprocessing_audit/tables/GSE117872_cell_metadata_audited.tsv',
 'GSE103322_map':ROOT/'03_analysis_ready/metadata/GSE103322_cell_patient_mapping.tsv',
 'GSE172577_qc':ROOT/'03_analysis_ready/metadata/GSE172577_all_cell_QC.tsv.gz',
 'GSE215403_qc':ROOT/'03_analysis_ready/metadata/GSE215403_all_cell_QC.tsv.gz',
}
meta=[]
for name,path in meta_files.items():
    if not path.exists():meta.append({'name':name,'exists':False,'path':str(path)});continue
    d=pd.read_csv(path,sep='\t',nrows=200)
    meta.append({'name':name,'exists':True,'path':str(path.relative_to(ROOT)),'columns':'|'.join(map(str,d.columns)),'preview_rows':len(d)})
    d.head(25).to_csv(OUT/f'{name}_preview_v1.tsv',sep='\t',index=False)
pd.DataFrame(meta).to_csv(OUT/'metadata_schema_v1.tsv',sep='\t',index=False)

(OUT/'README_v1.txt').write_text(
 'Exact AnnData and metadata schemas for the OSCC advanced-atlas workflow. No expression matrices were exported.\n'
 'Return atlas_schema_out_v1.zip to Codex; the next notebook will run the actual scVI/CINEMA-OT analysis.\n')
zip_path=Path('/content/atlas_schema_out_v1.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):z.write(p,p.relative_to(OUT.parent))
print('COMPLETE:',zip_path,zip_path.stat().st_size/1024**2,'MB')
files.download(str(zip_path))
