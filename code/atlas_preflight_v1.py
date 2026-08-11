"""Colab preflight for the high-information OSCC single-cell atlas upgrade."""
from google.colab import drive, files
drive.mount('/content/drive')

from pathlib import Path
import gzip, hashlib, json, os, re, shutil, zipfile
import pandas as pd

ROOT=Path('/content/drive/MyDrive/OSCC_Cisplatin_PhysicsInformed_StateTransitions')
OUT=Path('/content/atlas_preflight_out_v1')
OUT.mkdir(parents=True,exist_ok=True)

if not ROOT.exists():
    raise FileNotFoundError(f'Project root not found: {ROOT}')

interesting={'.h5ad','.h5','.loom','.rds','.mtx','.csv','.tsv','.txt','.gz','.npz','.parquet'}
keywords=re.compile(r'(GSE117872|GSE103322|GSE172577|GSE215403|matrix|count|feature|barcode|metadata|annotation|cell_state|coordinate|velocity|spliced|unspliced)',re.I)

def sha256_head(path,n=4*1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:h.update(f.read(n))
    return h.hexdigest()

def text_head(path,n=3):
    try:
        op=gzip.open if path.suffix=='.gz' else open
        with op(path,'rt',errors='replace') as f:return [next(f).rstrip('\n') for _ in range(n)]
    except Exception:return []

rows=[]
for p in ROOT.rglob('*'):
    if not p.is_file():continue
    rel=str(p.relative_to(ROOT)); suffix=p.suffix.lower()
    if suffix not in interesting and not keywords.search(rel):continue
    st=p.stat();head=text_head(p)
    rows.append({
        'path':rel,'bytes':st.st_size,'suffix':suffix,
        'sha256_first4MB':sha256_head(p),
        'header_1':head[0][:1000] if len(head)>0 else '',
        'header_2':head[1][:1000] if len(head)>1 else '',
        'mentions_spliced':bool(re.search(r'spliced|unspliced|ambiguous',rel+' '+' '.join(head),re.I)),
        'mentions_celltype':bool(re.search(r'cell.?type|annotation|malignant|epithelial',rel+' '+' '.join(head),re.I)),
        'mentions_sample':bool(re.search(r'patient|sample|origin|condition|resistant|sensitive|holiday',rel+' '+' '.join(head),re.I)),
    })

inv=pd.DataFrame(rows).sort_values(['bytes','path'],ascending=[False,True])
inv.to_csv(OUT/'file_inventory_v1.tsv',sep='\t',index=False)

datasets=['GSE117872','GSE103322','GSE172577','GSE215403']
summary=[]
for ds in datasets:
    d=inv[inv.path.str.contains(ds,case=False,na=False)]
    summary.append({
        'dataset':ds,'n_files':len(d),'total_GB':d.bytes.sum()/1024**3,
        'has_h5ad_or_h5':d.suffix.isin(['.h5ad','.h5']).any(),
        'has_mtx':d.path.str.contains(r'\.mtx',case=False,regex=True).any(),
        'has_features':d.path.str.contains('feature',case=False).any(),
        'has_barcodes':d.path.str.contains('barcode',case=False).any(),
        'has_spliced_layers':d.mentions_spliced.any(),
        'has_cell_annotations':d.mentions_celltype.any(),
        'has_sample_metadata':d.mentions_sample.any(),
    })
summary=pd.DataFrame(summary)
summary.to_csv(OUT/'dataset_capability_v1.tsv',sep='\t',index=False)

gates={
 'scVI_scANVI': 'GO if raw count matrix, batch/sample labels, and stable cell annotations are present for at least two atlas cohorts.',
 'Milo': 'GO only for comparisons with replicated patient/sample units; never use cells as replicates.',
 'LIANA_plus': 'GO for cohorts retaining malignant and stromal/immune cell types with replicated samples; run per sample then aggregate.',
 'CellOT_or_CINEMA_OT': 'GO for GSE117872 if sensitive/resistant/drug-holiday labels and count-level profiles are recoverable; validate by held-out origin.',
 'CellRank_velocity': 'GO only if spliced and unspliced layers are present. Otherwise do not run.',
 'decoupler_PROGENy_DoRothEA': 'GO with count/expression matrices and gene symbols; summarize at patient/sample level.',
 'pseudobulk_DE': 'GO with raw counts and replicated samples/paired origins; use edgeR/DESeq2 rather than cell-level tests.',
}
(OUT/'method_gates_v1.json').write_text(json.dumps(gates,indent=2)+'\n')

readme=[
 'OSCC atlas-upgrade preflight v1',
 f'Project root: {ROOT}',
 f'Files inventoried: {len(inv):,}',
 '',
 'This export contains filenames, sizes, short headers, and first-4-MB hashes—not expression matrices or patient-level raw data.',
 '',
 summary.to_string(index=False),
 '',
 'Return this ZIP to Codex. The next notebook will be selected from the data-supported methods only.'
]
(OUT/'README_v1.txt').write_text('\n'.join(readme)+'\n')

zip_path=Path('/content/atlas_preflight_out_v1.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(OUT.parent))
print('COMPLETE:',zip_path,zip_path.stat().st_size/1024**2,'MB')
files.download(str(zip_path))
