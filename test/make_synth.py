import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
rng=np.random.default_rng(1); G=3000; P=6
base=rng.gamma(0.6,1.0,G)
ct_prog={t:np.exp(rng.normal(0,0.8,G)*(rng.random(G)<0.05)) for t in ["T","Myeloid","Fibro","Endo"]}
X=[];obs=[]
for p in range(P):
    batch=np.exp(rng.normal(0,0.15,G))
    cnv=np.ones(G); 
    for _ in range(4):
        s=rng.integers(0,G-200); cnv[s:s+200]*=rng.choice([0.5,1.5,2.0])
    mal_prog=np.exp(rng.normal(0,0.6,G)*(rng.random(G)<0.08))
    for t in ["T","Myeloid","Fibro","Endo","Malignant"]:
        n=300 if t=="Malignant" else 120
        mu=base*batch*(ct_prog[t] if t!="Malignant" else mal_prog*cnv)
        lib=rng.lognormal(7.5,0.4,n)[:,None]
        lam=lib*mu/mu.sum()
        X.append(rng.poisson(lam)); obs+= [dict(pt=f"P{p}",ct=t,mal="malignant" if t=="Malignant" else "normal")]*n
X=sp.csr_matrix(np.vstack(X).astype(np.float32))
a=ad.AnnData(X,obs=pd.DataFrame(obs,index=[f"c{i}" for i in range(X.shape[0])]),var=pd.DataFrame(index=[f"G{i}" for i in range(G)]))
from cfm_eval.data import standardise, qc
a=standardise(a,"pt","mal","ct","SYNTH")
a=qc(a,min_genes=50)
a.write_h5ad("test/synth.h5ad"); print(a)
