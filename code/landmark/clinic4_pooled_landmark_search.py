# 탐색 스크립트: gangnam+sinchon 풀링 5-fold CV(5회 반복 OOF)에서 clinic4 대비 landmark 체성분 모델 DeLong 비교 (프로젝트 루트에서 python -I 로 실행)
import sys, pathlib as p, warnings
warnings.filterwarnings("ignore")
sys.path[:0] = [str(d) for d in p.Path("code").iterdir() if d.is_dir()]
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import landmark_mlp_sets as L
from clinic4_landmark_vat_auc import load_cohort
sys.stdout.reconfigure(encoding="utf-8")
D=["HTN","DM","CKD"];LMN=L.LMN
import io,contextlib
with contextlib.redirect_stdout(io.StringIO()):
    out={}
    for c in ["gangnam","sinchon"]:
        df,cur=load_cohort(c); df2=L.load(c)
        aec=pd.read_excel(f"data/{c}_landmark_filtered.xlsx",sheet_name="aec_total").set_index("PatientID").filter(regex=r"^aec_\d+$").loc[df2.PatientID].to_numpy(float)
        lm=pd.read_excel(f"data/{c}_landmark_filtered.xlsx",sheet_name="landmarks").set_index("PatientID").loc[df2.PatientID]
        a=[]
        for i in range(len(df2)):
            s_,e_=int(lm.iloc[i]["liver_dome_slice"]),int(lm.iloc[i]["inferior_pubic_margin_slice"])
            a.append(np.interp(np.linspace(s_,e_,128),np.arange(1,aec.shape[1]+1),np.nan_to_num(aec[i])))
        cur["AEC"]=np.array(a); df2["cohort"]=c; out[c]=(df2,cur)
g,gc=out["gangnam"];s,sc=out["sinchon"]
A=pd.concat([g,s],ignore_index=True);cur={t:np.vstack([gc[t],sc[t]]) for t in gc}
coh=A.cohort.to_numpy()
male=(A.PatientSex.astype(str).str.upper()=="M").to_numpy().astype(float)
C4=np.column_stack([male,A.PatientAge,A.Height,A.Weight])
# 성별 내 표준화(z within sex) 체성분 landmark 값
def within_sex(M):
    M=M.copy()
    for sx in (0,1):
        m=male==sx;M[m]=(M[m]-M[m].mean(0))/(M[m].std(0)+1e-9)
    return M
LM_cols=[f"{t}@{n}" for t in ["VAT","SAT","TAMA","AEC"] for n in LMN]
LMv=A[LM_cols].to_numpy(float)
bmi=(A.Weight/(A.Height/100)**2).to_numpy()
# 키/체중에 대한 잔차 (body size 보정된 체성분)
def resid(M):
    X=np.column_stack([np.ones(len(M)),male,A.Height,A.Weight,A.PatientAge,male*A.Weight,male*A.Height])
    b=np.linalg.lstsq(X,M,rcond=None)[0];return M-X@b
curveAll=np.hstack([cur[t] for t in ["VAT","SAT","TAMA","AEC"]])
sets={"clinic4":C4,
 "c4+LM(raw)":np.column_stack([C4,LMv]),
 "c4+LM(within-sex z)":np.column_stack([C4,within_sex(LMv)]),
 "c4+LM(resid)":np.column_stack([C4,resid(LMv)]),
 "c4+curves":np.column_stack([C4,curveAll]),
 "c4+curves(resid)":np.column_stack([C4,resid(curveAll)]),
 "c4+LM+curves":np.column_stack([C4,LMv,curveAll])}
def mk(C): return lambda: make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=5000))
def HGB(): return HistGradientBoostingClassifier(max_depth=3,learning_rate=0.04,max_iter=200,l2_regularization=10,random_state=0)
def cvpool(mkf,X,y,reps=3):
    r=[]
    for k in range(reps):
        o=np.zeros(len(y))
        strat=y*2+(coh=="sinchon")
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=k).split(X,strat):
            o[te]=mkf().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
        r.append(roc_auc_score(y,o))
    return np.mean(r)
def cross(mkf,X,y):
    a=coh=="gangnam";b=~a
    r1=roc_auc_score(y[b],mkf().fit(X[a],y[a]).predict_proba(X[b])[:,1])
    r2=roc_auc_score(y[a],mkf().fit(X[b],y[b]).predict_proba(X[a])[:,1])
    return r1,r2

from delong_utils import delong_paired_auc_test
def oof(mkf,X,y,reps=5):
    o=np.zeros(len(y))
    for k in range(reps):
        strat=y*2+(coh=="sinchon")
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=100+k).split(X,strat):
            o[te]+=mkf().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]/reps
    return o
sexint=lambda M:np.column_stack([M,M*male[:,None]])

lr_=lambda a,b:np.log((a+1)/(b+1))
rat=[]
for n in LMN:
    v,sa,t=A[f"VAT@{n}"].to_numpy(float),A[f"SAT@{n}"].to_numpy(float),A[f"TAMA@{n}"].to_numpy(float)
    rat+= [lr_(v,sa),lr_(v,t),lr_(t,v+sa)]
RAT=np.column_stack(rat)
LMR=np.column_stack([C4,LMv,RAT])
from sklearn.base import BaseEstimator,ClassifierMixin
class Ens(BaseEstimator,ClassifierMixin):
    def fit(self,X,y):
        self.a=mk(.01)().fit(X,y);self.b=HGB().fit(X,y);return self
    def predict_proba(self,X):
        from scipy.stats import rankdata
        p=(rankdata(self.a.predict_proba(X)[:,1])+rankdata(self.b.predict_proba(X)[:,1]))/2/len(X)
        return np.column_stack([1-p,p])
S2={"LM":(np.column_stack([C4,LMv]),mk(.01)),"LM+ratios":(LMR,mk(.01)),"LM+ratios C.003":(LMR,mk(.003)),"LM ens LR+HGB":(np.column_stack([C4,LMv]),Ens),
"LM+curves+ratios":(np.column_stack([LMR,curveAll]),mk(.005))}
for d in D:
    y=A[d].to_numpy(int);b=oof(mk(1),C4,y)
    print("==",d,flush=True)
    for n,(X,m) in S2.items():
        o=oof(m,X,y,3);r=delong_paired_auc_test(y,b,o)
        print(f"{n:18s} auc={roc_auc_score(y,o):.3f} diff={r['diff']:+.3f} p={r['p_value']:.4f}",flush=True)
