# Sens90 기준 민감도·특이도: OOF 예측에서 민감도 90% 이상을 만족하는 가장 큰 cutoff를 정하고 그 cutoff의 (Sens, Spec)을 구한다
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold


# 민감도 90% 이상을 만족하는 가장 큰 cutoff
def sens90(y, p):
    fpr, tpr, th = roc_curve(y, p); return th[np.argmax(tpr >= .9)]


# cutoff t에서의 (민감도, 특이도)
def sens_spec(y, p, t):
    q = p >= t; return q[y == 1].mean(), (~q[y == 0]).mean()


class Arr(np.ndarray):  # fold별 AUC 배열에 Sens90 기준 (Sens, Spec) 평균을 .ss로 붙여 다닌다 (통계량은 일반 float로 반환)
    mean = lambda self, *a, **k: np.asarray(self).mean(*a, **k)
    std = lambda self, *a, **k: np.asarray(self).std(*a, **k)
    var = lambda self, *a, **k: np.asarray(self).var(*a, **k)


# fold별 AUC와 반복(rep)별 OOF 예측으로 Arr 생성: .ss = 반복 평균 (Sens, Spec)
def with_ss(aucs, y, oofs):
    a = np.array(aucs).view(Arr); ss = [sens_spec(y, o, sens90(y, o)) for o in oofs]; a.ss, a.ss_sd = np.mean(ss, axis=0), np.std(ss, axis=0, ddof=1); return a  # ss_sd: 반복 간 SD


# 학습 코호트 OOF(5-fold)로 정한 Sens90 cutoff (외부검증에서 시험 코호트에 그대로 적용)
def train_cutoff(make, X, y):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return sens90(y, oof)
