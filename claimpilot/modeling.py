"""공개 라벨 기반 검토 순위 평가. 사기·지급 판단을 자동 확정하지 않는다."""
import math
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, precision_recall_curve, brier_score_loss

TARGET = 'FraudFound_P'
EXCLUDED = {
    TARGET: '정답 라벨', 'PolicyNumber': '식별자·정렬 순서 누수 위험',
    'RepNumber': '담당자 식별 대리변수', 'Year': '연도 분포 변화·분할 점검 전용',
    'Sex': '민감 인구통계', 'MaritalStatus': '인구통계 대리변수',
    'Age': '인구통계 및 0값 의미 불명', 'AgeOfPolicyHolder': '인구통계 대리변수',
    'Fault': '책임 판정 시점 불명·사후 판단 위험',
    'NumberOfSuppliments': '추가자료/변경 발생 시점과 의미 불명',
    'AddressChange_Claim': '청구 전후 시점 불명',
    'PoliceReportFiled': '확보 시점 불명', 'WitnessPresent': '확인 시점 불명',
    'DriverRating': '등급 산정 기준 미제공',
}

FEATURES = ['Month','WeekOfMonth','DayOfWeek','Make','AccidentArea',
    'DayOfWeekClaimed','MonthClaimed','WeekOfMonthClaimed','PolicyType',
    'VehicleCategory','VehiclePrice','Deductible','Days_Policy_Accident',
    'Days_Policy_Claim','PastNumberOfClaims','AgeOfVehicle','AgentType',
    'NumberOfCars','BasePolicy']

def feature_columns(df):
    """허용 목록 밖의 입력은 검토 없이 모델에 들어갈 수 없다."""
    unexpected = set(df.columns) - set(FEATURES) - set(EXCLUDED)
    missing = set(FEATURES) - set(df.columns)
    if unexpected or missing:
        raise ValueError(f'스키마 변경: 예상 밖 열={sorted(unexpected)}, 누락 열={sorted(missing)}')
    return list(FEATURES)

def prepare(df, features):
    """모든 변수를 범주형으로 취급. 코드값에 임의 순서/거리를 부여하지 않는다."""
    return df[features].fillna('__MISSING__').astype(str)

def evaluate(y, scores, threshold=.5):
    y = np.asarray(y); scores = np.asarray(scores)
    if len(y) == 0 or not np.isfinite(scores).all():
        raise ValueError('유효한 평가 표본과 점수가 필요합니다.')
    # 동점은 원본 순서 유지. 비교 모델 모두 동일한 최종평가 순서 사용.
    k = max(1, math.ceil(len(y) * .2))
    top = np.argsort(-scores, kind='stable')[:k]
    captured = int(y[top].sum()); positives = int(y.sum())
    precision20 = captured / k
    prevalence = float(y.mean())
    pred = scores >= threshold
    return dict(ap=float(average_precision_score(y, scores)),
        precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)),
        capture_at_20=captured/positives if positives else 0.,
        precision_at_20=precision20, lift_at_20=precision20/prevalence if prevalence else 0.,
        reviewed_count=k, captured_count=captured, positives=positives, n=len(y),
        brier=float(brier_score_loss(y,scores)))

def choose_threshold(y, scores):
    p,r,t = precision_recall_curve(y,scores)
    f = 2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    return float(t[np.argmax(f)]) if len(t) else .5
