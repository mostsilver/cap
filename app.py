import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import joblib

# 모델 및 기존 데이터 로드
@st.cache_resource
def load_model():
    return joblib.load("multi_output_model.pkl")

@st.cache_resource
def load_data():
    return pd.read_csv("x_data.csv")

multi_output_model = load_model()
x_data = load_data()
features = list(x_data.columns)

# 사용자 입력 함수
def get_user_input(features):
    user_data = {}
    for i, feature in enumerate(features):
        user_data[feature] = st.text_input(f"{feature}:", key=f"user_input_{i}")
    return user_data

# gym 데이터
gym_data = {
    '셔틀정류장': ['학교', '학교', '학교', '무실행정복지센터정거장', '무실행정복지센터정거장', '무실행정복지센터정거장',
                  '코오롱 아파트', '코오롱 아파트', '코오롱 아파트', '시외버스터미널 건너편', '시외버스터미널 건너편',
                  '원주남부복합체육센터', '청소년수련원', '청소년수련원', '청소년수련원', '예술회관', '예술회관',
                  '예술회관', '단구2차 아파트', '단구2차 아파트', '동보노빌리티', '동보노빌리티'],
    '헬스장 목록': ['학교 헬스장', '훈짐', '다르짐', 'ROAD GYM', '커브스 무실클럽', '무실 스쿼시센터',
                   '바디스튜디오 카모', '메가짐&필라테스', '크로스핏 컴벳', '메가짐&필라테스', '바디스튜디오 카모',
                   '원주남부복합체육센터', '히든헬스클럽', '컨비던스 짐', '노익스 짐', '원주국민체육센터', '유진 헬스클럽',
                   '크로스핏 사나래', '커브스 단구클럽', 'HM 휘트니스', '몸수선토탈케어pt샵', '펄핏 스튜디오pt'],
    'A': [3, 3.7, 2.5, 3.2, 3.2, 2.8, 3.2, 4.2, 3.5, 4.2, 3.2, 4.7, 3.7, 3.8, 3.4, 2.5, 2.4, 2.6, 2.3, 4.1, 3.1, 4.2],
    'B': [2.7, 3.5, 2.8, 4, 2.8, 2.7, 3.4, 3.4, 4.2, 3.4, 3.4, 3.7, 3.4, 3.8, 3.5, 2.6, 2.5, 3.9, 3.9, 3.9, 2.5, 2.8],
    'C': [1, 5, 3.3, 3.2, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3.2, 3, 3.9, 3.7, 2.8, 2.8, 2.8, 3.1, 2.9]
}

gym_df = pd.DataFrame(gym_data)

# Streamlit 앱
st.title("Multi-Output Model Prediction & Gym Recommendation")
st.write("이 앱은 사용자의 입력 데이터를 기반으로 타겟 값을 예측하고, 추천 헬스장을 제공합니다.")

# 사용자 입력 섹션
st.header("사용자 데이터 입력")
user_input = get_user_input(features)

# 셔틀 정류장 선택 섹션
st.header("셔틀 정류장 선택")
shuttle_options = gym_df['셔틀정류장'].unique()
selected_shuttle = st.selectbox("셔틀 정류장을 선택하세요:", shuttle_options)

# 셔틀 정류장 필터링
filtered_gyms = gym_df[gym_df['셔틀정류장'] == selected_shuttle]

# 필터링 결과 표시
st.subheader("선택한 셔틀 정류장에 해당하는 헬스장 목록")
st.write(filtered_gyms[['셔틀정류장', '헬스장 목록', 'A', 'B', 'C']])

# 중요도 예측 후 세션 상태로 진행
if st.button("Predict & Recommend"):
    try:
        # 입력 데이터를 DataFrame으로 변환
        x_new = pd.DataFrame([user_input])
        
        # 문자열 입력을 float로 변환하고, 빈값은 NaN 처리
        x_new = x_new.apply(pd.to_numeric, errors='coerce')

        # 기존 데이터와 사용자 입력 결합
        x_filled = x_data.copy()
        x_filled.loc[len(x_filled)] = x_new.iloc[0]

        # 결측값 채우기 (평균값으로)
        imputer = SimpleImputer(strategy='mean')
        x_filled = pd.DataFrame(imputer.fit_transform(x_filled), columns=x_data.columns)

        # 사용자 데이터만 추출
        x_user_filled = x_filled.iloc[-1:].reset_index(drop=True)

        # 모델로 예측
        y_pred = multi_output_model.predict(x_user_filled)
        y_pred_df = pd.DataFrame(y_pred, columns=[f"Target_Y{i+1}" for i in range(y_pred.shape[1])])

        # 예측 결과를 세션 상태에 저장
        st.session_state.prediction = y_pred_df

        # 중요도 가중치 설정 (모델 예측값 기반)
        weights = y_pred_df.iloc[0]
        filtered_gyms['합계 점수'] = (
            filtered_gyms['A'] * weights[0] +
            filtered_gyms['B'] * weights[1] +
            filtered_gyms['C'] * weights[2]
        )

        # 전체 헬스장에 점수 적용
        gym_df['합계 점수'] = (
            gym_df['A'] * weights[0] +
            gym_df['B'] * weights[1] +
            gym_df['C'] * weights[2]
        )

        # 전체 헬스장 목록 출력
        st.subheader("전체 헬스장 목록")
        st.write(gym_df[['셔틀정류장', '헬스장 목록', 'A', 'B', 'C', '합계 점수']])

        # 새로운 세션으로 이동할 버튼 추가
        st.button("다음 세션으로 이동", on_click=lambda: st.experimental_rerun())

    except Exception as e:
        st.error(f"오류 발생: {e}")

# 예측 결과 출력
if 'prediction' in st.session_state:
    st.subheader("Prediction Results")
    st.write(st.session_state.prediction)
