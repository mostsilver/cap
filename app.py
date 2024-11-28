import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import joblib


# 모델 및 기존 데이터 로드
@st.cache_resource  # 모델 로드를 캐시로 저장하여 효율성 향상
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
        # key를 명시적으로 지정하여 중복 방지
        user_data[feature] = st.text_input(f"{feature}:", key=f"user_input_{i}")
    return user_data


# Streamlit 앱
st.title("Multi-Output Model Prediction")
st.write("이 앱은 사용자의 입력 데이터를 기반으로 타겟 값을 예측합니다.")

# 사용자 입력 섹션
st.header("사용자 데이터 입력")
user_input = get_user_input(features)

# 결측값 처리 및 데이터 준비
if st.button("Predict"):
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

        # 예측 결과 출력
        st.subheader("Prediction Results")
        st.write(y_pred_df)
    except Exception as e:
        st.error(f"오류 발생: {e}")




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
#def get_user_input(features):
#    user_data = {}
#    for feature in features:
#        user_data[feature] = st.text_input(f"{feature}:")
#    return user_data
###############################################################다음장












