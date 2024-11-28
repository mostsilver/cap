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




# 사용자 입력 함수
#def get_user_input(features):
#    user_data = {}
#    for feature in features:
#        user_data[feature] = st.text_input(f"{feature}:")
#    return user_data
###############################################################다음장



if st.button("다음 세션으로 이동"):
    st.experimental_rerun()


import pandas as pd

# 예측값 (A, B, C별 예측값)
data = {
    's1': [3.301927249, 3, 3],
    's2': [3.914867641, 3.301927249, 3.174802104],
    's3': [2.15443469, 3.556893304, 3.301927249],
    's4': [2.080083823, 3.634241186, 3.301927249],
    's5': [2.884499141, 3.301927249, 3.301927249],
    's6': [3, 3.634241186, 4],
    's7': [3.301927249, 3.634241186, 3.301927249],
    's8': [4, 3.634241186, 3.301927249],
    's9': [3.301927249, 3.301927249, 4],
    's10': [3, 3.914867641, 3.301927249],
    's11': [3.174802104, 3, 3.634241186],
    's12': [3, 2.884499141, 3.301927249],
    's13': [2.289428485, 3, 3.914867641],
    's14': [2.080083823, 3.301927249, 4.217163327],
    's15': [2.289428485, 3, 4.217163327],
    's16': [2.080083823, 3, 4.641588834],
    's17': [2.289428485, 3, 4.30886938],
    's18': [2.289428485, 3, 3.914867641]
}

# DataFrame으로 변환
df = pd.DataFrame(data, index=['A', 'B', 'C'])


# 예측값을 DataFrame으로 변환
y_pred = pd.DataFrame(y_pred)

# 예측값을 각 항목별로 곱한 결과 계산
df_product = pd.DataFrame(columns=df.columns)

# s1, s2, ..., s18에 대해 y_pred 값을 곱하기
for i, col in enumerate(df.columns):
    df_product[col] = df.iloc[:, i] * y_pred.iloc[0, i]

# 결과 출력
print("A, B, C 항목별 예측값 곱한 결과:")
print(df_product)

# A, B, C 항목의 합 계산
sum_a = df_product.loc['A'].sum()
sum_b = df_product.loc['B'].sum()
sum_c = df_product.loc['C'].sum()

# 출력
print("\nA 항목 합:", sum_a)
print("B 항목 합:", sum_b)
print("C 항목 합:", sum_c)

# sum_a, sum_b, sum_c 순서대로 리스트로 저장
sums = [('A', sum_a), ('B', sum_b), ('C', sum_c)]

# 합산 값 기준으로 정렬 (내림차순)
sums_sorted = sorted(sums, key=lambda x: x[1], reverse=True)

# 정렬된 순서대로 사람 매칭
sorted_people = ''.join([x[0] for x in sums_sorted])

print("\n정렬된  순서:", sorted_people)

# 같은 값을 가지는 헬스장 찾기
# 예를 들어, 동일한 합을 가진 헬스장을 찾아 출력
target_value = sums_sorted[0][1]  # 예시: 최대값을 기준으로

# 해당 합을 가진 사람들 추출
matching_gyms = [gym for gym, value in sums if value == target_value]

print("\n같은 합을 가진 헬스장:", matching_gyms)







