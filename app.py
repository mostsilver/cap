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


        # 예측값을 DataFrame으로 변환하여 출력
        y_pred_df = pd.DataFrame(y_pred, columns=[f"Target_Y{i+1}" for i in range(y_pred.shape[1])])
        st.subheader("예측된 Target 값:")
        st.write(y_pred_df)

        

        # 예측값을 A, B, C로 계산된 결과에 곱해줍니다.
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

        # 예측값을 각 항목별로 곱한 결과 계산
        df_product = pd.DataFrame(columns=df.columns)

        for i, col in enumerate(df.columns):
            df_product[col] = df.iloc[:, i] * y_pred[0, i]

        # 결과 출력
        st.subheader("A, B, C 항목별 예측값 곱한 결과:")
        st.write(df_product)


        # A, B, C 항목의 합 계산
        sum_a = df_product.loc['A'].sum()
        sum_b = df_product.loc['B'].sum()
        sum_c = df_product.loc['C'].sum()

        # 출력
        st.write(f"\nA 항목 합: {sum_a}")
        st.write(f"B 항목 합: {sum_b}")
        st.write(f"C 항목 합: {sum_c}")

        # 합산 값 기준으로 정렬 (내림차순)
        sums = [('A', sum_a), ('B', sum_b), ('C', sum_c)]
        sums_sorted = sorted(sums, key=lambda x: x[1], reverse=True)

        # 정렬된 순서대로 출력
        sorted_people = ''.join([x[0] for x in sums_sorted])
        st.write(f"\n사용자의 유형: {sorted_people}")



        
        # 데이터 입력
        gym = {
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

        #'이성 A(환경) 점수'
        #감성 B(운영) 점수
        #지도자 C(지도자) 점수
        # DataFrame으로 변환
        gym = pd.DataFrame(gym)

        # a, b, c 점수에 따라 순서 계산 (내림차순으로 정렬)
        gym['abc 순서'] = gym.apply(lambda row: ''.join(sorted(['A', 'B', 'C'], key=lambda x: row[x], reverse=True)), axis=1)


        # 중요도 점수를 Series로 변환
        weights = df_sums_transposed.iloc[0]  # Value 행만 선택

        # A, B, C 점수와 중요도를 곱한 뒤 합산
        gym['합계 점수'] = (
            gym['A'] * weights['A'] +
            gym['B'] * weights['B'] +
            gym['C'] * weights['C']
        )

        # 합계 점수가 가장 큰 헬스장 선택
        best_gym = gym[gym['합계 점수'] == gym['합계 점수'].max()]

        # 결과 출력
        print("합계 점수가 가장 높은 헬스장:")
        print(best_gym[['셔틀정류장', '헬스장 목록', 'A', 'B', 'C', '합계 점수']])



          # 출력
        st.write(f"합계 점수가 가장 높은 헬스장:")
        st.write(f"best_gym[['셔틀정류장', '헬스장 목록', 'A', 'B', 'C', '합계 점수']]")









    
    except Exception as e:
        st.error(f"오류 발생: {e}")
