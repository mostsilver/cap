# cap
캡스톤_헬스장 추천

1. GitHub 저장소 준비
Streamlit 애플리케이션을 GitHub에 업로드해야 합니다.

GitHub 저장소 생성
GitHub에 로그인한 후 새 저장소를 생성합니다.
Streamlit 파일 추가
app.py 파일과 필요한 데이터 파일(예: x_data.csv, 모델 파일 등)을 저장소에 업로드합니다.

-폴더 구조 예:
.
├── app.py          # Streamlit 앱 코드
├── requirements.txt # 필요한 라이브러리 목록
├── x_data.csv       # 데이터 파일
└── multi_output_model.pkl # 모델 파일

requirements.txt 작성
앱에서 사용하는 Python 라이브러리를 requirements.txt 파일에 나열합니다.
예:
streamlit
pandas
numpy
scikit-learn
joblib



사용자에게 헬스장을 추천하는 시스템으로 작업중
