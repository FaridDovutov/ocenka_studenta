import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Загрузка обученной модели и списка признаков
@st.cache_resource
def load_resources():
    # Проверяем наличие файлов перед загрузкой
    if not os.path.exists('model.pkl') or not os.path.exists('feature_columns.pkl'):
        st.error("Файлы 'model.pkl' или 'feature_columns.pkl' не найдены. Сначала запустите train_model.py")
        return None, None
    
    model = joblib.load('model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, feature_columns

model, FEATURE_COLUMNS = load_resources()

if model is None:
    st.stop()


# ----------------------------------------------------
# 1. Заголовок и описание
# ----------------------------------------------------
st.title('🎓 Предсказание оценки студента (XGBoost)')
st.markdown("""
    Введите данные студента, чтобы предсказать его итоговую оценку на экзамене.
    Модель обучена на **Attendance Rate**, **Study Hours** и **Past Exam Scores**.
""")

# ----------------------------------------------------
# 2. Форма ввода данных
# ----------------------------------------------------

with st.form("prediction_form"):
    st.header("Входные данные студента")

    # Ввод данных для 3 самых важных признаков
    attendance = st.slider('Attendance Rate (%)', min_value=50.0, max_value=100.0, value=85.0, step=0.1)
    study_hours = st.slider('Study Hours per Week', min_value=5, max_value=40, value=25, step=1)
    past_scores = st.slider('Past Exam Scores (Баллы)', min_value=50, max_value=100, value=75, step=1)
    
    # Ввод остальных признаков (сгруппированы)
    st.subheader("Дополнительные факторы")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        internet = st.selectbox('Internet Access at Home', ['Yes', 'No'])
    with col2:
        parent_edu = st.selectbox('Parental Education Level', ['High School', 'Masters', 'Bachelors', 'PhD'])
        extracurricular = st.selectbox('Extracurricular Activities', ['Yes', 'No'])
    
    # Кнопка отправки
    submitted = st.form_submit_button("Предсказать оценку")

# ----------------------------------------------------
# 3. Логика предсказания
# ----------------------------------------------------

if submitted:
    # 1. Создание DataFrame из введенных данных
    input_data = {
        'Study_Hours_per_Week': [study_hours],
        'Attendance_Rate': [attendance],
        'Past_Exam_Scores': [past_scores],
        'Gender': [gender],
        'Parental_Education_Level': [parent_edu],
        'Internet_Access_at_Home': [internet],
        'Extracurricular_Activities': [extracurricular]
    }
    input_df = pd.DataFrame(input_data)

    # 2. One-Hot Encoding (Должно точно соответствовать обучению!)
    categorical_cols = input_df.select_dtypes(include=['object']).columns
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # 3. Выравнивание признаков: создание всех колонок, которые были при обучении
    # Это ключевой шаг для деплоя, чтобы избежать ошибок с недостающими колонками
    final_input = pd.DataFrame(0, index=input_encoded.index, columns=FEATURE_COLUMNS)
    
    # Копируем значения из входного DataFrame
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col]

    # 4. Предсказание
    try:
        prediction = model.predict(final_input)[0]
        
        # 5. Вывод результата
        st.success('✅ Предсказание готово!')
        st.metric(
            label="Итоговая оценка", 
            value=f"{prediction:.2f} баллов"
        )
        
        # Визуальный индикатор
        if prediction >= 70:
            st.balloons()
            st.write("Отличный результат! Студент, скорее всего, получит высокую оценку.")
        elif prediction >= 50:
            st.write("Хороший результат. Студент, вероятно, сдаст экзамен.")
        else:
            st.warning("Низкий результат. Студенту требуется дополнительная подготовка.")
            
    except Exception as e:
        st.error(f"Произошла ошибка при предсказании: {e}")
