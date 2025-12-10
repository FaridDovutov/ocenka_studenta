import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib # Добавляем библиотеку для сохранения модели

# 1. Загрузка данных
df = pd.read_csv('student_performance_dataset.csv')

# 2. Предобработка данных
if 'Student_ID' in df.columns:
    df = df.drop(columns=['Student_ID'])
if 'Pass_Fail' in df.columns:
    df = df.drop(columns=['Pass_Fail'])

# Сохраняем список колонок после кодирования (ВАЖНО для деплоя!)
# Нам нужен этот список, чтобы убедиться, что входные данные в Streamlit имеют те же колонки.
X = df.drop(columns=['Final_Exam_Score'])
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
FEATURE_COLUMNS = X_encoded.columns.tolist()

y = df['Final_Exam_Score']

# 3. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# 4. Создание и обучение модели XGBoost
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

print("Обучение модели...")
model.fit(X_train, y_train)

# 5. Оценка (для проверки)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R2 Score: {r2:.4f}")

# 6. СОХРАНЕНИЕ МОДЕЛИ И СПИСКА ПРИЗНАКОВ ДЛЯ DEPLOY
try:
    # Сохраняем обученную модель в файл model.pkl
    joblib.dump(model, 'model.pkl')
    print("Модель успешно сохранена в model.pkl")

    # Сохраняем список признаков, чтобы Streamlit знал, какие колонки нужны
    joblib.dump(FEATURE_COLUMNS, 'feature_columns.pkl')
    print("Список признаков сохранен в feature_columns.pkl")
except Exception as e:
    print(f"Ошибка при сохранении: {e}")