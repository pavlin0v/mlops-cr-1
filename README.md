# Предсказание диабета

Градиентный бустинг на датасете [Pima Indians Diabetes](https://www.kaggle.com/datasets/saurabh00007/diabetescsv).

## Обучение

```bash
uv sync
uv run python train.py             
uv run python train.py --csv path/to/diabetes.csv
```

Модель сохраняется в `models/model.joblib`.

## Запуск сервиса

```bash
docker build -t diabetes-api .
docker run --rm -p 8000:8000 diabetes-api
```

Swagger: http://localhost:8000/docs

## Пример запроса

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
       "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}'
```

```json
{"outcome": 1, "probability": 0.828}
```
