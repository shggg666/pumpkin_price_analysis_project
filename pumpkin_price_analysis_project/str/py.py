import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import joblib
import os

# 中文显示设置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 统一路径
DATA_DIR = "data"
NOOKS_DIR = "nooks"
MODEL_DIR = os.path.join(NOOKS_DIR, "models")
IMAGES_DIR = os.path.join(NOOKS_DIR, "images")
for d in [NOOKS_DIR, MODEL_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)


def load_and_preprocess_data(filename: str):
    """加载并预处理"""
    file_path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(file_path)

    rows, columns = df.shape
    if rows < 1 or columns < 1:
        raise ValueError("数据集为空或格式不正确")

    selected = ['Date', 'City Name', 'Package', 'Variety', 'Low Price', 'High Price']
    df = df[selected].dropna().reset_index(drop=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    def season(month):
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        else:
            return 'Winter'

    df['Season'] = df['Month'].apply(season)
    df['AveragePrice'] = (df['Low Price'] + df['High Price']) / 2
    return df


def feature_engineering(df):
    cat_cols = ['City Name', 'Package', 'Variety', 'Season']
    return pd.get_dummies(df, columns=cat_cols)


def prepare_data(df_encoded):
    X = df_encoded.drop(['Date', 'Low Price', 'High Price', 'AveragePrice', 'Year'], axis=1)
    y = df_encoded['AveragePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        '线性回归': LinearRegression(),
        '岭回归': Ridge(alpha=1.0),
        'K近邻': KNeighborsRegressor(n_neighbors=5),
        '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
        '梯度提升树': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    results = {}
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        results[name] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        print(f"{name} 测试结果 — MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    return results, models


def plot_feature_importance(model, X_train):
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        cols = X_train.columns
        idx = np.argsort(fi)[-20:]
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(idx)), fi[idx], align='center')
        plt.yticks(range(len(idx)), [cols[i] for i in idx])
        plt.title('特征重要性')
        plt.xlabel('重要性分数')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'feature_importance.png'))
        plt.show()


def plot_prediction_vs_actual(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('预测值 vs 实际值')
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(IMAGES_DIR, 'prediction_vs_actual.png'))
    plt.show()


def save_model(model, X_train, model_name='pumpkin_price_model.pkl'):
    joblib.dump(model, os.path.join(MODEL_DIR, model_name))
    np.save(os.path.join(MODEL_DIR, 'feature_names.npy'), X_train.columns)
    print(f"\n模型与特征名已保存到 {MODEL_DIR}")


def main():
    try:
        print("\n===== 数据加载与预处理 =====")
        df = load_and_preprocess_data('US-pumpkins.csv')

        print("\n===== 特征工程 =====")
        df_encoded = feature_engineering(df)

        print("\n===== 数据准备 =====")
        X_train, X_test, y_train, y_test = prepare_data(df_encoded)

        print("\n===== 模型训练与评估 =====")
        results, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

        # 跳过超参数调优，直接用默认 XGBoost
        print("\n===== 跳过超参数调优 =====")
        best_model = models['XGBoost']

        print("\n===== 特征重要性分析 =====")
        plot_feature_importance(best_model, X_train)

        y_test_pred = best_model.predict(X_test)
        plot_prediction_vs_actual(y_test, y_test_pred)

        save_model(best_model, X_train)

    except Exception as e:
        print(f"运行出错：{e}")


if __name__ == '__main__':
    main()