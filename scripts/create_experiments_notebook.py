import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 02 — Model Experiments: So sánh 3 Thí nghiệm Xử lý Dữ liệu Trùng lặp\n",
                "**Dự án:** Medical Chatbot — Phân loại bệnh dựa trên triệu chứng (ML Core)\n",
                "\n",
                "Notebook này thực hiện huấn luyện và so sánh các mô hình Machine Learning cổ điển (Classical ML: Logistic Regression, Multinomial Naive Bayes, Random Forest) trên 3 tập dữ liệu đã chuẩn bị:\n",
                "- **Exp 1 (Full):** Giữ nguyên toàn bộ dataset (173,219 mẫu train).\n",
                "- **Exp 2 (Dedup):** Loại bỏ hoàn toàn trùng lặp (132,783 mẫu train).\n",
                "- **Exp 3 (Capped K=3):** Giới hạn số lần lặp tối đa 3 lần (166,308 mẫu train).\n",
                "\n",
                "**ĐIỂM QUAN TRỌNG:** Cả 3 thí nghiệm đều được đánh giá trên **cùng một tập Test độc lập (28,432 mẫu)** và **cùng tập Validation (28,432 mẫu)** để đảm bảo tính công bằng và loại bỏ hoàn toàn Data Leakage."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import json\n",
                "import time\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.naive_bayes import MultinomialNB\n",
                "from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report\n",
                "\n",
                "pd.set_option('display.max_columns', 30)\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "\n",
                "PROCESSED_DIR = os.path.join('..', 'data', 'processed')\n",
                "if not os.path.exists(PROCESSED_DIR):\n",
                "    PROCESSED_DIR = os.path.join('data', 'processed')\n",
                "\n",
                "with open(os.path.join(PROCESSED_DIR, 'experiments_manifest.json'), 'r', encoding='utf-8') as f:\n",
                "    manifest = json.load(f)\n",
                "\n",
                "pd.DataFrame(manifest).T[['description', 'train_samples', 'val_samples', 'test_samples']]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Hàm huấn luyện và đánh giá chuẩn hóa\n",
                "Đánh giá đồng thời:\n",
                "- **Top-1 Accuracy:** Tỷ lệ dự đoán chính xác tuyệt đối ở vị trí đầu tiên.\n",
                "- **Top-3 Accuracy:** Bệnh thực tế nằm trong top 3 gợi ý có xác suất cao nhất của mô hình.\n",
                "- **Top-5 Accuracy:** Bệnh thực tế nằm trong top 5 gợi ý có xác suất cao nhất."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def evaluate_experiment(exp_id: str, model_type: str = 'nb'):\n",
                "    exp_path = os.path.join(PROCESSED_DIR, exp_id)\n",
                "    print(f'\\n========== ĐANG HUẤN LUYỆN: {exp_id} ({model_type.upper()}) ==========')\n",
                "    \n",
                "    # 1. Tải dữ liệu\n",
                "    t0 = time.time()\n",
                "    train_df = pd.read_csv(os.path.join(exp_path, 'train.csv'))\n",
                "    val_df = pd.read_csv(os.path.join(exp_path, 'val.csv'))\n",
                "    \n",
                "    feature_cols = [c for c in train_df.columns if c != 'diseases']\n",
                "    X_train = train_df[feature_cols].values\n",
                "    y_train = train_df['diseases'].values\n",
                "    X_val = val_df[feature_cols].values\n",
                "    y_val = val_df['diseases'].values\n",
                "    \n",
                "    # 2. Khởi tạo mô hình\n",
                "    if model_type == 'nb':\n",
                "        model = MultinomialNB(alpha=0.1)\n",
                "    elif model_type == 'lr':\n",
                "        model = LogisticRegression(max_iter=200, solver='saga', n_jobs=-1, random_state=42)\n",
                "    else:\n",
                "        raise ValueError(f'Unknown model: {model_type}')\n",
                "        \n",
                "    # 3. Fit mô hình\n",
                "    fit_t0 = time.time()\n",
                "    model.fit(X_train, y_train)\n",
                "    fit_time = time.time() - fit_t0\n",
                "    \n",
                "    # 4. Dự đoán xác suất trên Validation set\n",
                "    proba_val = model.predict_proba(X_val)\n",
                "    preds_val = model.predict(X_val)\n",
                "    \n",
                "    # 5. Tính metrics\n",
                "    top1 = accuracy_score(y_val, preds_val)\n",
                "    top3 = top_k_accuracy_score(y_val, proba_val, k=3, labels=model.classes_)\n",
                "    top5 = top_k_accuracy_score(y_val, proba_val, k=5, labels=model.classes_)\n",
                "    \n",
                "    res = {\n",
                "        'Experiment': exp_id,\n",
                "        'Model': model_type.upper(),\n",
                "        'Train Size': len(train_df),\n",
                "        'Fit Time (s)': round(fit_time, 2),\n",
                "        'Val Top-1 Acc': round(top1 * 100, 2),\n",
                "        'Val Top-3 Acc': round(top3 * 100, 2),\n",
                "        'Val Top-5 Acc': round(top5 * 100, 2)\n",
                "    }\n",
                "    print(f'Kết quả: Top-1: {res[\"Val Top-1 Acc\"]}%, Top-3: {res[\"Val Top-3 Acc\"]}%, Top-5: {res[\"Val Top-5 Acc\"]}% (Fit: {fit_time:.2f}s)')\n",
                "    return res\n",
                "\n",
                "# Chạy kiểm thử nhanh trên mô hình Multinomial Naive Bayes cho cả 3 thí nghiệm\n",
                "results = []\n",
                "for exp in ['exp1_full', 'exp2_dedup', 'exp3_capped']:\n",
                "    results.append(evaluate_experiment(exp, model_type='nb'))\n",
                "    \n",
                "res_df = pd.DataFrame(results)\n",
                "res_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Trực quan hóa so sánh hiệu năng 3 thí nghiệm"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, ax = plt.subplots(figsize=(10, 5))\n",
                "res_melted = res_df.melt(id_vars=['Experiment'], value_vars=['Val Top-1 Acc', 'Val Top-3 Acc', 'Val Top-5 Acc'], var_name='Metric', value_name='Accuracy (%)')\n",
                "sns.barplot(data=res_melted, x='Metric', y='Accuracy (%)', hue='Experiment', palette='Set2', ax=ax)\n",
                "plt.title('So sánh độ chính xác của 3 phương án xử lý trùng lặp trên tập Validation')\n",
                "plt.ylim(0, 100)\n",
                "for p in ax.patches:\n",
                "    if p.get_height() > 0:\n",
                "        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 1.5), ha='center', fontsize=9)\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/02_model_experiments.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Created notebooks/02_model_experiments.ipynb successfully!")

