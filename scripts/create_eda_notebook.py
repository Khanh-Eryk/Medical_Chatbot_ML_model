import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 01 — Data Audit & Exploratory Data Analysis (EDA)\n",
                "**Dự án:** Medical Chatbot — Phân loại bệnh dựa trên triệu chứng (ML Core)\n",
                "\n",
                "Notebook này phục vụ nghiên cứu khám phá, khảo sát dữ liệu thô và trực quan hóa phân phối:\n",
                "1. Khảo sát cấu trúc dataset (`Disease and symptoms dataset.csv`).\n",
                "2. Kiểm tra tính toàn vẹn (Missing values, Data types, Range).\n",
                "3. Phân tích phân phối cột nhãn `diseases` (Imbalance, Long-tail distribution).\n",
                "4. Phân tích độ thưa của triệu chứng và phát hiện 49 cột triệu chứng có tổng bằng 0.\n",
                "5. Phát hiện các bản ghi trùng lặp hoàn toàn (Exact Duplicates) và xung đột nhãn (Label Conflicts).\n",
                "6. Trực quan hóa các phân phối chính."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Cấu hình hiển thị\n",
                "pd.set_option('display.max_columns', 50)\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (12, 6)\n",
                "plt.rcParams['font.size'] = 11\n",
                "\n",
                "DATA_PATH = os.path.join('..', 'Disease and symptoms dataset.csv')\n",
                "if not os.path.exists(DATA_PATH):\n",
                "    DATA_PATH = 'Disease and symptoms dataset.csv'\n",
                "\n",
                "print(f'Đường dẫn file dữ liệu: {DATA_PATH}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Tải và kiểm tra tổng quan cấu trúc dữ liệu"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_csv(DATA_PATH, low_memory=False)\n",
                "print(f'Số dòng (Records): {len(df):,}')\n",
                "print(f'Số cột (Columns): {len(df.columns):,}')\n",
                "print(f'Số ô missing (null): {df.isnull().sum().sum()}')\n",
                "df.head(5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Phân tích cột mục tiêu (Target: `diseases`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "disease_counts = df['diseases'].value_counts()\n",
                "print(f'Tổng số bệnh duy nhất: {len(disease_counts)}')\n",
                "print(f'Bệnh có nhiều mẫu nhất: {disease_counts.index[0]} ({disease_counts.iloc[0]} mẫu)')\n",
                "print(f'Bệnh có ít mẫu nhất: {disease_counts.index[-1]} ({disease_counts.iloc[-1]} mẫu)')\n",
                "print(f'Số bệnh có đúng 1 mẫu duy nhất: {(disease_counts == 1).sum()}')\n",
                "print(f'Số bệnh có <= 5 mẫu: {(disease_counts <= 5).sum()}')\n",
                "print(f'Số bệnh có < 10 mẫu: {(disease_counts < 10).sum()}')\n",
                "\n",
                "# Trực quan hóa Top 20 bệnh phổ biến nhất\n",
                "plt.figure(figsize=(12, 6))\n",
                "sns.barplot(x=disease_counts.head(20).values, y=disease_counts.head(20).index, hue=disease_counts.head(20).index, palette='Blues_r', legend=False)\n",
                "plt.title('Top 20 bệnh có số lượng mẫu nhiều nhất trong dataset')\n",
                "plt.xlabel('Số lượng mẫu')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Phân tích đặc trưng triệu chứng và độ thưa (Sparsity)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature_cols = [c for c in df.columns if c != 'diseases']\n",
                "features_df = df[feature_cols]\n",
                "\n",
                "# Thống kê số triệu chứng trên từng mẫu\n",
                "symptom_counts_per_sample = features_df.sum(axis=1)\n",
                "print('Thống kê số lượng triệu chứng mỗi bệnh nhân:')\n",
                "print(symptom_counts_per_sample.describe())\n",
                "\n",
                "# Kiểm tra 49 cột triệu chứng có tổng bằng 0\n",
                "col_sums = features_df.sum(axis=0)\n",
                "zero_sum_cols = col_sums[col_sums == 0].index.tolist()\n",
                "print(f'\\nSố cột triệu chứng không bao giờ xuất hiện (sum == 0): {len(zero_sum_cols)}')\n",
                "\n",
                "plt.figure(figsize=(10, 4))\n",
                "sns.histplot(symptom_counts_per_sample, bins=12, discrete=True, color='teal')\n",
                "plt.title('Phân phối số lượng triệu chứng trên mỗi bản ghi')\n",
                "plt.xlabel('Số triệu chứng')\n",
                "plt.ylabel('Số bệnh nhân')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Phát hiện trùng lặp hoàn toàn (Exact Duplicates) & Xung đột nhãn (Label Conflicts)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Trùng lặp hoàn toàn\n",
                "exact_dups = int(df.duplicated().sum())\n",
                "print(f'Số dòng trùng lặp 100% (cả symptoms và disease): {exact_dups:,} ({exact_dups/len(df)*100:.2f}%)')\n",
                "\n",
                "# 2. Xung đột nhãn (Cùng tập triệu chứng nhưng khác bệnh)\n",
                "grouped = df.groupby(feature_cols)['diseases'].nunique()\n",
                "conflicts = int((grouped > 1).sum())\n",
                "print(f'Số vector triệu chứng bị gán cho >= 2 bệnh: {conflicts:,}')"
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

with open("notebooks/01_data_audit_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Created notebooks/01_data_audit_eda.ipynb successfully!")

