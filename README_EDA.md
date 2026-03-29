# 数据探索报告（EDA Report）查看说明

## 如何查看 EDA 报告

### 方式一：从 GitHub 拉取后直接打开（推荐）

1. 在项目目录下执行：`git pull origin main`
2. 用浏览器打开 **`poems_dataset_eda_report.html`**（双击或在浏览器中打开该文件即可，无需运行 Python）。

报告是静态 HTML，内嵌 SVG 图表，任何设备只要拉取了最新代码即可离线查看。

### 方式二：在本机重新生成报告

若没有拉取到最新的 `poems_dataset_eda_report.html`，只要手上有：

- **`poems_dataset_eda.py`**（脚本）
- **`poems_dataset_merged.csv`**（合并后的数据，须与脚本同目录或修改脚本中的路径）

在本机执行：

```bash
python3 poems_dataset_eda.py
```

会在当前目录生成 **`poems_dataset_eda_report.html`**，再用浏览器打开即可。  
脚本仅使用 Python 标准库，无需安装 pandas、matplotlib。

### 方式三：通过共享文件查看

若无法访问 GitHub，可让已有报告的成员通过 Google Drive / 邮件等把 **`poems_dataset_eda_report.html`** 发给你，用浏览器打开即可，无需安装任何环境。

---

**总结**：其他人只要 **从 GitHub 拉取最新代码**，即可在本地用浏览器打开 `poems_dataset_eda_report.html` 查看；若不拉取，则需拿到该 HTML 文件或在本机用脚本 + 合并 CSV 重新生成。
