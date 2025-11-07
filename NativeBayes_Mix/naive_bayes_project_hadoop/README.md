# 朴素贝叶斯文本分类（单机 + 分布式 Spark）— 完整项目

- 单机：`nb_run.py`（jieba→TF‑IDF→NB，支持 MNB/CNB/BNB，α 网格、混淆矩阵、误判导出）  
- 分布式：`distributed/spark_nb.py`（HashingTF+IDF+NaiveBayes，支持 local[*]/Standalone/YARN+HDFS）  
- 数据工具：UCI/THUCNews/CLUE TNEWS 转 CSV（`tools/*`）

## 目录
```
naive_bayes_project_full_v7/
├─ nb_run.py
├─ requirements.txt
├─ README.md
├─ data/
│  └─ sms_sample.csv
├─ resources/
│  └─ stopwords.txt
├─ figs/            # 运行后生成图
├─ outputs/         # 运行后生成报告/CSV
├─ tools/
│  ├─ prepare_uci_sms.py
│  ├─ convert_thucnews_to_csv.py
│  └─ convert_clue_tnews_to_csv.py
└─ distributed/
   └─ spark_nb.py
```

---

## 0）环境准备（Windows + PowerShell）
```powershell
conda init powershell   # 首次需要，完毕后请重开 PowerShell
conda create -n nbexp python=3.12 -y
conda activate nbexp
pip install -r requirements.txt
```
> 运行 PySpark 需 Java 8/11 并设置 `JAVA_HOME`。

---

## 1）数据准备（三选一，或都做）

### A. UCI SMS Spam Collection
```powershell
# 在线下载并转换为 CSV（label,text）
python .\tools\prepare_uci_sms.py --out .\data\sms_uci.csv

# 离线：已有 zip
python .\tools\prepare_uci_sms.py --url file:///D:/downloads/smsspamcollection.zip --out .\data\sms_uci.csv
```

### B. THUCNews → CSV
```powershell
python .\tools\convert_thucnews_to_csv.py --root D:\datasets\THUCNews --out .\data\thucnews.csv

# 指定类别+条数
python .\tools\convert_thucnews_to_csv.py --root D:\datasets\THUCNews --out .\data\thucnews_part.csv ^
  --classes 体育,财经,科技,娱乐 --limit_per_class 5000
```

### C. CLUE TNEWS → CSV
```powershell
# 合并 train+dev（可含 test；test 无标签填 unknown）
python .\tools\convert_clue_tnews_to_csv.py --dir D:\datasets\CLUE\tnews_public --out .\data\tnews.csv --include_test

# 或分别导出 train/dev/test
python .\tools\convert_clue_tnews_to_csv.py --dir D:\datasets\CLUE\tnews_public --out_dir .\data\tnews\ --include_test
```

---

## 2）单机（scikit‑learn）训练与评估
**推荐：ComplementNB (`--model cnb`)、`--alpha 0.5`、`--ngram 1 2`**
```powershell
mkdir .\outputs -Force

# UCI SMS
python .\nb_run.py --data .\data\sms_uci.csv --model cnb --alpha 0.5 --ngram 1 2 ^
  --save_report .\outputs\report.txt ^
  --save_pred .\outputs\preds.csv ^
  --save_mis .\outputs\mis.csv

# THUCNews 多分类（词表加大）
python .\nb_run.py --data .\data\thucnews.csv --model cnb --alpha 0.5 --ngram 1 2 ^
  --max_features 50000 --min_df 2 ^
  --save_report .\outputs\thuc_report.txt ^
  --save_pred .\outputs\thuc_preds.csv ^
  --save_mis .\outputs\thuc_mis.csv

# TNEWS
python .\nb_run.py --data .\data\tnews.csv --model cnb --alpha 0.5 --ngram 1 2 ^
  --max_features 50000 --min_df 2 ^
  --save_report .\outputs\tnews_report.txt ^
  --save_pred .\outputs\tnews_preds.csv ^
  --save_mis .\outputs\tnews_mis.csv
```

**查看产物**
```powershell
dir .\figs
type .\outputs\report.txt
Import-Csv .\outputs\mis.csv | Select-Object -First 5 | Format-Table -AutoSize
```

---

## 3）分布式（Spark）
**本机多核（local[*]）**
```powershell
pip install pyspark

# UCI
python .\distributed\spark_nb.py --data .\data\sms_uci.csv --lang en --master local[*] --out .\outputs\spark_local

# THUCNews
python .\distributed\spark_nb.py --data .\data\thucnews.csv --lang cn --master local[*] --out .\outputs\spark_local_thuc
```

**Standalone 集群（示例）**
```powershell
spark-submit --master spark://HOST:7077 distributed/spark_nb.py ^
  --data file:///D:/your_path/naive_bayes_project_full_v7/data/sms_uci.csv ^
  --lang en ^
  --out file:///D:/your_path/naive_bayes_project_full_v7/outputs/spark_cluster
```

**YARN + HDFS（Linux/Bash）**
```bash
hdfs dfs -mkdir -p /nb/data
hdfs dfs -put data/sms_uci.csv /nb/data/
spark-submit --master yarn --deploy-mode client   distributed/spark_nb.py   --data hdfs:///nb/data/sms_uci.csv   --lang en   --out hdfs:///nb/out/spark_nb
```

---

## FAQ
- Conda 激活失败 → `conda init powershell` 后**重开终端**；或 `& "C:\ProgramData\anaconda3\condabin\conda.bat" activate nbexp`  
- `stop_words` 报错 → 仅接受 **list/'english'/None**（项目已兼容）  
- `jieba` 警告 → 无害；可选 `pip install "setuptools<81"`  
- Spark/Java 报错 → 安装 Java 8/11，设置 `JAVA_HOME` 并加入 `PATH`  
- HDFS 读不到 → 用 `hdfs:///` 前缀，检查权限与 NameNode；建议 `--deploy-mode client`
