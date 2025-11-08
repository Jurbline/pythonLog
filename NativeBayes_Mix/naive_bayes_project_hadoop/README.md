# 朴素贝叶斯文本分类（单机 + 分布式 Spark）

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

## Linus + Bash

### (JDK17 + Spark 4.0.1 + Hadoop 3.4.0 + Python 3)

## 0) 环境准备

```bash
cd ~
python3 -m venv nbexp
source ~/nbexp/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

> 运行 PySpark 需 Java 8/11 并设置 `JAVA_HOME`。

```bash
cd ~
mv naive_bayes_project_hadoop naive_bayes_project
cd ~/naive_bayes_project
```

## 1) 数据准备

### UCI SMS Spam Collection

```bash
mkdir -p data outputs
python tools/prepare_uci_sms.py --out data/sms_uci.csv
```

### 上传到 HDFS

```bash
hdfs dfs -mkdir -p /user/$USER/data
hdfs dfs -put -f data/sms_uci.csv /user/$USER/data/
hdfs dfs -ls /user/$USER/data
```

## 2) HDFS 训练与评估

### 运行Spark on YARN

```bash
$SPARK_HOME/bin/spark-submit \
  --master yarn \
  --deploy-mode client \
  --conf spark.ui.port=4040 \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=python3 \
  --conf spark.executorEnv.PYSPARK_PYTHON=python3 \
  distributed/spark_nb.py \
    --data hdfs:///user/$USER/data/sms_uci.csv \
    --lang en \
    --master yarn \
    --out hdfs:///user/$USER/outputs/spark_nb
```

## 3) 查看结果

```bash
hdfs dfs -ls /user/$USER/outputs/spark_nb
hdfs dfs -get -f /user/$USER/outputs/spark_nb ./outputs/spark_nb_hdfs_copy
```

---

## Windows + PowerShell

## 0）环境准备

```powershell
conda init powershell   # 首次需要，完毕后请重开 PowerShell
conda create -n nbexp python=3.12 -y
conda activate nbexp
pip install -r requirements.txt
```

> 运行 PySpark 需 Java 8/11 并设置 `JAVA_HOME`。

---

## 1）数据准备

### UCI SMS Spam Collection

```powershell
# 在线下载并转换为 CSV（label,text）
python .\tools\prepare_uci_sms.py --out .\data\sms_uci.csv

# 离线：已有 zip
python .\tools\prepare_uci_sms.py --url file:///D:/downloads/smsspamcollection.zip --out .\data\sms_uci.csv
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

- Conda 激活失败 → `conda init powershell` 后**重开终端**
  ；或 `& "C:\ProgramData\anaconda3\condabin\conda.bat" activate nbexp`
- `stop_words` 报错 → 仅接受 **list/'english'/None**（项目已兼容）
- `jieba` 警告 → 无害；可选 `pip install "setuptools<81"`
- Spark/Java 报错 → 安装 Java 8/11，设置 `JAVA_HOME` 并加入 `PATH`
- HDFS 读不到 → 用 `hdfs:///` 前缀，检查权限与 NameNode；建议 `--deploy-mode client`
