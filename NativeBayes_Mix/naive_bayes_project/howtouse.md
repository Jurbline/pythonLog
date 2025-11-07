```text
# =========================
# 0) 进入项目根目录
# =========================
cd D:\Febbushi\naive_bayes_project_full_v7

# =========================
# 1) Conda 环境（若从未初始化过 Conda）
#    已经能用 conda 的话，跳过本小节
# =========================
# conda init powershell
# 重启 PowerShell 窗口后继续

# =========================
# 2) 创建并激活项目环境（已存在可跳过 create）
# =========================
conda create -n nbexp python=3.10 -y
conda activate nbexp

# 基础依赖（若项目有 requirements.txt，优先用 requirements.txt）
pip install -U pip
pip install pandas numpy scikit-learn matplotlib jieba tqdm pyspark requests

# =========================
# 3) 准备数据与输出目录
# =========================
# 3.1 创建输出目录（存在也不报错）
mkdir .\outputs -Force

# 3.2 从 UCI 准备短信垃圾分类数据为 CSV
#     (脚本会抓取/转换为两列：label,text)
python .\tools\prepare_uci_sms.py --out .\data\sms_uci.csv

# （可选）快速看前几行
# Get-Content .\data\sms_uci.csv -TotalCount 5

# =========================
# 4) 本地单机基线（scikit-learn）
# =========================
# 说明：用于论文里的“非分布式基线”部分
python .\nb_run.py --data .\data\sms_uci.csv --model cnb --alpha 0.5 --ngram 1 2

# 产物（在 nb_run.py 中配置的 fig_dir / 保存项）
# 可选：快速查看控制台分类报告

# =========================
# 5) 分布式 Spark（local[*] 多核）
# =========================
# 5.1 当次会话仅使用 JDK 21（不改系统默认 JDK）
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21"
$env:Path      = "$env:JAVA_HOME\bin;$env:Path"
java -version   # 应显示 21.x

# 5.2 让 PySpark 使用当前 Conda 的 Python
$env:PYSPARK_PYTHON        = (Get-Command python).Path
$env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON

# 5.3 JDK 21 兼容参数（注意：若以后切回 JDK 25，要清掉本行）
$env:PYSPARK_SUBMIT_ARGS = '--conf spark.driver.extraJavaOptions=-Djava.security.manager=allow --conf spark.executor.extraJavaOptions=-Djava.security.manager=allow pyspark-shell'

# 5.4 避免走 Hadoop/winutils 写盘分支（我们在代码里用 pandas 落盘，不需要它）
$env:HADOOP_HOME = $null

pip install pyspark -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5.5 运行 Spark 版（分布式实现展示用）
python .\distributed\spark_nb.py --data .\data\sms_uci.csv --lang en --master local[*] --out .\outputs\spark_local

# 生成成功后，预期会看到控制台打印：
# [OK] metrics: .\outputs\spark_local\metrics.csv
# [OK] confusion_matrix: .\outputs\spark_local\confusion_matrix.csv
# [OK] predictions: .\outputs\spark_local\predictions.csv

# =========================
# 6) 论文配图生成（混淆矩阵 PNG）
# =========================
python .\tools\plot_from_spark_outputs.py --out .\outputs\spark_local

# 生成：
# .\outputs\spark_local\figs\confusion_matrix.png
# 同时会写一份 metrics.txt（Accuracy/F1 摘要）

# =========================
# 7) 快速查看结果（可直接贴入论文/报告）
# =========================
Get-Content .\outputs\spark_local\metrics.csv
Import-Csv .\outputs\spark_local\confusion_matrix.csv | Format-Table
Import-Csv .\outputs\spark_local\predictions.csv      | Select-Object -First 5 | Format-Table

# =========================
# 8) 清理并重跑（可选）
# =========================
# Remove-Item .\outputs\spark_local -Recurse -Force
# python .\distributed\spark_nb.py --data .\data\sms_uci.csv --lang en --master local[*] --out .\outputs\spark_local
# python .\tools\plot_from_spark_outputs.py --out .\outputs\spark_local

# =========================
# 9) 恢复系统默认 JDK（通常是 25）
# =========================
# 关闭当前 PowerShell 窗口即可（本次设置只在当前会话生效）
```