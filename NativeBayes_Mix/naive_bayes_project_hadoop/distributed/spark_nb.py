# -*- coding: utf-8 -*-
import argparse, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, IndexToString
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def build_spark(master):
    return (SparkSession.builder.appName("SparkNaiveBayesText").master(master).getOrCreate())

def jieba_udf():
    import jieba
    def cut(s):
        if s is None: return []
        return [w for w in jieba.lcut(s) if w.strip()]
    return udf(cut, ArrayType(StringType()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./outputs/spark")
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--lang", choices=["en","cn"], default="en")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--num_features", type=int, default=20000)
    ap.add_argument("--min_df", type=float, default=2.0)
    args = ap.parse_args()

    spark = build_spark(args.master); sc = spark.sparkContext; sc.setLogLevel("WARN")
    df = (spark.read.option("header", True).option("multiLine", True).csv(args.data)
          .select("label","text").na.drop(subset=["label","text"]))
    df = df.withColumn("text", lower(regexp_replace(col("text"), r"\s+", " ")))

    if args.lang == "en":
        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern=r"\W+")
        tokenized = tokenizer.transform(df)
        remover = StopWordsRemover(inputCol="tokens", outputCol="tokens_clean")
        tokenized = remover.transform(tokenized)
    else:
        cut = jieba_udf()
        tokenized = df.withColumn("tokens_clean", cut(col("text")))

    hashing = HashingTF(inputCol="tokens_clean", outputCol="tf", numFeatures=args.num_features)
    idf = IDF(inputCol="tf", outputCol="features", minDocFreq=int(args.min_df) if args.min_df >= 1 else 0)
    lbl_indexer = StringIndexer(inputCol="label", outputCol="label_idx", handleInvalid="skip")
    nb = NaiveBayes(featuresCol="features", labelCol="label_idx", modelType="multinomial", smoothing=args.alpha)
    pipeline = Pipeline(stages=[lbl_indexer, hashing, idf, nb])

    train, test = tokenized.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train)
    preds = model.transform(test).cache()

    acc = MulticlassClassificationEvaluator(labelCol="label_idx", predictionCol="prediction", metricName="accuracy").evaluate(preds)
    f1  = MulticlassClassificationEvaluator(labelCol="label_idx", predictionCol="prediction", metricName="f1").evaluate(preds)

    labels = model.stages[0].labels
    itos = IndexToString(inputCol="prediction", outputCol="pred_label", labels=labels)
    preds2 = itos.transform(preds)

    cm = (preds2.groupBy("label","pred_label").count().orderBy("label","pred_label"))

    out = args.out.rstrip("/")
    preds2.select("label","text","pred_label").coalesce(1).write.mode("overwrite").option("header", True).csv(out + "/predictions")
    cm.coalesce(1).write.mode("overwrite").option("header", True).csv(out + "/confusion_matrix")
    spark.createDataFrame([(float(acc), float(f1))], ["accuracy","f1"]).coalesce(1).write.mode("overwrite").option("header", True).csv(out + "/metrics_csv")
    print(f"[OK] metrics: {out}/metrics_csv"); print(f"[OK] confusion_matrix: {out}/confusion_matrix"); print(f"[OK] predictions: {out}/predictions")
    spark.stop()

if __name__ == "__main__":
    main()
