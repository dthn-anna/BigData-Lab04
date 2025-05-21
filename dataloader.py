from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, explode, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, MapType, ArrayType
from pyspark.ml.linalg import DenseVector, Vectors
from pyspark.sql.functions import pandas_udf, PandasUDFType
import numpy as np
import  pandas as pd 
import json

from transforms import Transforms
from trainer import SparkConfig

# Khởi tạo SparkSession với Kafka package (nên đặt ở file chạy chính, không phải luôn trong dataloader)
scala_version = '2.12'
spark_version = '3.5.5'

packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
    'org.apache.kafka:kafka-clients:3.6.0'
]

spark = SparkSession.builder \
    .appName("Kafka-CIFAR-DataLoader") \
    .master("local[*]") \
    .config("spark.jars.packages", ",".join(packages)) \
    .getOrCreate()

sc = spark.sparkContext
sqlContext = spark._wrapped


class DataLoader:
    def __init__(self, 
                 spark: SparkSession,
                 sparkConf: SparkConfig, 
                 transforms: Transforms,
                 kafka_topic: str):
        self.spark = spark
        self.sparkConf = sparkConf
        self.transforms = transforms
        self.kafka_topic = kafka_topic

    def parse_stream(self):
        # Đọc stream từ Kafka topic
        df_raw = self.spark.readStream.format("kafka") \
            .option("kafka.bootstrap.servers", self.sparkConf.kafka_bootstrap_servers) \
            .option("subscribe", self.kafka_topic) \
            .option("startingOffsets", "latest") \
            .load()

        # Kafka message value là bytes, chuyển sang string
        df_str = df_raw.selectExpr("CAST(value AS STRING) as json_str")

        # JSON theo dạng:
        # {
        #   "0": {"feature-0": 123, ..., "label": 3},
        #   "1": {...}
        # }
        # Parse JSON thành MapType(StringType, MapType(StringType, Float/Int))
        schema = MapType(StringType(), MapType(StringType(), StringType()))
        df_json = df_str.select(from_json(col("json_str"), schema).alias("data"))

        # explode từng ảnh trong batch
        df_exploded = df_json.select(explode("data").alias("img_id", "features_map"))

        # UDF chuyển dict feature sang numpy array và label
        def parse_features(features_map):
            # features_map: dict như {"feature-0": "12", ..., "label": "3"}
            # Lọc các key "feature-*" theo index để tạo mảng pixel
            pixels = []
            label = None
            for k, v in features_map.items():
                if k.startswith("feature-"):
                    pixels.append(float(v))
                elif k == "label":
                    label = int(v)
            pixels_np = np.array(pixels).astype(np.uint8)
            # CIFAR dạng 3x32x32 (channels first), reshape:
            pixels_np = pixels_np.reshape(3, 32, 32).transpose(1, 2, 0)  # (H,W,C)
            return (pixels_np.tolist(), label)

        parse_udf = udf(parse_features, returnType=StructType([
            StructField("image",  # Mảng 3D list
                        ArrayType(ArrayType(ArrayType(IntegerType())))),
            StructField("label", IntegerType())
        ]))

        df_parsed = df_exploded.withColumn("parsed", parse_udf(col("features_map"))).select(
            col("parsed.image").alias("image"),
            col("parsed.label").alias("label")
        )

        # pandas UDF để áp dụng trên dataframe
        @pandas_udf("vector", PandasUDFType.SCALAR)
        def transform_udf(images_series):
            import numpy as np
            results = []
            for img_list in images_series:
                img_np = np.array(img_list, dtype=np.uint8)
                img_t = self.transforms.transform(img_np)
                vec = img_t.reshape(-1).tolist()
                results.append(DenseVector(vec))
            return pd.Series(results)

        # Áp dụng UDF
        # Ở đây để đơn giản, dùng UDF bình thường (không pandas)
        transform_udf_simple = udf(lambda img: DenseVector(self.transforms.transform(np.array(img, dtype=np.uint8)).reshape(-1).tolist()))

        df_final = df_parsed.withColumn("features", transform_udf_simple(col("image"))).select("features", "label")

        return df_final

