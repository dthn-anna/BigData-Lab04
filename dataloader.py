from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, explode, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, MapType, ArrayType
from pyspark.ml.linalg import DenseVector, Vectors
from pyspark.sql.functions import pandas_udf, PandasUDFType
import numpy as np
import  pandas as pd 
from pyspark.ml.linalg import VectorUDT

from transforms import Transforms
from trainer import SparkConfig

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
sqlContext = spark


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

        # Tạo biến broadcast để truyền transforms xuống các executor tránh lỗi pickle
        self.bc_transforms = self.spark.sparkContext.broadcast(transforms)

    def parse_stream(self):

        df_raw = self.spark.readStream.format("kafka") \
            .option("kafka.bootstrap.servers", self.sparkConf.kafka_bootstrap_servers) \
            .option("subscribe", self.kafka_topic) \
            .option("startingOffsets", "latest") \
            .load()

        df_str = df_raw.selectExpr("CAST(value AS STRING) as json_str")

        schema = MapType(StringType(), MapType(StringType(), StringType()))
        df_json = df_str.select(from_json(col("json_str"), schema).alias("data"))

        df_exploded = df_json.select(explode("data").alias("img_id", "features_map"))

        def parse_features(features_map):
            pixels = []
            label = None
            for k, v in features_map.items():
                if k.startswith("feature-"):
                    pixels.append(float(v))
                elif k == "label":
                    label = int(v)
            pixels_np = np.array(pixels).astype(np.uint8)
            pixels_np = pixels_np.reshape(3, 32, 32).transpose(1, 2, 0)  # (H,W,C)
            return (pixels_np.tolist(), label)

        parse_udf = udf(parse_features, returnType=StructType([
            StructField("image", ArrayType(ArrayType(ArrayType(IntegerType())))),
            StructField("label", IntegerType())
        ]))

        df_parsed = df_exploded.withColumn("parsed", parse_udf(col("features_map"))).select(
            col("parsed.image").alias("image"),
            col("parsed.label").alias("label")
        )

        # Định nghĩa UDF không dùng lambda, lấy biến broadcast
        def transform_image_udf(img):
            
            img_np = np.array(img, dtype=np.uint8)
            img_t = Transforms.transform(img_np)   # Gọi phương thức transform từ đối tượng broadcast
            vec = img_t.reshape(-1).tolist()
            return DenseVector(vec)

        transform_udf_simple = udf(transform_image_udf, VectorUDT())

        df_final = df_parsed.withColumn("features", transform_udf_simple(col("image"))).select("features", "label")

        return df_final