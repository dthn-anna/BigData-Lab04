import pyspark

from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT
from transforms import Transforms
from pyspark.sql import SparkSession


# config.py
class SparkConfig:
    appName = "CIFAR"
    receivers = 4
    host = "local"
    stream_host = "localhost"
    port = 9092
    batch_interval = 2
    kafka_bootstrap_servers = "localhost:9092"


from dataloader import DataLoader

class Trainer:
    def __init__(self, 
                 model, 
                 split: str, 
                 spark_config: SparkConfig, 
                 transforms: Transforms,
                 spark: SparkSession) -> None:

        self.model = model
        self.split = split
        self.sparkConf = spark_config
        self.transforms = transforms

        # Dùng lại SparkContext và SQLContext từ SparkSession
        self.spark = spark
        self.sc = spark.sparkContext
        self.sqlContext = SQLContext(self.sc)

        self.dataloader = DataLoader(spark=self.spark,
                                    sparkConf=self.sparkConf,
                                    transforms=self.transforms,
                                    kafka_topic= "cifar-images")


    def train(self):
    
        stream_df = self.dataloader.parse_stream()

        def process_batch(df, batch_id):
            if not df.rdd.isEmpty():
                # df đã có schema với "features" VectorUDT, "label" IntegerType
                predictions, accuracy, precision, recall, f1 = self.model.train(df)

                print("="*20)
                print(f"Batch {batch_id} results:")
                print(f"Predictions: {predictions}")
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")
                print("="*20)
            else:
                print(f"Batch {batch_id} is empty.")

        query = stream_df.writeStream \
                         .outputMode("append") \
                         .foreachBatch(process_batch) \
                         .start()

        query.awaitTermination()