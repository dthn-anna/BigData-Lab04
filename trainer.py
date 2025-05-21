import pyspark

from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT
from transforms import Transforms

class SparkConfig:
    appName = "CIFAR"
    receivers = 4
    host = "local"
    stream_host = "localhost"
    port = 6100
    batch_interval = 2 # Khoảng thời gian mỗi batch trong Spark Streaming

from dataloader import DataLoader

class Trainer:
    def __init__(self, 
                 model, 
                 split:str, 
                 spark_config:SparkConfig, 
                 transforms: Transforms) -> None:

        self.model = model
        self.split = split
        self.sparkConf = spark_config
        self.transforms = transforms
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]",f"{self.sparkConf.appName}")
        self.ssc = StreamingContext(self.sc,self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc,self.ssc,self.sqlContext,self.sparkConf,self.transforms)

    def train(self):
        # Gọi parse_stream() từ DataLoader để nhận dòng dữ liệu liên tục từ TCP socket.
        stream = self.dataloader.parse_stream()
        # Với mỗi batch (RDD) nhận được từ Spark Streaming, gọi hàm __train__.
        stream.foreachRDD(self.__train__)

        # Bắt đầu chạy Spark Streaming.
        self.ssc.start()
        # Chờ cho đến khi dừng bằng tay (Ctrl+C hoặc lỗi).
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        # Chỉ xử lý nếu batch đó có dữ liệu.
        if not rdd.isEmpty():
            # mỗi dòng có image là vector (đã được Transforms xử lý) và label là số nguyên.
            schema = StructType([
                StructField("image", VectorUDT(), True),
                StructField("label", IntegerType(), True)])

            # Chuyển RDD thành Spark DataFrame để dùng cho huấn luyện mô hình.
            df = self.sqlContext.createDataFrame(rdd, schema)
            
            predictions, accuracy, precision, recall, f1 = self.model.train(df)

            print("="*10)
            print(f"Predictions = {predictions}")
            print(f"Accuracy = {accuracy}")
            print(f"Precision = {precision}")
            print(f"Recall = {recall}")
            print(f"F1 Score = {f1}")
            print("="*10)
        
        print("Total Batch Size of RDD Received :",rdd.count())
        print("+"*20)

    # def predict(self):
    #     stream = self.dataloader.parse_stream()
    #     stream.foreachRDD(self.__predict__)

    #     self.ssc.start()
    #     self.ssc.awaitTermination()

    # def __predict__(self, rdd: pyspark.RDD) -> DataFrame:     
    #     if not rdd.isEmpty():
    #         schema = StructType([
    #             StructField(name="image", dataType=VectorUDT(), nullable=True),
    #             StructField(name="label",dataType=IntegerType(),nullable=True)])
            
    #         df = self.sqlContext.createDataFrame(rdd, schema)
            
    #         accuracy, loss, precision, recall, f1, cm = self.model.predict(df, self.raw_model)
    #         self.cm += cm
    #         self.test_accuracy += accuracy/total_batches
    #         self.test_loss += loss/total_batches
    #         self.test_precision += precision/total_batches
    #         self.test_recall += recall/total_batches
    #         self.test_f1 += f1/total_batches
    #         print(f"Test Accuracy : ", self.test_accuracy)
    #         print(f"Test Loss : ",self.test_loss)
    #         print(f"Test Precision :",self.test_precision)
    #         print(f"Test Recall : ", self.test_recall)
    #         print(f"Test F1 Score: ",self.test_f1)
    #         print(f"Confusion matrix: \n", self.cm)

    #     print(f"batch: {self.batch_count}")
    #     print("Total Batch Size of RDD Received :",rdd.count())
    #     print("---------------------------------------")   
        
""" 
+----------------------+      +---------------------+        +--------------------+
|  CIFAR Dataset (TCP) | ---> |  Spark Streaming    | --->   |  Model Trainer     |
|  (server socket)     |      |  (RDD mỗi 2 giây)   |        |  (train(), predict)|
+----------------------+      +---------------------+        +--------------------+
         |                               |                             |
         |-- JSON {"image":..., "label"} |                             |
                                         V                             |
                               RDD ---> DataFrame ---> model.train(df) |
                                         ^                             |
                                         |                             |                         |
                                    Transforms (normalize, resize...)  |



""" 