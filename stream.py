import time
import json
import pickle 
import socket 
import argparse 
import numpy as np
from tqdm import tqdm
import os
from kafka import KafkaProducer 


parser = argparse.ArgumentParser(description='Streams a file to a Kafka')
parser.add_argument('--folder', '-f', help='Data folder', required=True, type=str)
parser.add_argument('--batch-size', '-b', help='Batch size', required=True, type=int)
parser.add_argument('--endless', '-e', help='Enable endless stream',required=False, type=bool, default=False)
parser.add_argument('--split','-s', help="training or test split", required=False, type=str, default='train')
parser.add_argument('--sleep','-t', help="streaming interval", required=False, type=int, default=3)
parser.add_argument('--kafka-server', help="Kafka broker (e.g. localhost:9092)", required=False, type=str, default='localhost:9092')
parser.add_argument('--kafka-topic', help="Kafka topic name", required=False, type=str, default='cifar-images')


class Dataset:
    def __init__(self) -> None:
        self.data = []
        self.labels = []

  
    def data_generator(self, data_file: str, batch_size: int):
        batch = []
        with open(data_file, "rb") as batch_file:
            batch_data = pickle.load(batch_file, encoding='bytes')
            self.data.append(batch_data[b'data'])
            self.labels.extend(batch_data[b'labels'])


        data = np.vstack(self.data)
        self.data = list(map(np.ndarray.tolist, data))

        size_per_batch = (len(self.data) // batch_size) * batch_size
        for ix in range(0, size_per_batch, batch_size):
            image = self.data[ix:ix+batch_size]
            label = self.labels[ix:ix+batch_size]
            batch.append([image, label])

        self.data = self.data[ix+batch_size:]
        self.labels = self.labels[ix+batch_size:]
        
        return batch


    def sendCIFARBatchFileToKafka(self, producer: KafkaProducer,  topic: str, input_batch_file, batch_size, split="train"):
 
        if split == "train":
            total_batch = 50_000 / batch_size + 1
        else:
            total_batch = 10_000 / batch_size + 1

        pbar = tqdm(total_batch)


        data_sent = 0
        for file in input_batch_file:
            batches = self.data_generator(file, batch_size)
            
            for batch in batches:
                images, labels = batch
                images = np.array(images)
                images = images.reshape(images.shape[0], -1)
                batch_size, feature_size = images.shape 
                images = images.tolist()

                payload = dict()
            
                for batch_idx in range(batch_size): 
                    payload[batch_idx] = dict()

                   
                    for feature_idx in range(feature_size):
                        payload[batch_idx][f'feature-{feature_idx}'] = images[batch_idx][feature_idx]
                    payload[batch_idx]['label'] = labels[batch_idx]

                
                try:
                    message = json.dumps(payload).encode('utf-8')
                    producer.send(topic, message)
        
                except Exception as error_message: 
                    print(f"Failed to send message: {error_message}")

                data_sent += 1
                pbar.update(n=1)
                pbar.set_description(f"it: {data_sent} | received : {batch_size} images")
                time.sleep(sleep_time)

    def streamCIFARDataset(self, producer, topic, folder, batch_size):
        CIFAR_BATCHES = [
            os.path.join(folder, 'data_batch_1'),
            os.path.join(folder, 'data_batch_2'),
            os.path.join(folder, 'data_batch_3'),
            os.path.join(folder, 'data_batch_4'),
            os.path.join(folder, 'data_batch_5'),
            os.path.join(folder, 'test_batch'),
        ]
       
        CIFAR_BATCHES = CIFAR_BATCHES[:-1] if train_test_split=='train' else [CIFAR_BATCHES[-1]]
        self.sendCIFARBatchFileToKafka(producer, topic, CIFAR_BATCHES, batch_size, train_test_split) 
        # Mở từng  file  CIFAR_BATCHES --> đọc image và label --> Chia thành các batch --> Chuyển thành dạng json --> gửi từng batch qua socket 'tcp_connection' đến Spark Streaming để xử lý 
        

if __name__ == '__main__':
    args = parser.parse_args()

    data_folder = args.folder
    batch_size = args.batch_size
    endless = args.endless
    sleep_time = args.sleep
    train_test_split = args.split
    kafka_server = args.kafka_server
    kafka_topic = args.kafka_topic


    # Khởi tạo Kafka Producer
    producer = KafkaProducer(
        bootstrap_servers=kafka_server,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        retries=5
    )



    dataset = Dataset()

    
    if endless:
        while True:
            dataset.streamCIFARDataset(producer, kafka_topic, data_folder, batch_size)
    else:
        dataset.streamCIFARDataset(producer, kafka_topic, data_folder, batch_size)

    producer.flush()
    producer.close()