import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
import gc
from tqdm import tqdm


def video_generator(data_dir, target_size, batch_size):
    # 获取数据文件夹中所有的子文件夹（即分类标签）
    classes = os.listdir(data_dir)

    while True:
        # 创建一个空的numpy数组，用于存储一批视频帧
        batch = np.zeros((batch_size, target_size[0], target_size[1], target_size[2], 3))
        labels = []

        # 循环读取视频帧
        for i in range(batch_size):
            # 随机选择一个分类
            class_idx = np.random.randint(len(classes))
            class_name = classes[class_idx]

            # 获取该分类中所有视频文件的路径
            class_path = os.path.join(data_dir, class_name)
            video_paths = os.listdir(class_path)

            # 随机选择一个视频文件
            video_idx = np.random.randint(len(video_paths))
            video_path = os.path.join(class_path, video_paths[video_idx])

            # 打开视频文件
            cap = cv2.VideoCapture(video_path)

            # 获取视频分辨率和帧率
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # filter out too short videos (less than 1 unit)
            while frames < target_size[0]:
                cap.release()
                # 随机选择一个视频文件
                video_idx = np.random.randint(len(video_paths))
                video_path = os.path.join(class_path, video_paths[video_idx])
                # 打开视频文件
                cap = cv2.VideoCapture(video_path)

                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 打开视频文件
            # print(video_path)
            cap = cv2.VideoCapture(video_path)

            # 随机选择起始帧索引
            start_idx = np.random.randint(frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            
            # 循环读取视频帧
            for j in range(target_size[0]):
                # 读取视频帧
                ret, frame = cap.read()

                # 如果读取失败，则重置视频文件
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    print(ret)
                
                # 将帧调整为目标大小并添加到批次中
                frame_resized = cv2.resize(frame, [target_size[2], target_size[1]])
                batch[i, j, :, :, :] = frame_resized

            # 释放视频文件的资源
            cap.release()
            del cap
            gc.collect()
            
            # 添加对应的分类标签
            labels.append(class_idx)
            #labels.append(class_name)
        categorical_label = to_categorical(labels, num_classes=len(classes))
        # 将批次返回给Keras模型
        yield batch / 255., categorical_label

if __name__ == "__main__":
    data_dir = 'seizure_dataset/mar_16_splited/train'
    target_size = [15, 240, 360]
    # l, w, h  
    batch_size = 12
    test_gen = video_generator(data_dir, target_size, batch_size)
    for i in tqdm (range (2), desc="Loading..."):

        temp = next(test_gen)
        print(temp[1])
        
