import os
import shutil
import random

def split_files(input_folder, output_folder, train_ratio, test_ratio, val_ratio):
    # 获取所有的行为编号文件夹
    behavior_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

    # 遍历每个行为编号文件夹，将其中的视频文件随机分配到train, test和val文件夹中
    for behavior_folder in behavior_folders:
        # 获取行为编号文件夹中的所有视频文件
        behavior_folder_path = os.path.join(input_folder, behavior_folder)
        video_files = [file for file in os.listdir(behavior_folder_path) if os.path.isfile(os.path.join(behavior_folder_path, file)) and file.endswith('.avi')]

        # 计算每个文件夹中应该包含的视频数量
        num_videos = len(video_files)
        num_train_videos = int(num_videos * train_ratio)
        num_test_videos = int(num_videos * test_ratio)
        num_val_videos = num_videos - num_train_videos - num_test_videos

        # 创建train, test和val文件夹
        train_folder = os.path.join(output_folder, 'train', behavior_folder)
        test_folder = os.path.join(output_folder, 'test', behavior_folder)
        val_folder = os.path.join(output_folder, 'val', behavior_folder)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        # 随机选择视频文件，并将其复制到对应的文件夹中
        for i, video_file in enumerate(video_files):
            src_file_path = os.path.join(behavior_folder_path, video_file)

            if i < num_train_videos:
                dst_folder = train_folder
            elif i < num_train_videos + num_test_videos:
                dst_folder = test_folder
            else:
                dst_folder = val_folder

            dst_file_path = os.path.join(dst_folder, video_file)
            shutil.copyfile(src_file_path, dst_file_path)

if __name__ == '__main__':
    input_folder = 'UCF-101'
    output_folder = 'UCF-101_splited'

    split_files(input_folder, output_folder, 0.7, 0.2, 0.1)
