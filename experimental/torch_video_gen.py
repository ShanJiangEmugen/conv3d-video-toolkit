import os
import random
import cv2
import numpy as np
import torch

class VideoGenerator(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.class_names = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.class_names)}
        self.samples = self._make_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, target = self.samples[index]
        frames = self._load_frames(video_path)
        if self.transform is not None:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return frames, target

    def _make_dataset(self):
        samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            for video_name in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_name)
                samples.append((video_path, self.class_to_idx[class_name]))
        random.shuffle(samples)  # 随机打乱数据集顺序
        return samples

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = random.randint(0, max(0, frame_count - 15))
        frames = []
        for i in range(start_frame, start_frame + 15):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) < 15:
            # 如果视频帧数不足15帧，使用最后一帧来填充
            last_frame = frames[-1]
            while len(frames) < 15:
                frames.append(last_frame)
        return frames


if __name__ == '__main__':
    data_dir = 'MIT_data/train'
    VideoGenerator(data_dir, transform=None, target_transform=None)
