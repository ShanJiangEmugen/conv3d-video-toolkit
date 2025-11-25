import cv2
import numpy as np
from tqdm import tqdm
import os

def check_batches(video_path, batch_size=32, length=15, mode='all'):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if mode=='all':
        return int((frame_count-15)/batch_size)
    return int(frame_count/length/batch_size)


def infer_generator(video_path, batch_size=32, length=15, mode='all'):
    if mode == 'all':
    
        frames_buffer = []
        cap = cv2.VideoCapture(video_path)
        batch = []
        start_idx = 0
        while True:
            
            # print('set starting index to', start_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            
            for i in range(length):
                # load a batch of data here
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_resized = cv2.resize(frame, [180, 120])
                frames_buffer.append(frame_resized)
                
            # now, there will have 15 frames
            start_idx += 1
            # roll bach the start index to the next
            # and get another 15 frames
            batch.append(frames_buffer)
                
            frames_buffer = []
            
            if len(batch) == batch_size:
                yield np.array(batch)
                batch = []
            
           

    else:
        frames_buffer = []
        start_idx = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            frame_resized = cv2.resize(frame, [180, 120])
            frames_buffer.append(frame_resized)

            if len(frames_buffer) == length*batch_size:
                batch_frames = np.array(frames_buffer).reshape((batch_size,
                                                                length,
                                                                frame_resized.shape[0],
                                                                frame_resized.shape[1],
                                                                frame_resized.shape[2]))
                                                                
                yield batch_frames
                frames_buffer = []
        cap.release()
    
    
if __name__ == '__main__':
    video_path = 'video_4_infer'
    videos = os.listdir(video_path)
    
    bs = 2
    l = 15
    for video in videos:
        print('working on: ', video)
        temp_path = os.path.join(video_path, video)
        video_gen = infer_generator(temp_path, bs, l)
        # batch size here depends on the GPU capacity

        # the reshape step might takes long time if input a large video
        # but the model was trained on 4k resolution 
        # try different resolution to see accuracy difference
        
        steps = check_batches(temp_path, bs, l)
        
        print('total {} steps will be precessed'.format(steps))
        for i in tqdm (range (steps), desc="Loading..."):
            batch_frames = next(video_gen)
            print(batch_frames.shape)
