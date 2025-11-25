import os
import ffmpeg

input_folder = 'seizure_dataset/mar_16_splited'
output_folder = 'seizure_dataset/360p_splited'

def process_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # 

        if os.path.isdir(file_path):
            process_folder(file_path)
        # 

        elif file_name.endswith('.mp4') or file_name.endswith('.avi'):
            # 
            relative_path = os.path.relpath(folder_path, input_folder)
            output_folder_path = os.path.join(output_folder, relative_path)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            output_path = os.path.join(output_folder_path, file_name)
            # 

            stream = ffmpeg.input(file_path)
            stream = ffmpeg.output(stream, output_path, s='640x360')
            ffmpeg.run(stream)

for subfolder_name in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder_name)
    if os.path.isdir(subfolder_path):
        # 
        process_folder(subfolder_path)
