import ffmpeg
import os
import time


def vid_resize(vid_path, output_path, width, overwrite = False):
    '''
    use ffmpeg to resize the input video to the width given, keeping aspect ratio
    '''
    if not( os.path.isdir(os.path.dirname(output_path))):
        raise ValueError(f'output_path directory does not exists: {os.path.dirname(output_path)}')

    if os.path.isfile(output_path) and not overwrite:
        warnings.warn(f'{output_path} already exists but overwrite switch is False, nothing done.')
        return None

    input_vid = ffmpeg.input(vid_path)
    vid = (
        input_vid
        .filter('scale', width, -1)
        .output(output_path)
        .overwrite_output()
        .run()
    )
    return output_path
   

if __name__ == '__main__':
    start = time.time()

    resolutions = [1080, 720, 360]
    resolution_width = [1920, 1280, 640]
    
    video_path = 'Feb_14_videos/feb_14_3_mice/'
    video_fn = os.listdir(video_path)

    for j in range(len(resolutions)):
        
        out_folder = 'Feb_14_videos/{}p/'.format(resolutions[j]) # also change here if downgrading

        for i in video_fn:
	        current_path = video_path + i
	        output_path = out_folder + i[:-3]+ 'avi'
	        print('making ', output_path)
	        vid_resize(current_path, output_path = output_path, width = resolution_width[j]) 
	        # define output resolution here
	        # 1920*1080 --> 1080p
	        # 1280*720  --> 720p
	        # 640*360   --> 360p
	        # was       --> 2160p
	        
    end = time.time()
    print('total running time: ', end - start, ' seconds.')
