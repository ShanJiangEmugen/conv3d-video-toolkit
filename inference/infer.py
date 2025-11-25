from infer_generator import *
import os
import tensorflow as tf
import pandas as pd 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


def load_model(model_path,):
    # load model here
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # load model here
        
        model_path = 'models/lastest_model.h5'
        print('Loading path from {}'.format(model_path))
        
        ts_model = tf.keras.models.load_model(
            model_path, custom_objects=None, compile=True, options=None
        )
        
    return ts_model
    
def make_prediction(model, bs, infer_path):
    infer_video_fns = sorted([f for f in os.listdir(infer_path) \
                              if not f.startswith('.')])
                              
    save_path = 'predictions'

    for video in infer_video_fns:
            # save labels for each video
            columns = ['seizure_free', 'type_4', 'type_3']
            results = pd.DataFrame(columns=columns)
            
            video_path = os.path.join(infer_path, video)
            # initialize infer generator
            infer_gen = infer_generator(video_path, batch_size=bs)
            
            steps = check_batches(video_path, bs)
            for i in tqdm (range (steps), desc="Inferring..."):
                
                prediction = model.predict_on_batch(next(infer_gen))
                # print(prediction)
                # the output should looks like: [[0,1,0],
                #                                [1,0,0],
                #                                .......]
                # shape should be: [batch_size, 3]
                
                # convert to pd.DF and concat the temp with the final results DF
                prediction_df = pd.DataFrame(data=prediction, 
                                             columns=columns)
                                             
                results = pd.concat([results, prediction_df], ignore_index=True)
                
            
            save_name = '{}/{}_prediction.csv'.format(save_path, video[:-4])
            results.to_csv(save_name)
            
            
if __name__ == '__main__':
    infer_dir = 'video_4_infer'
    model_path = 'models/lastest_model.h5'
    bs = 2
    
    model = load_model(model_path)
    
    make_prediction(model, bs, infer_dir)
    

'''
    infer_path = 'video_4_infer'
    infer_video_fns = sorted([f for f in os.listdir(infer_path) \
                              if not f.startswith('.')])
                              
    save_path = 'predictions'
    
    
    # load model here
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # load model here
        model_path = 'models/lastest_model.h5'
        
        ts_model = tf.keras.models.load_model(
            model_path, custom_objects=None, compile=True, options=None
        )
        
        
        
    # main infer here
    bs = 2        # --> adjust this to fully occupied GPU,
    
    
    for video in infer_video_fns:
        # save labels for each video
        columns = ['seizure_free', 'type_3', 'type_4']
        results = pd.DataFrame(columns=columns)
        
        video_path = os.path.join(infer_path, video)
        # initialize infer generator
        infer_gen = infer_generator(video_path, batch_size=bs)
        
        steps = check_batches(video_path, bs)
        for i in tqdm (range (steps), desc="Inferring..."):
            
            prediction = ts_model.predict_on_batch(next(infer_gen))
            # the output should looks like: [[0,1,0],
            #                                [1,0,0],
            #                                .......]
            # shape should be: [batch_size, 3]
            print(prediction.shape)
            # convert to pd.DF and concat the temp with the final results DF
            prediction_df = pd.DataFrame(data=prediction, 
                                         columns=columns)
                                         
            results = pd.concat([results, prediction_df], ignore_index=True)
            
        
        save_name = '{}/{}_prediction.csv'.format(save_path, video)
        results.to_csv(save_name)
        '''
            
        
