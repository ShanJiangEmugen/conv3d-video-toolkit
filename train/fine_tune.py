# fine tuning the pre-trained model
from network import replace_intermediate_layer_in_keras
from network import insert_intermediate_layer_in_keras
import tensorflow as tf
from datetime import datetime
import numpy as np
import gc
import os
from video_data_generator import *
import time
from make_shortcut import force_symlink
from train_ss import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def train_on_batch(model,
                   epochs,
                   steps,
                   train_generator,
                   test_generator,
                   ):
    # 循环训练模型
    best_loss = float('inf')
    for epoch in range(epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('current learning rate: {:.8f}'.format( 
               tf.keras.backend.eval(model.optimizer.learning_rate)))
        for step in range(steps):
            x_batch, y_batch = next(train_generator)
            loss, accuracy, f1_t, precision_t, recall_t \
                        = model.train_on_batch(x_batch, 
                                               y_batch, 
                                               reset_metrics=True,
                                              )
            print('Batch:', step+1 , 
                  'Loss:', round(loss,4), 
                  'Accuracy:', round(accuracy,4),
                  'f1: ', round(f1_t,4),
                  'precision: ', round(precision_t,4), 
                  'recall: ', round(recall_t,4))

            
            # 计算在测试集上的 loss 和 accuracy
            if step == steps-1:
                test_loss, test_accuracy, f1_m, precision_m, recall_m=\
                        model.test_on_batch(
                                            next(test_generator)[0],
                                            next(test_generator)[1],
                                            reset_metrics=True,)
                print('Test Loss:', round(test_loss,4), 
                      'Test Accuracy:', round(test_accuracy,4),
                      'f1: ', round(f1_m,4),
                      'precision: ', round(precision_m,4), 
                      'recall: ', round(recall_m,4))
                      
                # 如果测试集的loss比上一次低，就保存当前模型
                if 'best_loss' not in locals():
                    best_loss = test_loss
                elif test_loss < best_loss:
                    best_loss = test_loss
                    current_date = make_time_stamp()
                    model_fn = 'models/' + current_date + '_conv3d_model_tunned.h5'
                    model.save(model_fn)
                    print('Saved best model with loss:', best_loss)
                    print()
                    saved_path = 'models/'       
                    force_symlink(saved_path)
                    last_weight = model.get_weights()
                # 如果测试级结果变差，回到上一个模型
                elif test_loss >= best_loss:
                    model.set_weights(last_weight)
                    print('Roll back to last model since worse performance!')
    
                                  
        if epoch%1==0:
            model.optimizer.learning_rate *= tf.math.exp(-0.1)   
        end = time.time()  
        print('time taken for this epoch: {:.4f} seconds'.format(end-start))
        print('current date: ', make_time_stamp())
        


if __name__ == '__main__':
    # prepare data here
    test_dir = 'seizure_dataset/mar_16_splited/test'
    train_dir = 'seizure_dataset/mar_16_splited/train'
    
    # on MIT mouse
    # test_dir = 'MIT_data/test'
    # train_dir = 'MIT_data/train'
    
    # For UCF-101 data
    test_dir = 'UCF-101/test'
    train_dir = 'UCF-101/train'
    
    width = 112
    height = 112
    channel = 3
    batch_size = 32
    length = 16
    classes = os.listdir(train_dir)
    
    nb_epoch = 500
    train_step = 10

    target_size = [length, height, width]
    
    train_gen = video_generator(train_dir, target_size, batch_size)
    test_gen = video_generator(test_dir, target_size, batch_size)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # load model here
        model_path = 'models/Mar_09_2023_14_10_conv3d_model.h5'
        
        # on MIT mouse
        model_path = 'models/lastest_model.h5'
        ts_model = tf.keras.models.load_model(
            model_path, custom_objects=None, compile=False, options=None
        )
        
        # Do NOT train the first few layers
        ts_model.trainable = False

        ts_model.summary()
        
        
        new_layer = tf.keras.layers.Dense(len(classes), activation = 'softmax',
                                          name='new_out')(ts_model.layers[-3].output)
        new_model = tf.keras.Model(inputs=ts_model.input, 
                                  outputs=new_layer)
        new_model.layers[-1].trainable == True     
        '''                                  
        #new_model = replace_intermediate_layer_in_keras(ts_model, 
                 #                           2, 
                     #                       new_layer)
                                            
        # new_model = insert_intermediate_layer_in_keras(ts_model, 
             #                                            21, 
                #                                         new_layer)

        # new_model.summary()
        
        # modify the output layer
        # to make the new model to predict new labels

        new_model= tf.keras.Model(inputs=ts_model.input, 
                                  outputs=ts_model.layers[-3].output)
        new_layer = tf.keras.layers.Dense(len(classes), activation = 'softmax',
                                          name='new_out')
        
        new_model = tf.keras.models.Sequential([
                new_model,
                new_layer
                ])
        new_model.summary()
        
        # new_model = ts_model
        '''
        new_model.summary()
        
        lr = 5e-4
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        new_model.compile(loss="categorical_crossentropy", optimizer=opt,
                          metrics=[tf.keras.metrics.CategoricalAccuracy(),
                                   f1_m, precision_m, recall_m])

        # fine tuneing model with new data
        train_on_batch(new_model,nb_epoch,train_step,
                       train_gen,test_gen)
        
        
