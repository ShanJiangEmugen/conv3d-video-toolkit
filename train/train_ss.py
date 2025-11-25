from network import *
import tensorflow as tf
from datetime import datetime
import numpy as np
import gc
import os
from video_data_generator import *
from tqdm import tqdm
import time
from make_shortcut import force_symlink


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
gpus = tf.config.experimental.list_physical_devices('GPU')
###############################################################################
# Notes:                                                                      #
# this script is for training conv3d model,                                   #
#                                                                             #
#                                                                             #
#                                                                             #
#                                                                             #
#                                                                             #
#                                                                             #
#                                                                             #
# Log:                                                                        #
# Version 0.0: Mar 3, 2023 script created                                     #
#                                                                             #
#                                                                             #
###############################################################################
def scheduler(epoch, lr):
    
    return lr * tf.math.exp(-0.1)


    
    
def train_on_batch_method(epochs, 
                          batch_size,
                          steps,
                          train_generator,
                          test_generator,
                          length,
                          height, 
                          width,
                          classes):
                          
    from fine_tune import f1_m, precision_m, recall_m
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        model = c3d_model(length, 
                          height, 
                          width, 
                          3, 
                          len(classes))
        lr = 3e-4
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        
        model.compile(
            loss="categorical_crossentropy", optimizer=opt, 
            metrics=[tf.keras.metrics.CategoricalAccuracy(),
            f1_m, precision_m, recall_m]
            )
        

    model.summary()
    # 定义训练轮数和 batch 大小

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
                    model_fn = 'models/' + current_date + '_conv3d_model.h5'
                    model.save(model_fn)
                    print('Saved best model with loss:', best_loss)
                    print()
                                  
        if epoch%3==0:
            model.optimizer.learning_rate *= tf.math.exp(-0.1)   
        end = time.time()  
        print('time taken for this epoch: {:.4f} seconds'.format(end-start))
        print('current date: ', make_time_stamp())
       

def check_exist_and_make_dir(path):
    """
    check if the folder is existed,
    if NOT! make one 
    """
    if os.path.isdir(path) == False:
        os.mkdir(path)
        print(path, "Created!")

    else:
        print(path, "Found!")


def make_time_stamp():
    today = datetime.now()
    d2 = today.strftime("%b_%d_%Y_%H_%M")
    return d2


if __name__ == "__main__":
    # For MIT mice data
    test_dir = 'MIT_data/test/'
    train_dir = 'MIT_data/train/'
    
    # For SCN1A mice data
    test_dir = 'seizure_dataset/mar_16_splited/test'
    train_dir = 'seizure_dataset/mar_16_splited/train'

    # For UCF-101 data
    test_dir = 'UCF-101/test'
    train_dir = 'UCF-101/train'
    
    width = 112
    height = 112
    channel = 3
    batch_size = 32
    length = 16
    classes = os.listdir(train_dir)
    
    nb_epoch = 20
    train_step = 200
    test_step = 30

    target_size = [length, height, width]
    
    train_gen = video_generator(train_dir, target_size, batch_size)
    test_gen = video_generator(test_dir, target_size, batch_size)
    
    saved_path = 'models/'

    # check model destination 
    check_exist_and_make_dir(saved_path)
    
    train_on_batch_method(nb_epoch, 
                          batch_size,
                          train_step,
                          train_gen,
                          test_gen,
                          length,
                          height, 
                          width,
                          classes)
    force_symlink(saved_path)
