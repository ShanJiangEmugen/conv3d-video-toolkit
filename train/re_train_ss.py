import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import date
import os
from video_data_generator import *
import time
from train_ss import make_time_stamp
from train_ss import scheduler
from make_shortcut import force_symlink


def train_model(test_dir, train_dir, lr):
    tf.keras.backend.clear_session()
    
    width = 180
    height = 120
    channel = 3
    batch_size = 32
    length = 15
    classes = os.listdir(train_dir)
    
    nb_epoch = 4
    train_step = 500
    test_step = 20

    target_size = [length, height, width]
    
    train_gen = video_generator(train_dir, target_size, batch_size)
    test_gen = video_generator(test_dir, target_size, batch_size)
    
    saved_path = 'models/'
    
    
    load_from = 'models/lastest_model.h5'
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
    
        model = load_model(load_from)
        
        current_date = make_time_stamp()
        model_fn = saved_path + current_date + '_conv3d_model_tuned.h5'

        callbacks = [tf.keras.callbacks.ModelCheckpoint(
            model_fn,
            verbose=1,
            monitor='val_loss',
            mode='auto',
            save_best_only=True),
            tf.keras.callbacks.LearningRateScheduler(
                scheduler, verbose=0),
            ]
        
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        
        model.compile(
            loss="categorical_crossentropy", optimizer=opt, 
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
            )

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        verbose=1,
        epochs=nb_epoch,
        callbacks=callbacks,
        steps_per_epoch=train_step,
        validation_steps=test_step,
        workers=20,
        use_multiprocessing=True,
    )
    
    del train_gen
    del test_gen
    
    return model
    
    
if __name__ == "__main__":
    
    test_dir = 'MIT_data/test/'
    train_dir = 'MIT_data/train/'
    model_dir = 'models'
    lr = 0.001
    # train the model in a loop
    # for changinge parameters, change them in the function!
    model = train_model(test_dir, train_dir, lr)
    force_symlink(model_dir)
    print()
    '''
    for i in range(25):

        model = train_model(test_dir, train_dir, lr)
        force_symlink(model_dir)
        lr = lr * (np.exp(-0.1))**3
        
        del model
        
        print()
        print((i+1)*4, ' epoch trained!')
   
'''






