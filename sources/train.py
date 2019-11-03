from sources.model import *
from tensorflow.keras.callbacks import *
from tensorboard import program


def trainning_session(train_generator, valid_generator, len_train, len_valid, input_size, weight_path, logs_path, Proportion,
                      loss, metric, patience=10, feature_maps=64, learning_rate=1e-4, batch=20, epochs=100):

    # Callbacks
    model_checkpoint = ModelCheckpoint(weight_path,
                                       monitor='val_accuracy',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max')

    Board = TensorBoard(log_dir=logs_path,
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True,
                        profile_batch=0)
    
    EarlyStop = EarlyStopping(monitor='val_accuracy',
                              mode='max',
                              patience=patience)

    # Tensorboard launch in virtualenv
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logs_path])
    url = tb.launch()
    
    print(f"Tensorboard visualization available at : {url}")

    # Loading the model
    model = VGG16(input_size=input_size,
                  feature_maps=feature_maps,
                  lr=learning_rate,
                  loss=loss,
                  metric=metric)

    model.fit_generator(train_generator,
                        steps_per_epoch=len_train / batch,
                        epochs=epochs,
                        validation_data=valid_generator,
                        validation_steps=len_valid / batch,
                        shuffle=True,
     #                  class_weight={0: Proportion, 1: (1. - Proportion)},
                        callbacks=[model_checkpoint, Board, EarlyStop])

