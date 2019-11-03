from sources.model import *


def testing_session(weight_path, x_test, y_test, input_size, loss, metric, feature_maps=64, learning_rate=1e-4):

    # Loading the model
    model = VGG16(input_size=input_size,
                  feature_maps=feature_maps,
                  lr=learning_rate,
                  loss=loss,
                  metric=metric)
    
    # Loading the last weights updated
    model.load_weights(weight_path)

    # Predict the class patch by patch 
    results = model.predict(x=x_test,
                            batch_size=1)

    # Threshold of the probability value
    results[results > 0.5] = int(1)
    results[results <= 0.5] = int(0)

    return results
