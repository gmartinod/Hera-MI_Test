from sources.model import *


def testing_session(weight_path, x_test, y_test, input_size, loss, metric, feature_maps=64, learning_rate=1e-4):

    model = VGG16(input_size=input_size,
                  feature_maps=feature_maps,
                  lr=learning_rate,
                  loss=loss,
                  metric=metric)
    model.load_weights(weight_path)

    results = model.predict(x=x_test,
                            batch_size=1)

    results[results > 0.5] = int(1)
    results[results <= 0.5] = int(0)

    print(results)

    return results