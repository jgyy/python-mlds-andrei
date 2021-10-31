"""
🐶 Using Transfer Learning and TensorFlow 2.0 to Classify Different Dog Breeds
"""
from os import listdir, system
from os.path import join, dirname, splitext
from datetime import datetime
from numpy import unique, array, where, max as nmax, sum as nsum, argmax, isin, arange
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from tensorflow_hub import __version__ as hub_version, KerasLayer
from tensorflow.keras import Sequential, layers, losses, optimizers, callbacks, models
from tensorflow import (
    __version__ as tf_version,
    float32,
    constant,
    config,
    io,
    image,
    data,
)
from matplotlib.image import imread
from matplotlib.pyplot import (
    show,
    imshow,
    figure,
    axis,
    tight_layout,
    subplot,
    title,
    xticks,
    yticks,
    bar,
)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_IMAGES = 1000
NUM_EPOCHS = 100
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]
file = lambda x: join(dirname(__file__), x)
readcsv = lambda x: DataFrame(read_csv(file(x)))


def process_image(image_path):
    """
    Takes an image file path and turns it into a Tensor.
    """
    img = io.read_file(image_path)
    img = image.decode_jpeg(img, channels=3)
    img = image.convert_image_dtype(img, float32)
    img = image.resize(img, size=[IMG_SIZE, IMG_SIZE])
    return img


def get_image_label(image_path, label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image, label).
    """
    _image = process_image(image_path)
    return _image, label


def create_data_batches(
    xda, yda=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False
):
    """
    Creates batches of data out of image (x) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
    Also accepts test data as input (no labels).
    """
    if test_data:
        print("Creating test data batches...")
        _data = data.Dataset.from_tensor_slices((constant(xda)))
        data_batch = _data.map(process_image).batch(batch_size)
    if valid_data:
        print("Creating validation data batches...")
        _data = data.Dataset.from_tensor_slices((constant(xda), constant(yda)))
        data_batch = _data.map(get_image_label).batch(batch_size)
    if not (test_data or valid_data):
        print("Creating training data batches...")
        print(type(constant(xda)))
        _data = data.Dataset.from_tensor_slices((constant(xda), constant(yda)))
        _data = _data.shuffle(buffer_size=len(xda))
        _data = _data.map(get_image_label)
        data_batch = _data.batch(batch_size)
    return data_batch


def show_25_images(_image, labels, unique_breeds):
    """
    Displays 25 images from a data batch.
    """
    figure(figsize=(10, 10))
    for i in range(25):
        subplot(5, 5, i + 1)
        imshow(_image[i])
        title(unique_breeds[labels[i].argmax()])
        axis("off")


def create_model(input_shape, output_shape, model_url=MODEL_URL):
    """
    Create a function which builds a Keras model
    """
    print("Building model with:", model_url)
    model = Sequential(
        [KerasLayer(model_url), layers.Dense(units=output_shape, activation="softmax")]
    )
    model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics=["accuracy"],
    )
    model.build(input_shape)
    return model


def create_tensorboard_callback():
    """
    Create a function to build a TensorBoard callback
    """
    logdir = file(join("logs", datetime.now().strftime("%Y%m%d-%H%M%S")))
    return callbacks.TensorBoard(logdir)


def train_model(train_data, val_data, early_stopping, output_shape):
    """
    Trains a given model and returns the trained version.
    """
    model = create_model(INPUT_SHAPE, output_shape)
    tensorboard = create_tensorboard_callback()
    model.fit(
        x=train_data,
        epochs=NUM_EPOCHS,
        validation_data=val_data,
        validation_freq=1,
        callbacks=[tensorboard, early_stopping],
    )
    return model


def get_pred_label(prediction_probabilities, unique_breeds):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[argmax(prediction_probabilities)]


def unbatchify(_data, unique_breeds):
    """
    Takes a batched dataset of (image, label) Tensors and returns separate arrays
    of images and labels.
    """
    images = []
    labels = []
    for _image, label in _data.unbatch().as_numpy_iterator():
        images.append(_image)
        labels.append(unique_breeds[argmax(label)])
    return images, labels


def plot_pred(prediction_probabilities, labels, images, unique_breeds, num=1):
    """
    View the prediction, ground truth label and image for sample n.
    """
    pred_prob, true_label, _image = (
        prediction_probabilities[num],
        labels[num],
        images[num],
    )
    pred_label = get_pred_label(pred_prob, unique_breeds)
    imshow(_image)
    xticks([])
    yticks([])
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"
    title(
        "{} {:2.0f}% ({})".format(pred_label, nmax(pred_prob) * 100, true_label),
        color=color,
    )


def plot_pred_conf(prediction_probabilities, labels, unique_breeds, num=1):
    """
    Plots the top 10 highest prediction confidences along with
    the truth label for sample n.
    """
    pred_prob, true_label = prediction_probabilities[num], labels[num]
    get_pred_label(pred_prob, unique_breeds)
    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]
    top_plot = bar(arange(len(top_10_pred_labels)), top_10_pred_values, color="grey")
    xticks(
        arange(len(top_10_pred_labels)), labels=top_10_pred_labels, rotation="vertical"
    )
    if isin(true_label, top_10_pred_labels):
        top_plot[argmax(top_10_pred_labels == true_label)].set_color("green")
    else:
        pass


def save_model(model, suffix=None):
    """
    Saves a given model in a models directory and appends a suffix (str)
    for clarity and reuse.
    """
    modeldir = file(join("models", datetime.now().strftime("%Y%m%d-%H%M%s")))
    model_path = modeldir + "-" + suffix + ".h5"
    print(f"Saving model to: {model_path}...")
    model.save(model_path)
    return model_path


def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")
    model = models.load_model(model_path, custom_objects={"KerasLayer": KerasLayer})
    return model


def wrapper():
    """
    wrapper function
    """
    print("TF version:", tf_version)
    print("Hub version:", hub_version)
    print(
        "GPU",
        "available (YESS!!!!)"
        if config.list_physical_devices("GPU")
        else "not available :(",
    )
    labels_csv = readcsv("labels.csv")
    print(labels_csv.describe())
    print(labels_csv.head())
    labels_csv["breed"].value_counts().plot.bar(figsize=(20, 10))
    filenames = [join(file("train"), f"{fname}.jpg") for fname in labels_csv["id"]]
    print(filenames[:10])
    if len(listdir(file("train"))) == len(filenames):
        print("Filenames match actual amount of files!")
    else:
        print(
            "Filenames do not match actual amount of files, check the target directory."
        )
    figure(figsize=(8, 8))
    imshow(imread(filenames[9000]))
    axis("off")
    tight_layout()
    labels = labels_csv["breed"].to_numpy()
    print(labels[:10])
    if len(labels) == len(filenames):
        print("Number of labels matches number of filenames!")
    else:
        print(
            "Number of labels does not match number of filenames, check data directories."
        )
    unique_breeds = unique(labels)
    len(unique_breeds)
    print(labels[0])
    print(labels[0] == unique_breeds)
    boolean_labels = [label == array(unique_breeds) for label in labels]
    print(boolean_labels[:2])
    print(labels[0])
    print(where(unique_breeds == labels[0])[0][0])
    print(boolean_labels[0].argmax())
    print(boolean_labels[0].astype(int))
    xda = filenames
    yda = boolean_labels
    print(NUM_IMAGES)

    train_test(xda, yda, filenames, unique_breeds)


def train_test(xda, yda, filenames, unique_breeds):
    """
    Split them into training and validation using NUM_IMAGES
    """
    x_train, x_val, y_train, y_val = train_test_split(
        xda[:NUM_IMAGES], yda[:NUM_IMAGES], test_size=0.2, random_state=42
    )
    print(len(x_train), len(y_train), len(x_val), len(y_val))
    print(x_train[:5], y_train[:2])
    _image = imread(filenames[42])
    print(_image.shape)
    print(constant(_image)[:2])
    train_data = create_data_batches(x_train, y_train)
    val_data = create_data_batches(x_val, y_val, valid_data=True)
    print(train_data.element_spec, val_data.element_spec)
    train_images, train_labels = next(train_data.as_numpy_iterator())
    show_25_images(train_images, train_labels, unique_breeds)
    val_images, val_labels = next(val_data.as_numpy_iterator())
    show_25_images(val_images, val_labels, unique_breeds)

    make_predict(train_data, val_data, unique_breeds, xda, yda)


def make_predict(train_data, val_data, unique_breeds, xda, yda):
    """
    Make predictions on the validation data (not used to train on)
    """
    model = create_model(INPUT_SHAPE, len(unique_breeds))
    print(model.summary())
    early_stopping = callbacks.EarlyStopping(monitor="val_accuracy", patience=3)
    print(
        "GPU",
        "available (YESS!!!!)"
        if config.list_physical_devices("GPU")
        else "not available :(",
    )
    model = train_model(train_data, val_data, early_stopping, len(unique_breeds))
    predictions = model.predict(val_data, verbose=1)
    print(predictions)
    print(predictions.shape)
    print(predictions[0])
    print(f"Max value (probability of prediction): {nmax(predictions[0])}")
    print(f"Sum: {nsum(predictions[0])}")
    print(f"Max index: {argmax(predictions[0])}")
    print(f"Predicted label: {unique_breeds[argmax(predictions[0])]}")
    pred_label = get_pred_label(predictions[0], unique_breeds)
    print(pred_label)
    val_images, val_labels = unbatchify(val_data, unique_breeds)
    print(val_images[0], val_labels[0])
    plot_pred(
        prediction_probabilities=predictions,
        labels=val_labels,
        images=val_images,
        unique_breeds=unique_breeds,
    )
    plot_pred_conf(
        prediction_probabilities=predictions,
        labels=val_labels,
        unique_breeds=unique_breeds,
        num=9,
    )
    num_rows = 3
    num_cols = 2
    num_images = num_rows * num_cols
    figure(figsize=(5 * 2 * num_cols, 5 * num_rows))
    for i in range(num_images):
        subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_pred(
            prediction_probabilities=predictions,
            labels=val_labels,
            images=val_images,
            unique_breeds=unique_breeds,
            num=i,
        )
        subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_pred_conf(
            prediction_probabilities=predictions,
            labels=val_labels,
            unique_breeds=unique_breeds,
            num=i,
        )
        tight_layout(h_pad=1.0)

    check_predict(unique_breeds, model, val_data, xda, yda)


def check_predict(unique_breeds, model, val_data, xda, yda):
    """
    Let's check a few predictions and their different values
    """
    model_path = save_model(model, suffix="1000-images-Adam")
    model_1000_images = load_model(model_path)
    model.evaluate(val_data)
    model_1000_images.evaluate(val_data)
    print(len(xda), len(yda))
    full_data = create_data_batches(xda, yda)
    full_model = create_model(INPUT_SHAPE, len(unique_breeds))
    full_model_tensorboard = create_tensorboard_callback()
    full_model_early_stopping = callbacks.EarlyStopping(monitor="accuracy", patience=3)
    full_model.fit(
        x=full_data,
        epochs=NUM_EPOCHS,
        callbacks=[full_model_tensorboard, full_model_early_stopping],
    )
    model_path = save_model(full_model, suffix="all-images-Adam")
    loaded_full_model = load_model(model_path)
    test_path = file("test")
    test_filenames = [join(test_path, fname) for fname in listdir(test_path)]
    print(test_filenames[:10])
    print(len(test_filenames))
    test_data = create_data_batches(test_filenames, test_data=True)

    custom_image(test_data, loaded_full_model, unique_breeds)


def custom_image(test_data, loaded_full_model, unique_breeds):
    """
    Making predictions on custom images
    """
    test_predictions = loaded_full_model.predict(test_data, verbose=1)
    print(test_predictions[:10])
    preds_df = DataFrame(columns=["id"] + list(unique_breeds))
    print(preds_df.head())
    test_path = file("test")
    preds_df["id"] = [splitext(path)[0] for path in listdir(test_path)]
    print(preds_df.head())
    preds_df[list(unique_breeds)] = test_predictions
    print(preds_df.head())
    preds_df.to_csv(file("full_submission_1_mobilienetV2_adam.csv"), index=False)
    custom_path = file("dogs")
    custom_image_paths = [join(custom_path, fname) for fname in listdir(custom_path)]
    custom_data = create_data_batches(custom_image_paths, test_data=True)
    custom_preds = loaded_full_model.predict(custom_data)
    custom_pred_labels = [
        get_pred_label(custom_preds[i], unique_breeds) for i in range(len(custom_preds))
    ]
    print(custom_pred_labels)
    custom_images = []
    for _image in custom_data.unbatch().as_numpy_iterator():
        custom_images.append(_image)
    figure(figsize=(10, 10))
    for i, _image in enumerate(custom_images):
        subplot(1, 3, i + 1)
        xticks([])
        yticks([])
        title(custom_pred_labels[i])
        imshow(_image)


if __name__ == "__main__":
    wrapper()
    show()
    system(f"tensorboard --logdir {file('logs')}")
