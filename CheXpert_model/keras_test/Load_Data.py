from tensorflow import keras
import numpy as np
import csv
from PIL import Image

root_path = '/extra1/Dataset/'
train_csv_path = '/home/zhouli/model_test/radio.csv'
test_csv_path = '/home/zhouli/model_test/valid_test.csv'
img_size = 320


def load_image(filename, is_train=False):
    global img_size

    img = Image.open(filename)
    img_resized = img.resize((img_size, img_size), Image.ANTIALIAS)
    img_data = np.array(img_resized)
    img_data = img_data[:,:,np.newaxis]
    # print(f"img shape:{img_data.shape}")
    if is_train:
        return img_data.astype(np.float32)
    return (img_data / 255.0).astype(np.float32)


def shuffle_data(x_data, y_data):
    state = np.random.get_state()
    np.random.shuffle(x_data)
    np.random.set_state(state)
    np.random.shuffle(y_data)
    return x_data, y_data


def get_all(root_path, train_csv_path, test_csv_path):
    class Dataset:
        pass
    
    is_train = True
    for file_path in [train_csv_path, test_csv_path]:
        images = []
        labels = []
        with open(file_path) as csv_file:
            lines = csv.reader(csv_file)
            for line in lines:
                img_path = root_path + line[0]
                images.append(img_path)
                labels.append(int(line[1]))
            if is_train:
                Dataset.train_images = images
                Dataset.train_labels = labels
            else:
                Dataset.test_images = images
                Dataset.test_labels = labels
        is_train = False

    Dataset.train_images = np.array(Dataset.train_images)
    Dataset.train_labels = np.array(Dataset.train_labels)
    Dataset.test_images = np.array(Dataset.test_images)
    Dataset.test_labels = np.array(Dataset.test_labels)
    
    return Dataset


def DATA_ITERATOR(x_data, y_data, batch_size=16, is_train=True):
    shuffle_data(x_data, y_data)

    # split data into batches
    batch_num = int(len(x_data) / batch_size)
    # standardize the batch size
    split_sign = int(batch_num * batch_size)

    x_batches = np.array_split(x_data[:split_sign], batch_num)
    y_batches = np.array_split(y_data[:split_sign], batch_num)
    # print(batch_num, batch_size, len(x_data), len(y_data), len(x_batches), len(y_batches))
    # print(y_batches[0].shape, x_batches[0].shape)

    # get data batch
    for i in range(len(x_batches)):
        if is_train:
            x_batch = np.array(list(map(load_image, x_batches[i], [True for _ in range(batch_size)])))
        else:
            x_batch = np.array(list(map(load_image, x_batches[i], [False for _ in range(batch_size)])))

        y_batch = np.array(y_batches[i])
        # print(x_batch.shape, y_batch.shape)

        yield x_batch, keras.utils.to_categorical(y_batch, num_classes=3)


def init():
    return get_all(root_path, train_csv_path, test_csv_path)


""" dataset = get_all(root_path, train_csv_path, test_csv_path)
# print(dataset.train_images.shape)
# print(dataset.train_labels.shape)
for x, y in DATA_ITERATOR(dataset.train_images, dataset.train_labels, 16, True):
    print(x[1].dtype, y[1].dtype) """
