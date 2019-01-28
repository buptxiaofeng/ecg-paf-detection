import numpy
import glob
import os
from sklearn.utils import shuffle
class Data:

    def __init__(self):
        pass

    def load_train_data(self, path):
        negative_path = os.path.join(path, "n*.dat")
        sample_list = glob.glob(negative_path)
        sample_list = sorted(sample_list)

        #each segmentation is 30 minutes with 128hz
        negative_train_data = numpy.zeros((len(sample_list), 30 * 60 * 128, 2))
        for index, sample in enumerate(sample_list):
            #why divided by 200?
            tmp_data = numpy.fromfile(sample, dtype=numpy.int16) / 200
            tmp_data = numpy.reshape(tmp_data, (1, int(tmp_data.shape[0] / 2), 2))
            negative_train_data[index, : , :] = tmp_data

        positive_path = os.path.join(path, "p*.dat")
        sample_list = glob.glob(positive_path)
        sample_list = sorted(sample_list)
        positive_train_data = numpy.zeros((len(sample_list), 30 * 60 * 128, 2))
        for index, sample in enumerate(sample_list):
            tmp_data = numpy.fromfile(sample, dtype=numpy.int16) / 200
            tmp_data = numpy.reshape(tmp_data, (1, int(tmp_data.shape[0] / 2), 2))
            positive_train_data[index, :, :] = tmp_data

        shape = negative_train_data.shape
        negative_train_data = numpy.reshape(negative_train_data, (shape[0] * 6, int(shape[1] / 6), shape[2]))
        negative_label = numpy.zeros(negative_train_data.shape[0], dtype=numpy.uint8)

        shape = positive_train_data.shape
        positive_train_data = numpy.reshape(positive_train_data, (shape[0] * 6, int(shape[1] / 6), shape[2]))
        positive_label = numpy.ones(positive_train_data.shape[0])

        data = numpy.concatenate((negative_train_data, positive_train_data), axis = 0)
        label = numpy.concatenate((negative_label, positive_label), axis = 0)

        data, label = shuffle(data, label)
        data = data.reshape((data.shape[0], 1, data.shape[1], data.shape[2]))
        label = label.tolist()
        label = [round(x) for x in label]
        label = numpy.eye(2)[label]

        return data, label

    def load_test_data(self, path):
        #parse label.txt
        label_file = open("test_data/labels.txt")
        lines = label_file.readlines()
        label_dict = {}
        for line in lines:
            index = line.split(" ")[0]
            label = line.split(" ")[1]
            if int(index) < 10:
                label_dict["t0"+index+".dat"] = int(label)
            else:
                label_dict["t" + index + ".dat"] = int(label)

        path = os.path.join(path, "*.dat")
        sample_list = glob.glob(path)
        sample_list = sorted(sample_list)
        test_data = numpy.zeros((len(sample_list), 30 * 60 * 128, 2))
        test_label = numpy.zeros((len(sample_list)))
        for index, sample in enumerate(sample_list):
            tmp_data_name = sample.split("/")[-1]
            tmp_data = numpy.fromfile(sample, dtype=numpy.int16) / 200
            tmp_data = numpy.reshape(tmp_data, (1, int(tmp_data.shape[0] / 2), 2))
            test_data[index, :, :] = tmp_data
            test_label[index] = label_dict[tmp_data_name]

        test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1], test_data.shape[2]))
        test_label = test_label.tolist()
        test_label = [round(x) for x in test_label]
        test_label = numpy.eye(2)[test_label]

        return test_data, test_label

if __name__ == "__main__":
    data = Data()
    train_path = "train_data"
    test_path = "test_data"
    data.load_train_data(path = train_path)
    data.load_test_data(path = test_path)
