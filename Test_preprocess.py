import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
from preprocess import setup_trainFeature, split_train_test, normalize
from data_windowing import WindowGenerator

class Test_proprecess(unittest.TestCase):

    def setUp(self) -> None:

        self.filename = "data/Solar_Wind_Dst_1997_2016_shifted_forward.csv"
        self.train_col = ["Bx","By","Bz","Sv","Den", "Dst"]

    def test_split_train_test(self):

        train_ratio = 0.7
        val_ratio = 0.2

        traindf, valdf, testdf = split_train_test(self.filename, train_ratio, val_ratio, self.train_col)


        print(traindf)


        self.assertEqual((len(traindf) + len(valdf) + len(testdf)), 175320)

    def test_setup_trainFeature(self):



        df= pd.read_csv(self.filename)

        df = setup_trainFeature(df, self.train_col)

        self.assertEqual(df.shape[1], len(self.train_col))

    def test_normalize(self):
        train_ratio = 0.7
        val_ratio = 0.2

        trainALL, valALL, testALL = split_train_test(self.filename, train_ratio, val_ratio, self.train_col)



        traindf, valdf, testdf = normalize(trainALL, valALL, testALL)

        print(valdf)

class Test_data_windowing(unittest.TestCase):

    def setUp(self) -> None:

        self.filename = "data/Solar_Wind_Dst_1997_2016_shifted_forward.csv"
        self.train_col = ["Bx", "By", "Bz", "Sv", "Den", "Dst"]

        train_ratio = 0.7
        val_ratio = 0.2

        trainALL, valALL, testALL = split_train_test(self.filename, train_ratio, val_ratio, self.train_col)




        self.traindf, self.valdf, self.testdf = normalize(trainALL, valALL, testALL)

    def test_setup_WindowGenerator(self):

        input_width = 30
        label_width = 30
        shift = 1
        label_columns = ["Dst"]


        wg = WindowGenerator(input_width, label_width, shift,
                        self.traindf, self.valdf, self.testdf, label_columns)

        print(wg)

    def test_timeseries_dataset_from_array(self):

        input_data = np.array(self.testdf[:90], dtype=np.float32)
        target_data = None

        dataset = tf.keras.utils.timeseries_dataset_from_array(
            input_data, target_data, sequence_length=30)

        for inputs in dataset:

            assert np.array_equal(inputs[0], input_data[:30])
            print(inputs[0])
            break

    def test_make_dataset(self):

        input_width = 32
        label_width = 1
        shift = 1
        label_columns = ["Dst"]

        wg = WindowGenerator(input_width, label_width, shift,
                             self.traindf, self.valdf, self.testdf, label_columns)

        train_set = wg.make_dataset(self.traindf)

        print(train_set)
        print(train_set.element_spec)

    def test_property_train(self):
        input_width = 30
        label_width = 30
        shift = 1
        label_columns = ["Dst"]

        wg = WindowGenerator(input_width, label_width, shift,
                             self.traindf, self.valdf, self.testdf, label_columns)

        trainSet = wg.train


        for example_inputs, example_labels in trainSet.take(1):
            print(f"Inputs shape (batch, time, features): {example_inputs.shape}")
            print(f"Labels shape (batch, time, features): {example_labels.shape}")


























