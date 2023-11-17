import unittest
import pandas as pd
from preprocess import setup_trainFeature, split_train_test, calc_zscore

class Test_proprecess(unittest.TestCase):

    def setUp(self) -> None:

        self.filename = "data/Solar_Wind_Dst_1997_2016_shifted_forward.csv"
        self.train_col = ["Bx","By","Bz","Sv","Den", "Dst"]

    def test_split_train_test(self):

        train_ratio = 0.7
        val_ratio = 0.2

        traindf, valdf, testdf = split_train_test(self.filename, train_ratio, val_ratio)


        print(traindf)


        self.assertEqual((len(traindf) + len(valdf) + len(testdf)), 175320)

    def test_setup_trainFeature(self):

        train_ratio = 0.7
        val_ratio = 0.2

        trainALL, valALL, testALL = split_train_test(self.filename, train_ratio, val_ratio)

        train_df = setup_trainFeature(trainALL, self.train_col)
        val_df = setup_trainFeature(valALL, self.train_col)
        test_df = setup_trainFeature(testALL, self.train_col)

        self.assertEqual(train_df.shape[1], len(self.train_col))

    def test_calc_zscore(self):
        train_ratio = 0.7
        val_ratio = 0.2

        trainALL, valALL, testALL = split_train_test(self.filename, train_ratio, val_ratio)

        train_df = setup_trainFeature(trainALL, self.train_col)
        val_df = setup_trainFeature(valALL, self.train_col)
        test_df = setup_trainFeature(testALL, self.train_col)

        train_df = calc_zscore(train_df)

        print(train_df["Dst"])









