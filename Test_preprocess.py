import unittest
import pandas as pd
from preprocess import setup_trainFeature, split_train_test, calc_zscore

class Test_proprecess(unittest.TestCase):

    def setUp(self) -> None:

        self.filename = "data/Solar_Wind_Dst_1997_2016_shifted_forward.csv"
        self.train_col = ["Bx","By","Bz","Sv","Den", "Dst"]

    def test_split_train_test(self):

        traindf, testdf = split_train_test(self.filename)

        df = traindf.loc[(traindf['Decyear'] >= 2002.0) & (traindf['Decyear'] <= 2003.0)]

        print(traindf)

        self.assertEqual(len(df), 0)
        self.assertEqual((len(traindf) + len(testdf)), 175320)

    def test_setup_trainFeature(self):

        traindf, testdf = split_train_test(self.filename)

        train = setup_trainFeature(traindf, self.train_col)
        test = setup_trainFeature(testdf, self.train_col)

        self.assertTrue( ("Dst" in train.keys()))



    def test_calc_zscore(self):

        train_set, test_set = split_train_test(self.filename)

        traindf = setup_trainFeature(train_set, self.train_col)
        testdf = setup_trainFeature(test_set, self.train_col)

        traindf = calc_zscore(traindf)

        print(traindf["Dst"])






