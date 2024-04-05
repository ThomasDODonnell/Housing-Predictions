import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 
# from sklearn.preprocessing import CategoricalEncoder
from category_encoders import BinaryEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class DataPrepClass():
    def __init__(self, file):
        self.dataframe = pd.read_csv(file)
        self.ordinal_cleaner()
        self.binary_encode()
        self.one_hot_encode()
        self.time()
        self.fill_zeros()
        self.standard_scale()
        return self.dataframe

    def ordinal_cleaner(self):
        Excellent_to_poor_dict = {"Ex": 5, "Gd": 4, "TA":3, "Fa":2, "Po":1}
        Na_to_excellent_dict = {"Ex": 5, "Gd": 4, "Ta":3, "Fa":2, "Po":1, "NA":0}
        Na_to_gd_dict = {"Gd": 4, "Av": 3, "Mn":2, "No":1, "NA":0}
        Bsmt_finish_dict = {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"NA":0}
        Functionality_dict = {"Typ":8,"Min1":7,"Min2":6,"Mod":5,"Maj1":4,"Maj2":3,"Sev":2,"Sal":1} # should salvage be 0 or 1
        Garage_fin_dict = {"Fin":3,"RFn":2,"Unf":1,"NA":0}
        Na_to_fa_ex_dict = {"Ex": 4, "Gd": 3, "Ta":2, "Fa":1, "NA":0}
        Fence_dict = {"GdPrv":4,"MnPrv":3,"GdWo":2,"MnWw":1,"NA":0}
        Land_contour_dict = {"Lvl":4, "Bnk":3, "HLS":2, "Low": 1}
        Utilities_dict = {"AllPub":4,"NoSewr":3,"NoSeWa":2,"ELO":1}
        slope_dict = {"Gtl":3,"Mod":2,"Sev":1}
        boolian_encode_dict = {"Y":1, 'N':0}

        key = {
            "ExterQual":Excellent_to_poor_dict,
            "KitchenQual":Excellent_to_poor_dict,
            "HeatingQC":Excellent_to_poor_dict,
            "ExterCond":Excellent_to_poor_dict,
            "BsmtQual":Na_to_excellent_dict, 
            "BsmtCond":Na_to_excellent_dict, 
            "FireplaceQu":Na_to_excellent_dict, 
            "GarageFinish":Na_to_excellent_dict, 
            "GarageCond":Na_to_excellent_dict,
            "BsmtExposure":Na_to_gd_dict,
            "BsmtFinType1":Bsmt_finish_dict, 
            "BsmtFinType2":Bsmt_finish_dict,
            "Functional":Functionality_dict,
            "GarageFinish":Garage_fin_dict,
            "PoolQC":Na_to_fa_ex_dict,
            "Fence":Fence_dict,
            "LandContour":Land_contour_dict,
            "Utilities":Utilities_dict,
            "LandSlope":slope_dict,
            "CentralAir":boolian_encode_dict
        }

        columns=["ExterQual", "ExterCond", "HeatingQC", "KitchenQual","BsmtQual", "BsmtCond", "FireplaceQu", 
                 "GarageFinish", "GarageCond","BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Functional",
                 "GarageFinish", "PoolQC", "Fence", "LandContour", "Utilities", "LandSlope", "CentralAir"]

        for i in range(len(columns)):
            self.dataframe[columns[i]] = self.dataframe[columns[i]].map(key[columns[i]]).fillna(0).astype(int)
    
    def binary_encode(self):
        Binary_encode_list = ["MSSubClass", "MSZoning", "Neighborhood","Condition1","Condition2", "HouseStyle", 
                              "RoofMatl", "Exterior1st", "Exterior2nd", "SaleType"]
        self.dataframe = BinaryEncoder(cols=Binary_encode_list).fit_transform(self.dataframe)
    
    def one_hot_encode(self):
        one_hot_encode_list = ["Street", "Alley", "LotShape", "LotConfig", "BldgType", "RoofStyle", "MasVnrType", "Foundation", "Heating", "Electrical", "GarageType", "PavedDrive", "MiscFeature", "SaleCondition"]

        for item in one_hot_encode_list:
            one_hot_df = pd.get_dummies(self.dataframe[item], prefix=item, dtype=int)
            self.dataframe = self.dataframe.join(one_hot_df)
            self.dataframe.drop(axis=1, labels=item, inplace=True)
    
    def time(self):
        time_vars = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "MoSold", "YrSold"]

        for i in range(len(time_vars)):
            self.dataframe[time_vars[i]] = self.dataframe[time_vars[i]].fillna(0).astype(int)
    
    def fill_zeros(self):
        for i in range(len(self.dataframe.columns)):
            self.dataframe[self.dataframe.columns[i]] = self.dataframe[self.dataframe.columns[i]].fillna(0).astype(int)
    
    def standard_scale(self):
        scaler = StandardScaler()
        copy = self.dataframe.copy()
        copy.drop(['Id', "SalePrice"],  axis=1, inplace=True)
        finaldata = scaler.fit_transform(copy)
        scaled_features_df = pd.DataFrame(finaldata, index=copy.index, columns=copy.columns)
        scaled_features_df.join(self.dataframe["SalePrice"])
        self.dataframe = scaled_features_df
    
df = DataPrepClass("train.csv")
print(df.head)