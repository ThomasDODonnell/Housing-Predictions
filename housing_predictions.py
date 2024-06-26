# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt 
# from sklearn.model_selection import train_test_split 
# from sklearn.linear_model import LinearRegression 
# from sklearn.metrics import mean_squared_error, mean_absolute_error 
# from sklearn import preprocessing 
# # from sklearn.preprocessing import CategoricalEncoder
# from category_encoders import BinaryEncoder
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

# tdata = pd.read_csv("train.csv")
# # print(tdata.head())

# Excellent_to_poor_dict = {"Ex": 5, "Gd": 4, "TA":3, "Fa":2, "Po":1}
# Na_to_excellent_dict = {"Ex": 5, "Gd": 4, "Ta":3, "Fa":2, "Po":1, "NA":0}
# Na_to_gd_dict = {"Gd": 4, "Av": 3, "Mn":2, "No":1, "NA":0}
# Bsmt_finish_dict = {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"NA":0}
# Functionality_dict = {"Typ":8,"Min1":7,"Min2":6,"Mod":5,"Maj1":4,"Maj2":3,"Sev":2,"Sal":1} # should salvage be 0 or 1
# Garage_fin_dict = {"Fin":3,"RFn":2,"Unf":1,"NA":0}
# Na_to_fa_ex_dict = {"Ex": 4, "Gd": 3, "Ta":2, "Fa":1, "NA":0}
# Fence_dict = {"GdPrv":4,"MnPrv":3,"GdWo":2,"MnWw":1,"NA":0}
# Land_contour_dict = {"Lvl":4, "Bnk":3, "HLS":2, "Low": 1} 
# Utilities_dict = {"AllPub":4,"NoSewr":3,"NoSeWa":2,"ELO":1}
# slope_dict = {"Gtl":3,"Mod":2,"Sev":1}
# boolian_encode_dict = {"Y":1, 'N':0}

# tdata["ExterQual"] = tdata["ExterQual"].map(Excellent_to_poor_dict).fillna(0).astype(int)
# tdata["ExterCond"] = tdata["ExterCond"].map(Excellent_to_poor_dict).fillna(0).astype(int)
# tdata["BsmtQual"] = tdata["BsmtQual"].map(Na_to_excellent_dict).fillna(0).astype(int)
# tdata["BsmtCond"] = tdata["BsmtCond"].map(Na_to_excellent_dict).fillna(0).astype(int)
# tdata["BsmtExposure"] = tdata["BsmtExposure"].map(Na_to_gd_dict) # check this way of doing it. There is a no and an NA which might be problematic
# tdata["BsmtFinType1"] = tdata["BsmtFinType1"].map(Bsmt_finish_dict).fillna(0).astype(int)
# tdata["BsmtFinType2"] = tdata["BsmtFinType2"].map(Bsmt_finish_dict).fillna(0).astype(int)
# tdata["HeatingQC"] = tdata["HeatingQC"].map(Excellent_to_poor_dict).fillna(0).astype(int)
# tdata["KitchenQual"] = tdata["KitchenQual"].map(Excellent_to_poor_dict).fillna(0).astype(int)
# tdata["Functional"] = tdata["Functional"].map(Functionality_dict).fillna(0).astype(int)
# tdata["FireplaceQu"] = tdata["FireplaceQu"].map(Na_to_excellent_dict).fillna(0).astype(int)
# tdata["GarageFinish"] = tdata["GarageFinish"].map(Garage_fin_dict).fillna(0).astype(int)
# tdata["GarageQual"] = tdata["GarageQual"].map(Na_to_excellent_dict).fillna(0).astype(int)
# tdata["GarageCond"] = tdata["GarageCond"].map(Na_to_excellent_dict).fillna(0).astype(int)
# tdata["PoolQC"] = tdata["PoolQC"].map(Na_to_fa_ex_dict).fillna(0).astype(int)
# tdata["Fence"] = tdata["Fence"].map(Fence_dict).fillna(0).astype(int)
# tdata["LandContour"] = tdata["LandContour"].map(Land_contour_dict).fillna(0).astype(int)
# tdata["Utilities"] = tdata["Utilities"].map(Utilities_dict).fillna(0).astype(int)
# tdata["LandSlope"] = tdata["LandSlope"].map(slope_dict).fillna(0).astype(int)
# tdata["CentralAir"] = tdata["CentralAir"].map(boolian_encode_dict).fillna(0).astype(int)

# Binary_encode_list = ["MSSubClass", "MSZoning", "Neighborhood","Condition1","Condition2", "HouseStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "SaleType"]
# tdata = BinaryEncoder(cols=Binary_encode_list).fit_transform(tdata)

# one_hot_encode_list = ["Street", "Alley", "LotShape", "LotConfig", "BldgType", "RoofStyle", "MasVnrType", "Foundation", "Heating", "Electrical", "GarageType", "PavedDrive", "MiscFeature", "SaleCondition"]

# for item in one_hot_encode_list:
#     one_hot_df = pd.get_dummies(tdata[item], prefix=item, dtype=int)
#     tdata = tdata.join(one_hot_df)
#     tdata.drop(axis=1, labels=item, inplace=True)

# # time_vars = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "MoSold", "YrSold"]

# # for i in range(len(time_vars)):
# #     tdata[time_vars[i]] = tdata[time_vars[i]].fillna(0).astype(int)

# for i in range(len(tdata.columns)):
#     tdata[tdata.columns[i]] = tdata[tdata.columns[i]].fillna(0).astype(int)

# scaler = StandardScaler()

# # transform = ['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition']
# # copy = tdata[['Id','MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition']].copy

# # copy = tdata.loc[:, tdata.columns != 'Id' or 'SalePrice']
# prices = tdata["SalePrice"]
# variablesdf = tdata.copy()
# variablesdf.drop(['Id', "SalePrice"],  axis=1, inplace=True)
# variables = scaler.fit_transform(variablesdf)
# variablesdf = pd.DataFrame(variablesdf, index=variablesdf.index, columns=variablesdf.columns)

# # nulls = variablesdf.isnull()
# # print(nulls)
# # for i in range(len(variablesdf.columns)):
# #     for j in range(0, 1460):
# #         if nulls[variablesdf.columns[i]][j] == True:
# #             print("{} at position {}".format(variablesdf[i], j))

# # for i in range(len(scaled_features_df.columns)):
# #     scaled_features_df[scaled_features_df.columns[i]] = scaled_features_df[scaled_features_df.columns[i]].fillna(0).astype(int)

# coeff_fit = LinearRegression().fit(variablesdf, prices)
# coeff_fit_list = []
# for i in range(len(variablesdf.columns)):
#     # print("The coefficient for {} is {}".format(variablesdf.columns[i], coeff_fit.coef_[i]))
#     coeff_fit_list.append((variables[i], coeff_fit.coef_[i]))

# print("max is ", np.max(coeff_fit.coef_))
# print("min is ", np.min(coeff_fit.coef_))
# print(sorted(coeff_fit_list, key=lambda x: x[1]))

# X_train, X_test, y_train, y_test = train_test_split(variablesdf, prices, test_size=0.20, random_state=42)

# fit = LinearRegression().fit(X_train, y_train)

# fit_predictions = fit.predict(X_test)

# print("score ", fit.score(X_test, y_test))

# plt.plot(fit_predictions, y_test, "bo")
# plt.xlabel("Predicted")
# plt.ylabel("actual")
# plt.show()

# def RMS_log(actual, predicted):
#     actual = np.log(actual)
#     predicted = np.log(predicted)
#     sum = 0
#     for i in range(len(predicted)):
#         error = (actual[i] - predicted[i]) ** 2
#         sum += error
#     return sum / len(predicted)

# print("log RMS is: ", RMS_log(fit_predictions, y_test))

# final_fit = LinearRegression().fit(variablesdf, prices)

# final_fit.predict()

from housing_predictions_functions import DataPrepClass
from sklearn.linear_model import LinearRegression 
import csv
import pandas as pd
import numpy as np
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


j = 0 
i=0
found = False
vals = np.arange(20, 200, 10)
# print(vals)
# def find(j):
#     for i in range(len(train["MSSubClass"])):
#         if train["MSSubClass"][i] == vals[j]:
#             print("not ", vals[j])
#             j+=1
#         if vals[j] not in train["MSSubClass"]:
#             print("culperate", vals[j])
#             return True

found = False
# while j < len(vals) - 1:
#     if train["MSSubClass"][i] == vals[j]:
#         print("not ", vals[j])
#         j+=1
#         i = 0 
#     if vals[j] not in train["MSSubClass"]:
#         print("culperate", vals[j])
#         break
#     i+=1
    
for i in range(len(vals)):
    if vals[i] not in test["MSSubClass"]:
        print("culperate", vals[i])
        break



print(len(train["MSSubClass"].value_counts()))
print(len(test["MSSubClass"].value_counts()))
# print(type(test["MSSubClass"].value_counts()))
# for i in range(len(train["MSSubClass"].value_counts())):
#     if train["MSSubClass"].value_counts()[i][0] not in test["MSSubClass"].value_counts():
#         print(train["MSSubClass"].value_counts()[i][0])



traindata = DataPrepClass("train.csv", True, "TA")
traindatadf = traindata.get_df()

s = 'SalePrice'
 
# check if string is present in list
if any(s in i for i in traindatadf.columns):
    print(f'{s} is present in the list')
else:
    print(f'{s} is not present in the list')

print(traindatadf.columns)

prices = traindata.get_prices()
trainids = traindata.get_ids()
# trainvariables = traindatadf.drop(['Id', "SalePrice"],  axis=1, inplace=True)

testdata = DataPrepClass("test.csv", train=True)
testdatadf = testdata.get_df()
testids = testdata.get_ids()
# testvariables = testdatadf.drop('Id',  axis=1, inplace=True)

print(testdatadf.head())

print("train", traindatadf.columns, len(traindatadf.columns))
print("test", testdatadf.columns, len(traindatadf.columns))

# # print(traindatadf["MSSubClass_0"], traindatadf["MSSubClass_1"], traindatadf["MSSubClass_2"], traindatadf["MSSubClass_3"], testdatadf["MSSubClass_0"],testdatadf["MSSubClass_1"],testdatadf["MSSubClass_2"])
# for i in range(len(traindatadf.columns)):
#     if testdatadf.columns[i] != traindatadf.columns[i]:
#         print(i, testdatadf.columns[i], traindatadf.columns[i])

fit = LinearRegression().fit(traindatadf, prices)


predictions = fit.predict(testdatadf)

with open("predictions.csv", "w") as file:
    writer = csv.writer(file)

    writer.writerow("Id", " SalePrice")
    for i in range(len(predictions)):
        writer.writerow(testids[i], " "+predictions[i])
    


