# Data Wranglers
import numpy as np
import pandas as pd
# Convinience
from glob import glob
# SK Learn
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer

def Native():
    print("Native Run")
    return None

def data():
    def get_housing_data(*, verbose: bool=True):
        """
        Reads original CSV files from Directory structure.
        """
        path = r"..\Data\*"
        file_path = [ _ for _ in glob(path) ]
        return [ pd.read_csv(_, verbose=verbose) for _ in file_path ]

    def mean_reduction(var_name_1:str, var_name_2:str="SalePrice")->pd.DataFrame:
        tabel    = train.loc[:, [var_name_1, var_name_2]]
        tabel_gb = tabel.groupby(var_name_1)[var_name_2].agg([np.mean, np.std])
        cat_mean, cat_std   = tabel_gb.loc[:, "mean"].mean(), tabel_gb.loc[:, "std"].mean()
        full_mean, full_std = tabel.loc[:, var_name_2].mean(), tabel.loc[:, var_name_2].std()
        return tabel_gb
    
    sample, test, train = get_housing_data()
    
    #
    SF_List = ['TotalBsmtSF','GrLivArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','GarageArea']
    train["Total_SF"] = train.loc[:, SF_List].sum(axis=1)
    test["Total_SF"]  =  test.loc[:, SF_List].sum(axis=1)
    #
    train["Age_At_Sale"] = train.loc[:, "YrSold"] - train.loc[:, "YearBuilt"]
    test["Age_At_Sale"]  =  test.loc[:, "YrSold"] -  test.loc[:, "YearBuilt"]
    #
    train["Remodel_Age"] = train.YrSold - train.YearRemodAdd
    test["Remodel_Age"]  = test.YrSold  -  test.YearRemodAdd
    #
    train["Remodel_Done"] = (train.YearRemodAdd - train.YearBuilt).apply(lambda x: 1 if x>0 else 0)
    test["Remodel_Done"]  = (test.YearRemodAdd  -  test.YearBuilt).apply(lambda x: 1 if x>0 else 0)
    #
    train["Half_bath"] = train.loc[:, [ _ for _ in train.columns if "Bath" in _ and "Half" in _ ]].sum(axis=1)
    train["Full_bath"] = train.loc[:, [ _ for _ in train.columns if "Bath" in _ and "Half" not in _ ]].sum(axis=1)
    train["All_bath"]  = train["Full_bath"] + train["Half_bath"]
    test["Half_bath"]  = test.loc[:, [ _ for _ in test.columns if "Bath" in _ and "Half" in _ ]].sum(axis=1)
    test["Full_bath"]  = test.loc[:, [ _ for _ in test.columns if "Bath" in _ and "Half" not in _ ]].sum(axis=1)
    test["All_bath"]   = test["Full_bath"] + test["Half_bath"]
    #
    MSS_dict = {}
    for obj, key in enumerate(list(mean_reduction("MSSubClass").sort_values(by="mean").index)):
        MSS_dict[key] = obj
    train.MSSubClass = train.MSSubClass.map(MSS_dict)
    test.MSSubClass  = test.MSSubClass.map(MSS_dict)
    #
    MSZoning_dict = {}
    for obj, key in enumerate(list(mean_reduction("MSZoning").sort_values("mean").index)):
        MSZoning_dict[key] = obj
    train.MSZoning = train.MSZoning.map(MSZoning_dict)
    test.MSZoning  = test.MSZoning.map(MSZoning_dict)
    #
    train["Road"] = train.Street + train.Alley.fillna("")
    test["Road"]  =  test.Street + test.Alley.fillna("")
    road_dict = {}
    for obj, key in enumerate(list(mean_reduction("Road").sort_values("mean").index)):
        road_dict[key] = obj
    train.Road = train.Road.map(road_dict)
    test.Road  = test.Road.map(road_dict)
    #
    LotShape_dict = {}
    for obj, key in enumerate(mean_reduction("LotShape").sort_values("mean").index):
        LotShape_dict[key] = obj
    train.LotShape = train.LotShape.map(LotShape_dict)
    test.LotShape  = test.LotShape.map(LotShape_dict)
    #
    LandContour_dict = {}
    for obj, key in enumerate(mean_reduction("LandContour").sort_values("mean").index):
        LandContour_dict[key] = obj
    train.LandContour = train.LandContour.map(LandContour_dict)
    test.LandContour  = test.LandContour.map(LandContour_dict)
    #
    LotConfig_dict = {}
    for obj, key in enumerate(mean_reduction("LotConfig").sort_values("mean").index):
        LotConfig_dict[key] = obj
    train.LotConfig = train.LotConfig.map(LotConfig_dict)
    test.LotConfig  = test.LotConfig.map(LotConfig_dict)
    #
    neigh_dict = {}
    for obj, key in enumerate(mean_reduction("Neighborhood").sort_values("mean").index):
        neigh_dict[key] = obj
    train.Neighborhood = train.Neighborhood.map(neigh_dict)
    test.Neighborhood  = test.Neighborhood.map(neigh_dict)
    #
    cond1_dict = {}
    for obj, key in enumerate(mean_reduction("Condition1").sort_values("mean").index):
        cond1_dict[key] = obj
    train.Condition1 = train.Condition1.map(cond1_dict)
    test.Condition1  = test.Condition1.map(cond1_dict)
    #
    cond2_dict = {}
    for obj, key in enumerate(mean_reduction("Condition2").sort_values("mean").index):
        cond2_dict[key] = obj
    train.Condition2 = train.Condition2.map(cond2_dict)
    test.Condition2  = test.Condition2.map(cond2_dict)
    #
    BldgType_dict = {}
    for obj, key in enumerate(mean_reduction("BldgType").sort_values("mean").index):
        BldgType_dict[key] = obj
    train.BldgType = train.BldgType.map(BldgType_dict)
    test.BldgType  = test.BldgType.map(BldgType_dict)
    #
    HouseStyle_dict = {}
    for obj, key in enumerate(mean_reduction("HouseStyle").sort_values("mean").index):
        HouseStyle_dict[key] = obj
    train.HouseStyle = train.HouseStyle.map(HouseStyle_dict)
    test.HouseStyle  = test.HouseStyle.map(HouseStyle_dict)
    #
    RoofStyle_dict = {}
    for obj, key in  enumerate(mean_reduction("RoofStyle").sort_values("mean").index):
        RoofStyle_dict[key] = obj
    train.RoofStyle = train.RoofStyle.map(RoofStyle_dict)
    test.RoofStyle  = test.RoofStyle.map(RoofStyle_dict)
    # 
    Exterior1st_dict = {}
    for obj, key in enumerate(mean_reduction("Exterior1st").sort_values("mean").index):
        Exterior1st_dict[key] = obj
    train.Exterior1st = train.Exterior1st.map(Exterior1st_dict)
    test.Exterior1st  = test.Exterior1st.map(Exterior1st_dict)
    Exterior1st_dict = {}
    for obj, key in enumerate(mean_reduction("Exterior1st").sort_values("mean").index):
        Exterior1st_dict[key] = obj
    Exterior1st_dict[0]  = 0
    Exterior1st_dict[13] = 0
    Exterior1st_dict[1]  = 0
    Exterior1st_dict[2]  = 0
    Exterior1st_dict[14] = 0
    train.Exterior1st = train.Exterior1st.map(Exterior1st_dict)
    test.Exterior1st = train.Exterior1st.map(Exterior1st_dict)
    #
    train.Exterior2nd = train.Exterior2nd.map(
        {
            "VinylSd": "VinylSd",
            "MetalSd": "MetalSd",
            "HdBoard": "HdBoard",
            "Wd Sdng": "Wd Sdng",
            "Plywood": "Plywood",
        }
    )
    train.Exterior2nd.fillna("REST", inplace=True)
    test.Exterior2nd = test.Exterior2nd.map(
        {
            "VinylSd": "VinylSd",
            "MetalSd": "MetalSd",
            "HdBoard": "HdBoard",
            "Wd Sdng": "Wd Sdng",
            "Plywood": "Plywood",
        }
    )
    test.Exterior2nd.fillna("REST", inplace=True)
    Exterior2nd_dict = {}
    for obj, key in enumerate(mean_reduction("Exterior2nd").sort_values("Exterior2nd").index):
        Exterior2nd_dict[key] = obj
    train.Exterior2nd = train.Exterior2nd.map(Exterior2nd_dict)
    test.Exterior2nd  = test.Exterior2nd.map(Exterior2nd_dict)
    #
    train.MasVnrType.fillna("None", inplace=True)
    test.MasVnrType.fillna( "None", inplace=True)
    MasVnrType_dict = {}
    for obj, key in enumerate(mean_reduction("MasVnrType").sort_values("mean").index):
        MasVnrType_dict[key] = obj
    train.MasVnrType = train.MasVnrType.map(MasVnrType_dict)
    test.MasVnrType  = test.MasVnrType.map(MasVnrType_dict)
    #
    train.MasVnrArea.fillna(0, inplace=True)
    test.MasVnrArea.fillna( 0, inplace=True)
    #
    train.ExterQual = train.ExterQual.map(
        {
            "Fa": 0,
            "TA": 1,
            "Gd": 2,
            "Ex": 3,
        }
    )
    test.ExterQual = test.ExterQual.map(
        {
            "Fa": 0,
            "TA": 1,
            "Gd": 2,
            "Ex": 3,
        }
    )
    #
    Foundation_dict = {}
    for obj, key in enumerate(mean_reduction("Foundation").sort_values("mean").index):
        Foundation_dict[key] = obj
    train.Foundation = train.Foundation.map(Foundation_dict)
    test.Foundation  = test.Foundation.map(Foundation_dict)
    #
    train.BsmtQual = train.BsmtQual.map(
        {
            "Fa": 0,
            "TA": 1,
            "Gd": 2,
            "Ex": 3,
        }
    )
    test.BsmtQual = test.BsmtQual.map(
        {
            "Fa": 0,
            "TA": 1,
            "Gd": 2,
            "Ex": 3,
        }
    )
    train.BsmtQual.fillna(-1, inplace=True)
    test.BsmtQual.fillna( -1, inplace=True)
    #
    train.BsmtExposure = train.BsmtExposure.map(
        {
            "No": 0,
            "Mn": 1,
            "Av": 2,
            "Gd": 3,
        }
    )
    test.BsmtExposure = train.BsmtExposure.map(
        {
            "No": 0,
            "Mn": 1,
            "Av": 2,
            "Gd": 3,
        }
    )
    train.BsmtExposure.fillna(-1, inplace=True)
    test.BsmtExposure.fillna(-1, inplace=True)
    #
    train.BsmtFinType1 = train.BsmtFinType1.map(
        {
            "Unf": 0,
            "LwQ": 1,
            "Rec": 2,
            "BLQ": 3,
            "ALQ": 4,
            "GLQ": 5,
        }
    )
    test.BsmtFinType1 = test.BsmtFinType1.map(
        {
            "Unf": 0,
            "LwQ": 1,
            "Rec": 2,
            "BLQ": 3,
            "ALQ": 4,
            "GLQ": 5,
        }
    )
    train.BsmtFinType1.fillna(-1, inplace=True)
    test.BsmtFinType1.fillna( -1, inplace=True)
    #
    train.BsmtFinType2 = train.BsmtFinType2.map(
        {
            "Unf": 0,
            "LwQ": 1,
            "Rec": 2,
            "BLQ": 3,
            "ALQ": 4,
            "GLQ": 5,
        }
    )
    test.BsmtFinType2 = test.BsmtFinType2.map(
        {
            "Unf": 0,
            "LwQ": 1,
            "Rec": 2,
            "BLQ": 3,
            "ALQ": 4,
            "GLQ": 5,
        }
    )
    train.BsmtFinType2.fillna(-1, inplace=True)
    test.BsmtFinType2.fillna( -1, inplace=True)
    #
    HeatingQC_dict = {
        "Ex": 0,
        "Gd": 1,
        "TA": 2,
        "Fa": 3,
        "Po": 3,
    }
    train.HeatingQC = train.HeatingQC.map(HeatingQC_dict)
    test.HeatingQC  = train.HeatingQC.map(HeatingQC_dict)
    #
    CentralAir_dict = {
        "N": 0,
        "Y": 1,
    }
    train.CentralAir = train.CentralAir.map(CentralAir_dict)
    test.CentralAir  = test.CentralAir.map(CentralAir_dict)
    #
    train.Electrical = train.Electrical.apply(lambda x: 1 if (x=="SBrkr") else 0)
    test.Electrical  = test.Electrical.apply(lambda x: 1 if (x=="SBrkr") else 0)
    #
    train.BedroomAbvGr = train.BedroomAbvGr.map(
        {
            0: 1,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 5,
            7: 5,
            8: 5,
        }
    )
    test.BedroomAbvGr = test.BedroomAbvGr.map(
        {
            0: 1,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 5,
            7: 5,
            8: 5,
        }
    )
    #
    train.KitchenQual = train.KitchenQual.map(
        {
            "Fa": 0,
            "TA": 1,
            "Gd": 2,
            "Ex": 3
        }
    )
    test.KitchenQual = test.KitchenQual.map(
        {
            "Fa": 0,
            "TA": 1,
            "Gd": 2,
            "Ex": 3
        }
    )
    #
    train.TotRmsAbvGrd = train.TotRmsAbvGrd.map(
        {
            1 : 3,
            2 : 3,
            3 : 3,
            4 : 4,
            5 : 5,
            6 : 6,
            7 : 7,
            8 : 8,
            9 : 9,
            10:10,
            11:11,
            12:12,
            13:12,
            14:12,
        }
    )
    test.TotRmsAbvGrd = train.TotRmsAbvGrd.map(
        {
            1 : 3,
            2 : 3,
            3 : 3,
            4 : 4,
            5 : 5,
            6 : 6,
            7 : 7,
            8 : 8,
            9 : 9,
            10:10,
            11:11,
            12:12,
            13:12,
            14:12,
        }
    )
    #
    train.Fireplaces = train.Fireplaces.apply(lambda x: 2 if (x > 2) else x)
    test.Fireplaces  = test.Fireplaces.apply(lambda x: 2 if (x > 2) else x)
    #
    train.FireplaceQu = train.FireplaceQu.fillna("NA")
    test.FireplaceQu  = train.FireplaceQu.fillna("NA")
    train.FireplaceQu = train.FireplaceQu.map(
        {
            "Po": 0,
            "NA": 1,
            "Fa": 2,
            "TA": 3,
            "Gd": 4,
            "Ex": 5,
        }
    )
    test.FireplaceQu = test.FireplaceQu.map(
        {
            "Po": 0,
            "NA": 1,
            "Fa": 2,
            "TA": 3,
            "Gd": 4,
            "Ex": 5,
        }
    )
    #
    train.GarageType.fillna("NA", inplace=True)
    test.GarageType.fillna("NA", inplace=True)
    GarageType_dict = {}
    for obj, key in enumerate(mean_reduction("GarageType").sort_values("mean").index):
        GarageType_dict[key] = obj
    train.GarageType = train.GarageType.map(GarageType_dict)
    test.GarageType  = test.GarageType.map(GarageType_dict)
    #
    train["Age_Garage"] = train.YrSold - train.GarageYrBlt
    test["Age_Garage"]  = test.YrSold - test.GarageYrBlt
    #
    train.GarageFinish.fillna("NA", inplace=True)
    test.GarageFinish.fillna( "NA", inplace=True)
    train.GarageFinish = train.GarageFinish.map(
        {
            "NA" : 0,
            "Unf": 1,
            "RFn": 2,
            "Fin": 3,
        }
    )
    test.GarageFinish = test.GarageFinish.map(
        {
            "NA" : 0,
            "Unf": 1,
            "RFn": 2,
            "Fin": 3,
        }
    )
    #
    train.GarageCars = train.GarageCars.apply(lambda x: 3 if (x >= 3) else x)
    test.GarageCars  = train.GarageCars.apply(lambda x: 3 if (x >= 3) else x)
    #
    train.GarageQual.fillna("NA", inplace=True)
    test.GarageQual.fillna("NA", inplace=True)
    train.GarageQual = train.GarageQual.map(
        {
            "Po": 0,
            "NA": 0,
            "Fa": 1,
            "TA": 2,
            "Gd": 3,
            "Ex": 3,
        }
    )
    test.GarageQual = test.GarageQual.map(
        {
            "Po": 0,
            "NA": 0,
            "Fa": 1,
            "TA": 2,
            "Gd": 3,
            "Ex": 3,
        }
    )
    #
    train.GarageCond.fillna("NA", inplace=True)
    test.GarageCond.fillna("NA", inplace=True)
    train.GarageCond = train.GarageCond.map(
        {
            "Po": 0,
            "NA": 0,
            "Fa": 1,
            "TA": 2,
            "Gd": 3,
            "Ex": 3
        }
    )
    test.GarageCond = test.GarageCond.map(
        {
            "Po": 0,
            "NA": 0,
            "Fa": 1,
            "TA": 2,
            "Gd": 3,
            "Ex": 3
        }
    )
    #
    PavedDrive_dict = {}
    for obj, key in  enumerate(mean_reduction("PavedDrive").sort_values("mean").index):
        PavedDrive_dict[key] = obj
    train.PavedDrive = train.PavedDrive.map(PavedDrive_dict)
    test.PavedDrive  = test.PavedDrive.map(PavedDrive_dict)
    #
    train.SaleType = train.SaleType.apply( lambda x: 0 if (x == "WD") else 1 )
    test.SaleType  = test.SaleType.apply( lambda x: 0 if (x == "WD") else 1 )
    #
    SaleCondition_dict = {}
    for obj, key in enumerate(mean_reduction("SaleCondition").sort_values("mean").index):
        SaleCondition_dict[key] = obj
    train.SaleCondition = train.SaleCondition.map(SaleCondition_dict)
    test.SaleCondition  = test.SaleCondition.map(SaleCondition_dict)
    #
    train.Functional = train.Functional.map(
        {
            "Sal" : 0,
            "Sev" : 1,
            "Maj2": 2,
            "Maj1": 3,
            "Mod" : 4,
            "Min2": 5,
            "Min1": 6,
            "Typ" : 7,
        }
    )
    test.Functional = test.Functional.map(
        {
            "Sal" : 0,
            "Sev" : 1,
            "Maj2": 2,
            "Maj1": 3,
            "Mod" : 4,
            "Min2": 5,
            "Min1": 6,
            "Typ" : 7,
        }
    )
    #
    train["Floors_no"] = (train.loc[:,["1stFlrSF", "2ndFlrSF", "LowQualFinSF"]] != 0).sum(axis=1)
    test["Floors_no"]  = (train.loc[:,["1stFlrSF", "2ndFlrSF", "LowQualFinSF"]] != 0).sum(axis=1)

    # Drop
    train.drop(
        labels  =
        [
            "Utilities","Condition2","YearBuilt","RoofMatl","ExterCond","Heating","KitchenAbvGr","PoolArea","Fence",
            "MiscFeature","Street","Alley","PoolQC","LandSlope", "BsmtCond", "Id", "BsmtFinSF2","BsmtUnfSF",'1stFlrSF',
            '2ndFlrSF','WoodDeckSF','OpenPorchSF','EnclosedPorch',"Half_bath","Full_bath",'BsmtFullBath','BsmtHalfBath',
            'FullBath','HalfBath',
        ],
        axis    = 1,
        inplace = True
    )

    test.drop(
        labels  =
        [
            "Utilities","Condition2","YearBuilt","RoofMatl","ExterCond","Heating","KitchenAbvGr","PoolArea","Fence",
            "MiscFeature","Street","Alley","PoolQC","LandSlope", "BsmtCond", "Id", "BsmtFinSF2","BsmtUnfSF",'1stFlrSF',
            '2ndFlrSF','WoodDeckSF','OpenPorchSF','EnclosedPorch',"Half_bath","Full_bath",'BsmtFullBath','BsmtHalfBath',
            'FullBath','HalfBath',
        ],
        axis    = 1,
        inplace = True
    )

    # Impute
    QT = QuantileTransformer(output_distribution='normal')
    KI = KNNImputer()

    y_train = train.SalePrice
    x_train = train.drop(labels=["SalePrice"], axis=1)
    train_QT = pd.DataFrame(QT.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    test_QT  = pd.DataFrame(QT.transform(test), columns=test.columns, index=test.index)

    x_train_impute = KI.fit_transform(train_QT)
    x_test_impute  = KI.transform(test_QT)

    x_train_filled = pd.DataFrame(QT.inverse_transform(x_train_impute), columns=x_train.columns, index=x_train.index)
    x_test_filled  = pd.DataFrame(QT.inverse_transform(x_test_impute),  columns=test.columns,     index=test.index)

    return x_train_filled, y_train, x_test_filled, sample

if __name__ == "__name__":
    Native()
