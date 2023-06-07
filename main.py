
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn import linear_model

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error


# 这里我包装了一个异常值处理的代码，可以随便调用。
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n


# 数据读取
Train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
TestA_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
#以power属性为例，使用箱线图删除异常值
Train_data = outliers_proc(Train_data, 'power', scale=3)

#以下两个特征严重倾斜，删除。
del Train_data["seller"]
del Train_data["offerType"]
del TestA_data["seller"]
del TestA_data["offerType"]

print(Train_data.shape)
print(TestA_data.shape)
Train_data.head()
numerical_cols = Train_data.select_dtypes(exclude='object').columns
print(numerical_cols)
feature_cols = [col for col in numerical_cols if col not in ['SaleID', 'name', 'regDate', 'price']]
X_data = Train_data[feature_cols]
Y_data = Train_data['price']

X_test = TestA_data[feature_cols]

print('X train shape:', X_data.shape)
print('X test shape:', X_test.shape)


def Sta_inf(data):
    print('_min', np.min(data))
    print('_max:', np.max(data))
    print('_mean', np.mean(data))
    print('_ptp', np.ptp(data))
    print('_std', np.std(data))
    print('_var', np.var(data))


print('Sta of label:')
Sta_inf(Y_data)
X_data = X_data.fillna(-1)
X_test = X_test.fillna(-1)


def build_model_lr(x_train, y_train):
    reg_model = linear_model.LinearRegression()
    reg_model.fit(x_train, y_train)
    return reg_model


def build_model_ridge(x_train, y_train):
    reg_model = linear_model.Ridge(alpha=0.8)  # alphas=range(1,100,5)
    reg_model.fit(x_train, y_train)
    return reg_model


#LassoCV是沿着正则化路径迭代拟合的Lasso线性模型，主要就是基于Lasso的方法，加上KFold交叉验证来自动帮我们找出最优的alpha取值以及其对应最优模型。
def build_model_lasso(x_train, y_train):
    reg_model = linear_model.LassoCV()
    reg_model.fit(x_train, y_train)
    return reg_model


def build_model_gbdt(x_train, y_train):
    estimator = GradientBoostingRegressor(subsample=0.85, max_depth=5, n_estimators=100)
    param_grid = {
        'learning_rate': [0.05, 0.08, 0.1, 0.2],
    }
    gbdt = GridSearchCV(estimator, param_grid, cv=3)
    try:
        gbdt.fit(x_train, y_train)
        print(gbdt.best_params_)
    # print(gbdt.best_estimator_ )
        return gbdt
    except Exception as e:
        print("error:", e)

def build_model_xgb(x_train, y_train):
    model = xgb.XGBRegressor(n_estimators=120, learning_rate=0.08, gamma=0, subsample=0.8, \
                             colsample_bytree=0.9, max_depth=5)  # , objective ='reg:squarederror'
    try:
        model.fit(x_train, y_train)
        return model
    except Exception as e:
        print("error:", e)

def build_model_lgb(x_train, y_train):
    estimator = lgb.LGBMRegressor(num_leaves=63, n_estimators=100)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
    }
    gbm = GridSearchCV(estimator, param_grid)
    try:
        gbm.fit(x_train, y_train)
        return gbm
    except Exception as e:
        print("error:", e)

# XGBoost的五折交叉回归验证实现
## xgb
xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, subsample=0.8,
                       colsample_bytree=0.9, max_depth=7)  # ,objective ='reg:squarederror'

scores_train = []
scores = []

## 5折交叉验证方式
sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_ind, val_ind in sk.split(X_data, Y_data):
    train_x = X_data.iloc[train_ind].values
    train_y = Y_data.iloc[train_ind]
    val_x = X_data.iloc[val_ind].values
    val_y = Y_data.iloc[val_ind]

    xgr.fit(train_x, train_y)
    pred_train_xgb = xgr.predict(train_x)
    pred_xgb = xgr.predict(val_x)

    score_train = mean_absolute_error(train_y, pred_train_xgb)
    scores_train.append(score_train)
    score = mean_absolute_error(val_y, pred_xgb)
    scores.append(score)

print('Train mae:', np.mean(score_train))
print('Val mae', np.mean(scores))

# 划分数据集，并用多种方法训练和预测
## Split data with val
x_train, x_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.3)

## Train and Predict
print('Predict LR...')
model_lr = build_model_lr(x_train, y_train)
val_lr = model_lr.predict(x_val)
subA_lr = model_lr.predict(X_test)

print('Predict Ridge...')
model_ridge = build_model_ridge(x_train, y_train)
val_ridge = model_ridge.predict(x_val)
subA_ridge = model_ridge.predict(X_test)

print('Predict Lasso...')
model_lasso = build_model_lasso(x_train, y_train)
val_lasso = model_lasso.predict(x_val)
subA_lasso = model_lasso.predict(X_test)

print('Predict GBDT...')
model_gbdt = build_model_gbdt(x_train, y_train)
val_gbdt = model_gbdt.predict(x_val)
subA_gbdt = model_gbdt.predict(X_test)

# 一般比赛中效果最为显著的两种方法
print('predict XGB...')
model_xgb = build_model_xgb(x_train, y_train)
val_xgb = model_xgb.predict(x_val)
subA_xgb = model_xgb.predict(X_test)

print('predict lgb...')
model_lgb = build_model_lgb(x_train, y_train)
val_lgb = model_lgb.predict(x_val)
subA_lgb = model_lgb.predict(X_test)

print('Sta inf of lgb:')
Sta_inf(subA_lgb)


def Weighted_method(test_pre1, test_pre2, test_pre3, w=[1 / 3, 1 / 3, 1 / 3]):
    Weighted_result = w[0] * pd.Series(test_pre1) + w[1] * pd.Series(test_pre2) + w[2] * pd.Series(test_pre3)
    return Weighted_result


## Init the Weight
w = [0.3, 0.4, 0.3]

## 测试验证集准确度
val_pre = Weighted_method(val_lgb, val_xgb, val_gbdt, w)
MAE_Weighted = mean_absolute_error(y_val, val_pre)
print('MAE of Weighted of val:', MAE_Weighted)

## 预测数据部分
subA = Weighted_method(subA_lgb, subA_xgb, subA_gbdt, w)
print('Sta inf:')
Sta_inf(subA)
## 生成提交文件
sub = pd.DataFrame()
sub['SaleID'] = X_test.index
sub['price'] = subA
sub.to_csv('./sub_Weighted.csv', index=False)



## Starking

## 第一层
train_lgb_pred = model_lgb.predict(x_train)
train_xgb_pred = model_xgb.predict(x_train)
train_gbdt_pred = model_gbdt.predict(x_train)

Strak_X_train = pd.DataFrame()
Strak_X_train['Method_1'] = train_lgb_pred
Strak_X_train['Method_2'] = train_xgb_pred
Strak_X_train['Method_3'] = train_gbdt_pred

Strak_X_val = pd.DataFrame()
Strak_X_val['Method_1'] = val_lgb
Strak_X_val['Method_2'] = val_xgb
Strak_X_val['Method_3'] = val_gbdt

Strak_X_test = pd.DataFrame()
Strak_X_test['Method_1'] = subA_lgb
Strak_X_test['Method_2'] = subA_xgb
Strak_X_test['Method_3'] = subA_gbdt
## level2-method
model_lr_Stacking = build_model_lr(Strak_X_train,y_train)
## 训练集
train_pre_Stacking = model_lr_Stacking.predict(Strak_X_train)
print('MAE of Stacking-LR:',mean_absolute_error(y_train,train_pre_Stacking))

## 验证集
val_pre_Stacking = model_lr_Stacking.predict(Strak_X_val)
print('MAE of Stacking-LR:',mean_absolute_error(y_val,val_pre_Stacking))

## 预测集
print('Predict Stacking-LR...')
subA_Stacking = model_lr_Stacking.predict(Strak_X_test)

subA_Stacking[subA_Stacking<10]=10  ## 去除过小的预测值

sub = pd.DataFrame()
sub['SaleID'] = TestA_data.SaleID
sub['price'] = subA_Stacking
sub.to_csv('./sub_Stacking.csv',index=False)

print('Sta inf:')
Sta_inf(subA_Stacking)