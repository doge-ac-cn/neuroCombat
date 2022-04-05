from neuroCombat import neuroCombat
import pandas as pd
import numpy as np

# # 隐去个人信息
# data_ge = pd.read_csv('testdata/GE.csv', delimiter=",").iloc[:, 1:]
# data_philips = pd.read_csv('testdata/PHILIPS.csv', delimiter=",").iloc[:, 1:]
# data_ge.to_csv('testdata/GE.csv')
# data_philips.to_csv('testdata/PHILIPS.csv')

# 载入GE和飞利浦数据
data_ge = np.genfromtxt('testdata/GE.csv', delimiter=",", skip_header=1)[:, 2:]
data_philips = np.genfromtxt('testdata/PHILIPS.csv', delimiter=",", skip_header=1)[:, 2:]
print("合并前数据", data_ge.shape, data_philips.shape)

# 合并数据
data_summary = np.concatenate([data_ge, data_philips], axis=0).T
print("合并后数据：", data_summary.shape)

# 标识是ge还是飞利浦
batch = [0 for i in range(data_ge.shape[0])] + [1 for i in range(data_philips.shape[0])]
covars = {'batch': batch,
          }
covars = pd.DataFrame(covars)

# combat和谐化
data_combat = neuroCombat(dat=data_summary,
                          covars=covars,
                          batch_col='batch',
                          # categorical_cols=categorical_cols
                          )["data"]

# 保存GE和飞利浦数据
result_ge = data_combat.T[:data_ge.shape[0], :]
result_philips = data_combat.T[data_ge.shape[0]:, :]
print('转换后维度', result_ge.shape, result_philips.shape)

# 添加label列
label_ge = np.genfromtxt('testdata/GE.csv', delimiter=",", skip_header=1)[:, 1]
label_philips = np.genfromtxt('testdata/PHILIPS.csv', delimiter=",", skip_header=1)[:, 1]

result_ge = np.insert(result_ge, 0, label_ge, axis=1)
result_philips = np.insert(result_philips, 0, label_philips, axis=1)

# 添加一列用于留空位给患者名
result_ge = np.insert(result_ge, 0, 6666, axis=1)
result_philips = np.insert(result_philips, 0, 6666, axis=1)

# 添加title
title_ge = pd.read_csv('testdata/GE.csv', delimiter=",").columns
title_philips = pd.read_csv('testdata/PHILIPS.csv', delimiter=",").columns
result_ge = pd.DataFrame(result_ge, columns=title_ge)
result_philips = pd.DataFrame(result_philips, columns=title_philips)

# 保存
result_ge.to_csv('result/result_ge.csv', index=False)
result_philips.to_csv('result/result_philips.csv', index=False)
