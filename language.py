#!/usr/bin/python
#-*- coding:UTF-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

def createAobject():
    a = A()

class A:
    __count = 0
    count = 0
    def __init__(self):
        self.__class__.__count += 1
        self.__class__.count = self.__class__.__count

print(A.count)
createAobject()
print(A.count)
#datatime timedelta
t = dt.timedelta(days=10)
print(t)
date = '2016/10/3'
test_date = dt.datetime.strptime(date, '%Y/%m/%d').date()
b = test_date - t
print(b)
#dataFrame dropna
df = pd.DataFrame(np.random.randn(6,3),columns=['one','two','three'])
df.iloc[1, 0] = np.nan
df.iloc[2, 1] = np.nan
df.iloc[3, 2] = np.nan
print(df)
dfdrop = df['one'].dropna()
print(dfdrop)
dfdrop_shift = dfdrop.shift()
print(dfdrop_shift)
dfdrop_diff = dfdrop - dfdrop_shift
print (dfdrop_diff)

