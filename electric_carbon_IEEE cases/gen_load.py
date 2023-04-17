# -*- coding: utf-8 -*-
"""

This is to generate gen & load profile
# Zhiyuan @ 13/3 2023 15:21

"""


import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

start = 100
end = 120


df = pd.read_json('cigre_timeseries_15min.json')
n = 100

coe3 = np.random.uniform(start, end, size=n)
coe3 = np.round(coe3, 2)

coe1 = np.random.uniform(100, 150, size=n)
coe1 = np.round(coe1, 2)

coe2 = np.random.uniform(100, 150, size=n)
coe2 = np.round(coe2, 2)

gen_load = df.values


load_1 = []
load_2 = []
gen_1 = []

origin_load = gen_load[:, 3]
origin_pv = gen_load[:, 1]
origin_wind = gen_load[:, 2]

for i in range(len(coe3)):
    load_1 = np.hstack((load_1, origin_load*coe3[i]))
    gen_1 = np.hstack((gen_1, origin_pv*coe1[i]))
    load_2 = np.hstack((load_2, origin_wind * coe2[i]))


load_1 = np.round(load_1, 2)
load_2 = np.round(load_2, 2)
gen_1 = np.round(gen_1, 2)


plt.plot(load_1[0:96])
plt.plot(load_2[0:96])
plt.plot(gen_1[0:96])
plt.show()

gen_load_mtrx = np.vstack((load_1, load_2, gen_1)).T
time_step = [x for x in range(len(gen_1))]
time_step_df = pd.DataFrame(time_step, columns=['time_steps'])
gen_load_table = pd.DataFrame(gen_load_mtrx, columns = ['load1', 'load2','gen1'])
gen_load_table = pd.concat([time_step_df, gen_load_table], axis=1)

gen_load_table.to_excel('./carbon_emission_loop/gen_load_table.xlsx')


# gen_load_table = pd.DataFrame([load_1.T, load_2.T, gen_1.T], columns = ['load1', 'load2''gen1'])




