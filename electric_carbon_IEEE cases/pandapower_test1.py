# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:55:13 2023

@author: Zhiyuan

This project is to test the pandapower as the major simulation platforms to develop
our carbon-emission studies.
net.load.at[0,'p_mw']=100 设置值
loc 不能设置，只能查看

tutorials on https://www.youtube.com/watch?v=sAHoJbfLhas&ab_channel=pandapower

"""
import pandapower.control
import pandapower.networks as nw
import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt
from pandapower.control.controller.const_control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.timeseries.output_writer import OutputWriter
import matplotlib
matplotlib.rc("font", family='simsun')

plt.rcParams['figure.constrained_layout.use'] = True
matplotlib.use('TkAgg')


# net = nw.case30()
#
# bus = net.bus
# gen_profile = net.gen
# load_profile = net.load
# bus_geo = net.bus_geodata
# line_profile = net.line
# bus_geo = net.bus_geodata
#
# pp.runpp(net)
#
# res1_bus = net.res_bus
# res1_branch = net.res_line
# res1_gen = net.res_gen
#
# net.load.at[0,'p_mw']=100
#
# pp.runpp(net)
#
# res_bus = net.res_bus
# res_branch = net.res_line
# res_gen = net.res_gen


# pandapower time-series simulations
# net1 = nw.example_simple()




# debugtesting --------------

# end debugging ------------------



# print(net1.gen)
# net1.gen.drop(0, inplace=True)
# pp.create_sgen(net1, 5, p_mw=1.0)   #
df = pd.read_json('cigre_timeseries_15min.json')
# df.loc[:, ['pv','wind', 'residential']].plot()

# ds = DFData(df)
df = df.values

# ConstControl(net1, 'sgen', 'p_mw', element_index=net1.sgen.index, profile_name=['wind', 'pv'], data_source=ds)
# ConstControl(net1, 'load', 'p_mw', element_index=net1.load.index, profile_name=['residential'], data_source=ds)

# OW = OutputWriter(net1, time_steps=(0, 96), output_path='./results/', output_file_type='.xlsx')

# OW.log_variable('res_bus', 'vm_pu', '')
# OW.log_variable('res_line', 'loading_percent')
# OW.log_variable('res_load', 'p_mw')

# run_timeseries(net1, time_steps=(range(96)))
# df = pd.read_excel('./results/res_line/loading_percent.xlsx', index_col=0)
# df.plot()
# plt.show()



##------- plotting ---------------------
x_ = range(96)
pv_ = df[:,1]
wnd_ = df[:,2]
load_ = df[:,3]

plt.figure(figsize=(9,5))
plt.plot(x_, pv_, linestyle='solid', color = 'coral', linewidth=1.5, label = '光伏曲线')
plt.plot(x_, wnd_, linestyle='solid', color = 'deepskyblue', linewidth=1.5, label = '风电曲线')
plt.plot(x_, load_, linestyle='solid', color = 'hotpink', linewidth=1.5, label = '负荷曲线')

plt.tick_params(labelsize=14)

plt.grid()
legend = plt.legend(loc = 'upper left', bbox_to_anchor=(0.005,1.0), fontsize=20, ncol = 1)

plt.savefig('plt_cigre.pdf', dpi = 150)

plt.show()






