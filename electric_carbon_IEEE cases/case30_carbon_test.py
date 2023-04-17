# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:55:13 2023

@author: Zhiyuan

This project is to test the pandapower as the major simulation platforms to develop a time-series simulations using
IEEE case30 testcases

net.load.at[0,'p_mw']=100 设置值
loc 不能设置，只能查看

tutorials on https://www.youtube.com/watch?v=sAHoJbfLhas&ab_channel=pandapower



**************************************************************************************************
Important notes: the profile size should be equal to the size of the testing cases(load and generators)
ConstControl里Profile的设置必须要跟我们需要改变的集合一致
**************************************************************************************************


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
import numpy as np


# time-serious simulations using case30 cases

# load testing cases
def loadcase30():
    net = nw.case30()
    return net


# remove and add the static generators on the certain bus node
def gen_modify(net_x, index):
    bus_number = net_x.gen.loc[index, 'bus']
    net_x.gen.drop(index, inplace=True)
    pp.create_sgen(net_x, bus_number, p_mw=10.0)
    return net_x


# define the time series control and write functions for testing cases
def control_func(gen_, bus_, net):
    df = pd.read_json('cigre_timeseries_15min.json')
    ds = DFData(df) # make it to pandapower readable
    N_time_steps = len(ds.df)


    ConstControl(net, 'sgen', 'p_mw', element_index=net.sgen.index, profile_name=['pv'], data_source=ds)

    mask = net.load.loc[net.load['bus'].isin(bus_)]

    mask.reset_index(drop=True, inplace=True)
    mask.index = list(mask.index)

    ConstControl(net, 'load', 'p_mw', element_index=mask.index, profile_name=['residential', 'residential'], data_source=ds)

    OW = OutputWriter(net, time_steps=(0, 96), output_path='./results/', output_file_type='.xlsx')

    # run_timeseries(net, time_steps=(range(96)))

    return N_time_steps, net

# time series running functions
def ts_run(ts_net, N_time_steps):
    run_timeseries(ts_net, time_steps=(range(N_time_steps)))

    return

if __name__ == "__main__":
    net = loadcase30()
    gen_index = 1
    load_bus = [3, 6]
    net_mdf = gen_modify(net, gen_index)
    N_time_steps, ts_net = control_func(gen_index, load_bus, net_mdf)
    ts_run(ts_net, N_time_steps)