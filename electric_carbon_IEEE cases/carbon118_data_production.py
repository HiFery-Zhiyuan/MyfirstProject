# -*- coding: utf-8 -*-
"""
Created on Mon Mar  13 14:04 2023

@author: Zhiyuan

This file is to generate testing data for PG/PL/EG/ and the corresponding table results
USING IEEE 14-bus system

# ********************** Learning Notes included **********************
pip list
pip show pandapower
hasattr(obj, attri) 判断是否有属性
np.array.shape
# ********************** Learning Notes included **********************


"""


import pandapower.control
import pandapower.networks as nw
import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import warnings

import case118_test as ct118

def read_excel():
    gl_profile = pd.read_excel('./carbon_emission_loop/gen_load_table.xlsx', index_col=0)
    EG_profile = pd.read_excel('./carbon_emission_loop/118case.xlsx', index_col=0)

    return EG_profile, gl_profile

def carb_lvrg(net, temp_EG, bus_name):

    pp.runpp(net)
    PB_, branch_flows = ct118.matrix_PB(net)
    PG_ = ct118.matrix_PG(net)
    PL_ = ct118.matrix_PL(net)
    PN_ = ct118.matrix_PN(PB_, PG_)
    EN_ = ct118.matrix_EN(PN_, PB_, PG_, temp_EG)
    RB_ = ct118.matrix_RB(EN_, PB_)
    RL_ = ct118.matrix_RL(PL_, EN_)
    RPB_ = ct118.matrix_RPB(RB_, PB_)
    t2 = ct118.create_table_2(PN_, EN_, net)
    t3 = ct118.create_table_3(branch_flows, PB_, RPB_, RB_)
    t4 = ct118.create_table_4(bus_name, net, temp_EG, RL_)

    return t2, t3, t4



if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    load_list = [2,3]
    gen_list = [1,2]  # 在第1和第2个gen节点修改gen profile

    net = ct118.loadcase118()
    bus_name = net.bus['name']
    EG_profile, gl_profile = read_excel()
    EG_matrix = EG_profile.values
    gl_matrix = gl_profile.values

    width_1, length_1 = EG_matrix.shape
    length_2, width_2 = gl_matrix.shape
    temp_i = []

    for i in range(length_1):
        temp_EG = EG_matrix[:, i].reshape(-1, 1).astype(float)

        print(i)

        temp_j = []

        for j in range(800):                 # 一共10*800 个训练数据

            net = ct118.loadcase118()

            # 在第2和第6个load节点处修改load profile
            # net.load.at[2, 'p_mw'] = net.load.at[2, 'p_mw'] + gl_matrix[j][1]
            # net.load.at[6, 'p_mw'] = gl_matrix[j][2]
            net.load.loc[:, 'p_mw'] = net.load['p_mw'] + gl_matrix[j][1] / 20

            # 在第1个gen节点修改gen profile
            net.gen.loc[:, 'p_mw'] = net.gen['p_mw'] + gl_matrix[j][3]


            table2, table3, table4 = carb_lvrg(net, temp_EG,bus_name)


            # save the data
            carbon_input = temp_EG.reshape(-1, 1)

            gen_list = net.gen['p_mw'].values
            slack_gen = net.res_ext_grid.loc[0, 'p_mw']
            slack_gen = np.round(slack_gen, 2)
            gen_input = np.append(slack_gen, gen_list).reshape(-1,1)
            load_input = net.load['p_mw'].values.reshape(-1,1)
            temp_input = np.concatenate((carbon_input, gen_input, load_input), axis=0).T

            t2_data1 = np.array(np.round(table2.values[:, 1].tolist(), 2)).reshape(-1, 1)
            t2_data2 = np.array(np.round(table2.values[:, 2].tolist(), 2)).reshape(-1, 1)

            t3_data1 = np.array(np.round(table3.values[:, 2].tolist(), 2)).reshape(-1, 1)
            t3_data2 = np.array(np.round(table3.values[:, 3].tolist(), 2)).reshape(-1, 1)
            t3_data3 = np.array(np.round(table3.values[:, 4].tolist(), 2)).reshape(-1, 1)

            t4_data1 = np.array(np.round(table4.values[:, 1].tolist(), 2)).reshape(-1, 1)
            t4_data2 = np.array(np.round(table4.values[:, 2].tolist(), 2)).reshape(-1, 1)

            temp_output = np.concatenate((t2_data1, t2_data2, t3_data1, t3_data2,t3_data3,t4_data1, t4_data2), axis=0).T
            temp_list = np.concatenate((temp_input, temp_output), axis=1)

            if len(temp_j) != 0:
                temp_j = np.vstack((temp_j, temp_list))
            else:
                temp_j = np.append(temp_j, temp_list)


        if len(temp_i) != 0:
            temp_i = np.vstack((temp_i, temp_j))
        else:
            temp_i = temp_j

    temp_i = np.around(temp_i, 3)
    np.savetxt("./case118_data_rslt/data_origin_118_8000.csv", temp_i, delimiter=",",fmt='%.3f')
