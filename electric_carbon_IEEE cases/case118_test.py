# -*- coding: utf-8 -*-
"""
Created on 18/3 13:47 2023

@author: Zhiyuan

This file is to generate the carbon emission flow algorithm in the paper《电力系统碳排放流的计算方法初探》
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
from pandapower.control.controller.const_control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.timeseries.output_writer import OutputWriter
import numpy as np
import warnings
import case30_test as c30



def loadcase118():
    net = nw.case118()
    return net

def runpf(net):
    pp.runpp(net)
    return

def case118_CE():

    df = pd.read_excel('118casecarbon_emission.xlsx')
    EG = df[1].values.astype(float).reshape(-1,1)
    # print('debug helper')

    return EG



#---------------------------------------------------------------
#----------------- 2.1 支路潮流分布矩阵 --------------------------
#---------------------------------------------------------------
def matrix_PB(net):

    if len(net.trafo) == 0:
        # print('/-------------------------------------------------------------/')
        # print('*********** This network DOES NOT contain transformer **********')
        # print('/-------------------------------------------------------------/')
        N_bus = len(net.bus)
        PB = np.zeros((N_bus, N_bus))


        line_oder = net.line.loc[:, ['from_bus', 'to_bus']]
        line_pf = net.res_line.loc[:, 'p_from_mw']
        line_pf = pd.concat([line_oder, line_pf], axis=1, ignore_index=False)

        # trans_pf.columns = line_pf.columns
        #
        # powerflow = pd.concat([line_pf, trans_pf], axis=0, ignore_index=True)

        for row in line_pf.iterrows():
            if row[1]['p_from_mw'] > 0:
                x_coord = row[1]['from_bus'].astype(int)
                y_coord = row[1]['to_bus'].astype(int)
                PB[x_coord, y_coord] = row[1]['p_from_mw']
            else:
                x_coord = row[1]['to_bus'].astype(int)
                y_coord = row[1]['from_bus'].astype(int)
                PB[x_coord, y_coord] = np.absolute(row[1]['p_from_mw'])

            # print('debug helper')
        powerflow=line_pf




    else:
        # print('\n/-------------------------------------------------------/ ')
        # print('*********** This network CONTAINS transformers **********')
        # print('/-------------------------------------------------------/\n')
        # get the size of N
        N_bus = len(net.bus)
        PB = np.zeros((N_bus, N_bus))

        trans_oder = net.trafo.loc[:, ['hv_bus', 'lv_bus']]
        trans_pf = net.res_trafo.loc[:, 'p_hv_mw']
        trans_pf = pd.concat([trans_oder,trans_pf], axis=1, ignore_index=False)
        line_oder = net.line.loc[:, ['from_bus', 'to_bus']]
        line_pf = net.res_line.loc[:, 'p_from_mw']
        line_pf = pd.concat([line_oder, line_pf], axis=1, ignore_index=False)

        trans_pf.columns = line_pf.columns

        powerflow = pd.concat([line_pf, trans_pf], axis=0, ignore_index=True)

        for row in powerflow.iterrows():
            if row[1]['p_from_mw'] > 0:
                x_coord = row[1]['from_bus'].astype(int)
                y_coord = row[1]['to_bus'].astype(int)
                PB[x_coord, y_coord] = row[1]['p_from_mw']
            else:
                x_coord = row[1]['to_bus'].astype(int)
                y_coord = row[1]['from_bus'].astype(int)
                PB[x_coord, y_coord] = np.absolute(row[1]['p_from_mw'])

        # print('debug helper')

    return PB,powerflow

#---------------------------------------------------------------
#----------------- 2.2 机组注入分布矩阵 -------------------------
#---------------------------------------------------------------
def matrix_PG(net):
    gen_order = net.gen.loc[:,['bus']]
    gen_reslt = net.res_gen.loc[:,'p_mw']
    gen_reslt = pd.concat([gen_order,gen_reslt], axis=1, ignore_index=False)
    slack_order = net.ext_grid.loc[:,'bus']
    slack_gen = net.res_ext_grid.loc[:,'p_mw']
    slack_reslt = pd.concat([slack_order, slack_gen], axis=1, ignore_index=False)

    # !!! 14节点算例的slack bus 为 0 ；如果 slack bus 不为 0，!!!需要调整!!!
    gen_df = pd.concat([slack_reslt, gen_reslt], axis=0, ignore_index=True)

    N_bus = len(net.bus)
    N_gen = len(gen_df)

    PG = np.zeros((N_gen, N_bus))

    for index, row in gen_df.iterrows():
        x_coord = index
        y_coord = row['bus'].astype(int)
        PG[x_coord,y_coord] = row['p_mw']

    # print('debug helper')
    return PG

#---------------------------------------------------------------
#----------------- 2.3 负荷分布矩阵 ------------------------------
#---------------------------------------------------------------
def matrix_PL(net):
    N_bus = len(net.bus)
    N_load = len(net.load)

    PL = np.zeros([N_load, N_bus])

    for index, row in net.load.iterrows():
        x_coord = index
        y_coord = row['bus']
        PL[x_coord, y_coord] = row['p_mw']

    return PL


#---------------------------------------------------------------
#----------------- 2.4 节点有功通量矩阵 --------------------------
#---------------------------------------------------------------
def matrix_PN(PB, PG):
    PZ = np.vstack((PB,PG))
    sigma = np.ones((1, PZ.shape[0]))
    temp = sigma@PZ


    # 判断是否有节点的注入功率为零
    if np.any(temp[0] == 0):
        position = np.where(temp[0]== 0)

        # print('#---------------------------------#')
        # print('存在注入功率为零的节点。节点编号：', position[0], sep='\n')
        # print('节点功率注入设置 0.001MW ')
        # print('#---------------------------------#')
        for i in position:
            temp[0][i]=0.001

    else:
        a = 1
        # print('#---------------------------------#')
        # print('所有节点均为非零功率注入 ')
        # print('#---------------------------------#')

    PN = np.diag(temp[0]) # 取0 降维

    return PN




#---------------------------------------------------------------
# --------------------- 2.6 节点碳势向量 -------------------------
#---------------------------------------------------------------
def matrix_EN(PN, PB, PG, EG):

    EN = np.linalg.inv(PN-PB.T)@PG.T@EG

    # print('debug helper')

    return EN


#---------------------------------------------------------------
# --------------------- 2.7 支路碳流率分布矩阵 --------------------
#---------------------------------------------------------------
def matrix_RB(EN, PB):
    temp = np.diag(EN[:,0])
    RB = PB@temp

    # print('debug helper')

    return RB

#---------------------------------------------------------------
# --------------------- 2.8 负荷碳流率向量 -----------------------
#---------------------------------------------------------------

def matrix_RL(PL, EN):
    RL = PL@EN

    return RL

#---------------------------------------------------------------
# ---------- 2.9 支路碳流密度（支路碳流率RB./支路功率PB） -------------
#---------------------------------------------------------------
def matrix_RPB(RB, PB):

    RPB = RB/PB

    return RPB



# ==============================================================
#                    Write Table 2
#       used for import functions of outer py file
# ==============================================================

def create_table_2(PN_,EN_,net ):


    diagPN = np.diag(PN_)
    table2 = np.c_[diagPN, EN_[:, 0]]
    bus_name = net.bus['name']  # it's a series type
    df1_1 = pd.DataFrame(table2, columns=['节点有功通量PN', '节点碳势EN'])
    df1_2 = df1_1.merge(bus_name, left_index=True, right_index=True)
    bus_order = df1_2['name']
    df1_2.drop(labels=['name'], axis=1, inplace=True)
    df1_2.insert(0, 'name', bus_order)  # final output
    table2 = df1_2.rename(columns={'name': 'bus_name'})


    return table2


# ==============================================================
#                    Write Table 3
#       used for import functions of outer py file
# ==============================================================
def create_table_3(branch_flows, PB_, RPB_, RB_):

    branch_orders = branch_flows.loc[:, ['from_bus', 'to_bus']]
    branch_orders = branch_orders + 1
    PB_tri_lower = np.tril(PB_, -1)
    PB_delta = PB_ - PB_tri_lower.T

    RPB_[np.isnan(RPB_)] = 0
    RPB_delta = RPB_ + np.tril(RPB_, -1).T

    RB_delta = RB_ - np.tril(RB_, -1).T

    df3 = np.zeros((len(branch_orders), 3))
    for index, row in branch_orders.iterrows():
        df3[index][0] = PB_delta[row[0] - 1][row[1] - 1]
        df3[index][1] = RPB_delta[row[0] - 1][row[1] - 1]
        df3[index][2] = RB_delta[row[0] - 1][row[1] - 1]

    table3_1 = pd.DataFrame(df3, columns=['有功潮流PB', '碳流密度RPB', '碳流率RB'])

    # double check the power flow values of the branches, 是否为零
    zero_list = table3_1[table3_1['有功潮流PB'] < 0.001].index.to_list()
    if zero_list:
        for i in range(len(zero_list)):
            table3_1.loc[zero_list[i], '碳流密度RPB'] = 0

        table3 = pd.concat([branch_orders, table3_1], axis=1, ignore_index=False)
    else:
        table3 = pd.concat([branch_orders, table3_1], axis=1, ignore_index=False)


    return table3





# ==============================================================
#                    Write Table 4
#       used for import functions of outer py file
# ==============================================================
def create_table_4(bus_name,net,EG_, RL_):
    load_list = np.zeros(len(bus_name))
    load_bus = net.load['bus']
    j = 0
    for i in load_bus:
        load_list[i - 1] = RL_[j, 0]
        j += 1

    Name_bus = bus_name.to_frame()
    df_Load = pd.DataFrame({'Carbon on Loads': load_list})

    list_genbus = net.gen['bus'].to_list()
    list_slackbus = net.ext_grid['bus'].to_list()
    list_GEN = list_slackbus + list_genbus
    list_gen_output = net.res_gen['p_mw'].to_list()
    list_slack_output = net.res_ext_grid['p_mw'].to_list()
    list_GEN_output = list_slack_output + list_gen_output
    list_GEN_output = np.array(list_GEN_output)
    list_input_carbon = EG_[:, 0] * list_GEN_output

    GEN_carbon_list = np.zeros(len(bus_name))

    j = 0

    for i in list_GEN:
        GEN_carbon_list[i] = list_input_carbon[j]

        j += 1

    df_GEN = pd.DataFrame({'Carbon on Gens': GEN_carbon_list})
    table4 = pd.concat([Name_bus, df_Load, df_GEN], axis=1, ignore_index=False)
    table4 = table4.rename(columns={'name': 'bus_name'})
    return table4


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    net = loadcase118()
    runpf(net)

    PB_, branch_flows = matrix_PB(net)
    PG_ = matrix_PG(net)
    PL_ =  matrix_PL(net)
    PN_ = matrix_PN(PB_, PG_)

    # read CSV files of carbon emission intensity vector
    EG_ = case118_CE()
    EN_ = matrix_EN(PN_, PB_, PG_, EG_)
    RB_ = matrix_RB(EN_, PB_)
    RL_ = matrix_RL(PL_, EN_)
    RPB_ = matrix_RPB(RB_, PB_)

    # Write and output functions
    # 1) 节点有功通量PN 与 节点碳势EN
    diagPN = np.diag(PN_)
    table2 = np.c_[diagPN, EN_[:, 0]]
    bus_name = net.bus['name']  # it's a series type
    df1_1 = pd.DataFrame(table2, columns=['节点有功通量PN', '节点碳势EN'])
    df1_2 = df1_1.merge(bus_name, left_index=True, right_index=True)
    bus_order = df1_2['name']
    df1_2.drop(labels=['name'], axis=1, inplace=True)
    df1_2.insert(0, 'name', bus_order)  # final output
    table2 = df1_2.rename(columns={'name': 'bus_name'})

    # 2) 支路有功潮流PB、碳流密度RPB、碳流率RB
    branch_orders = branch_flows.loc[:, ['from_bus', 'to_bus']]
    branch_orders = branch_orders + 1
    PB_tri_lower = np.tril(PB_, -1)
    PB_delta = PB_ - PB_tri_lower.T

    RPB_[np.isnan(RPB_)] = 0
    RPB_delta = RPB_ + np.tril(RPB_, -1).T

    RB_delta = RB_ - np.tril(RB_, -1).T

    # PB_in_order = PB_[PB_ != 0].reshape(-1, 1)
    # RPB_in_order = RPB_[~np.isnan(RPB_)].reshape(-1, 1)
    # RB_in_order = RB_[RB_ != 0].reshape(-1, 1)

    # df1 = pd.DataFrame(branch_orders, columns=['From_bus', 'To_bus'])
    df3 = np.zeros((len(branch_orders), 3))
    for index, row in branch_orders.iterrows():
        df3[index][0] = PB_delta[row[0] - 1][row[1] - 1]
        df3[index][1] = RPB_delta[row[0] - 1][row[1] - 1]
        df3[index][2] = RB_delta[row[0] - 1][row[1] - 1]

    # table3_2 = np.c_[PB_in_order, RPB_in_order, RB_in_order]

    table3_1 = pd.DataFrame(df3, columns=['有功潮流PB', '碳流密度RPB', '碳流率RB'])

    # double check the power flow values of the branches, 是否为零
    zero_list = table3_1[table3_1['有功潮流PB'] < 0.001].index.to_list()
    if zero_list:
        for i in range(len(zero_list)):
            table3_1.loc[zero_list[i], '碳流密度RPB'] = 0

        table3 = pd.concat([branch_orders, table3_1], axis=1, ignore_index=False)
    else:
        table3 = pd.concat([branch_orders, table3_1], axis=1, ignore_index=False)

    # 3）负荷碳流率 和 机组注入碳流率
    load_list = np.zeros(len(bus_name))
    load_bus = net.load['bus']
    j = 0
    for i in load_bus:
        load_list[i - 1] = RL_[j, 0]
        j += 1

    Name_bus = bus_name.to_frame()
    df_Load = pd.DataFrame({'Carbon on Loads': load_list})

    list_genbus = net.gen['bus'].to_list()
    list_slackbus = net.ext_grid['bus'].to_list()
    list_GEN = list_slackbus + list_genbus
    list_gen_output = net.res_gen['p_mw'].to_list()
    list_slack_output = net.res_ext_grid['p_mw'].to_list()
    list_GEN_output = list_slack_output + list_gen_output
    list_GEN_output = np.array(list_GEN_output)
    list_input_carbon = EG_[:, 0] * list_GEN_output

    GEN_carbon_list = np.zeros(len(bus_name))

    j = 0

    for i in list_GEN:
        GEN_carbon_list[i] = list_input_carbon[j]

        j += 1

    df_GEN = pd.DataFrame({'Carbon on Gens': GEN_carbon_list})
    table4 = pd.concat([Name_bus, df_Load, df_GEN], axis=1, ignore_index=False)
    table4 = table4.rename(columns={'name': 'bus_name'})
