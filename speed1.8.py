import numpy as np
import pandas as pd
import coptpy as cp
import time
import os

def read_scuc_data(folder_path):
    """读取SCUC所需的所有输入数据"""
    
    # 读取负荷数据 slf.txt
    load_data = pd.read_csv(rf'{folder_path}/slf.txt', sep=r'\s+', header=None, encoding='gb2312', skiprows=1)
    load = load_data[1].astype(float).values  # 24时段系统负荷数据
    T_num = 24  # 时段数

    # 读取机组参数 unitdata.txt
    unit_data = pd.read_csv(rf'{folder_path}/unitdata.txt', sep=r'\s+', header=None, encoding='gb2312', skiprows=1)
    unit_data = unit_data.astype(float).values
    # ubus = unit_data[:, 1]  # 所处节点
    Pmax = unit_data[:, 2]  # 最大出力
    Pmin = unit_data[:, 3]  # 最小出力
    Ru = unit_data[:, 4]    # 上爬坡率
    Rd = unit_data[:, 5]    # 下爬坡率
    To = unit_data[:, 6].astype(int)    # 最小开机时间（转为整数）
    Tf = unit_data[:, 7].astype(int)    # 最小停机时间（转为整数）
    Co = unit_data[:, 8]    # 启动成本
    U0 = unit_data[:, 9].astype(int)    # 初始状态（转为整数）
    P0 = unit_data[:, 10]   # 初�����力
    Tc = unit_data[:, 11].astype(int)   # 初始状态持续时间（转为整数）

    unit_num = len(unit_data)  # 机组数量

    # 读取报价容量数据 bidcapacity.txt - 移到这里
    bid_data = pd.read_csv(rf'{folder_path}/bidcapacity.txt', sep=r'\s+', header=None, encoding='gb2312', skiprows=1)
    bid_data = bid_data.iloc[:, 1:].astype(float).values  # 机组报价功率段

    # 读取报价数据 bidprice.txt
    price_data = pd.read_csv(rf'{folder_path}/bidprice.txt', sep=r'\s+', header=None, encoding='gb2312', skiprows=1)
    point_price = price_data.iloc[:, 2:].astype(float).values
    bid_segments = bid_data.shape[1]  # 获取实际的报价段数
    price = np.zeros((unit_num, bid_segments + 1))  # +1 是因为包含最小出力的价格
    
    # 计算分段价格
    price[:, 0] = point_price[:, 0] / Pmin  # 最小出力的单位价格
    for j in range(bid_segments):
        if j < point_price.shape[1]-1:
            price[:, j+1] = (point_price[:, j+1] - point_price[:, j]) / bid_data[:, j]

    # 新增储能数据读取 - 移到前面来
    # 读取储能基础数据 storagebasic.txt
    sto_data = pd.read_csv(rf'{folder_path}/storagebasic.txt', sep=r'\s+', header=None, encoding='gb2312', skiprows=1)
    sto_data = sto_data.astype(float).values
    sto_num = len(sto_data)  # 储能机组数量
    Tp = sto_data[:, 1].astype(int)    # 最小抽水时段
    Ts = sto_data[:, 2].astype(int)    # 最小停机时段
    Tg = sto_data[:, 3].astype(int)    # 最小发电时段
    Ci = sto_data[:, 4]    # 初始容量(MWh)
    Cf = sto_data[:, 5]    # 终止容量(MWh)
    Cm = sto_data[:, 6]    # 最大容量(MWh)
    Ec = sto_data[:, 7]    # 充放电效率
    Pf = sto_data[:, 8]    # 抽水固定功率(MW)
    Pgmax = sto_data[:, 9] # 最大发电率(MW)
    Pgmin = sto_data[:, 10]# 最小发电功率(MW)

    # 读取储能报价容量 stbidcapacity.txt
    sto_bid_data = pd.read_csv(rf'{folder_path}/stbidcapacity.txt', sep=r'\s+', header=None, encoding='gb2312', skiprows=1)
    sto_bid_data = sto_bid_data.iloc[:, 1:].astype(float).values

    # 读取储能报价 stbidprice.txt
    sto_price_data = pd.read_csv(rf'{folder_path}/stbidprice.txt', sep=r'\s+', header=None, encoding='gb2312', skiprows=1)
    sto_point_price = sto_price_data.iloc[:, 1:].astype(float).values
    
    # 计算储能分段价格
    sto_price = np.zeros((sto_num, 6))
    sto_price[:, 0] = sto_point_price[:, 0] / Pf  # 充电费用
    for j in range(1, 5):
        sto_price[:, j+1] = (sto_point_price[:, j+1] - sto_point_price[:, j]) / sto_bid_data[:, j]

    # 构建分段点数据 - 到储能数据之后
    point = np.zeros((unit_num, 7))  # 6个分段点
    point[:, 0] = 0  # 第一个点为0
    point[:, 1] = Pmin  # 第二个点为最小出力
    for j in range(5):  # 后续点为累加分段容量
        point[:, j+2] = point[:, j+1] + bid_data[:, j]
    
    # 构建储能分段点数据
    sto_point = np.zeros((sto_num, 6))  # 5个分段点
    sto_point[:, 0] = 0  # 第一个点为0
    sto_point[:, 1] = Pgmin  # 第二个点为最小发电功率
    for j in range(4):  # 后续点为累加分段容量
        sto_point[:, j+2] = sto_point[:, j+1] + sto_bid_data[:, j]

    # 读取断面数据 section.txt
    section_data = pd.read_csv(rf'{folder_path}/section.txt', sep=r'\s+', header=None, encoding='gb2312', skiprows=1)
    section_data = section_data.astype(float).values
    sec_num = section_data[:, 1].astype(int)  # 断面编号
    Plmax = section_data[:, 2]  # 断面

    # 读取branch_1.log文件
    with open(rf'{folder_path}/branch_1.log', 'r', encoding='gb2312') as f:
        log_content = f.read()

    # 提取灵敏度数据
    sens_start = log_content.find('<BranchUnitSensi::dky')
    sens_end = log_content.find('</BranchUnitSensi::dky>')
    sens_block = log_content[sens_start:sens_end].split('\n')[2:-1]
    
    line_num = len(sens_block)
    sens = np.zeros((unit_num, line_num))
    
    for i, line in enumerate(sens_block):
        data = line.strip('#').strip().split()
        sens[:, i] = [float(x) for x in data[3:]]
    
    sens = sens.T

    # 提取支路数据
    branch_start = log_content.find('<BranchData::dky')
    branch_end = log_content.find('</BranchData::dky>')
    branch_block = log_content[branch_start:branch_end].split('\n')[2:-1]
    
    sens_list = np.zeros(line_num * T_num)
    for i, line in enumerate(branch_block):
        data = line.strip('#').strip().split()
        sens_list[i] = float(data[-1])

    sum_sens = np.zeros((line_num, T_num))
    for i in range(line_num):
        sum_sens[i, :] = sens_list[T_num*i:T_num*(i+1)]

    # 仅提取关键断面的灵敏度信息
    sens = sens[sec_num-1, :]
    sum_sens = sum_sens[sec_num-1, :]

    # 新增储能数据读取
    # 读取储能基础数据 storagebasic.txt
    sto_data = pd.read_csv(rf'{folder_path}/storagebasic.txt', sep=r'\s+', header=None, encoding='gb2312', skiprows=1)
    sto_data = sto_data.astype(float).values
    Tp = sto_data[:, 1].astype(int)    # 最小抽水时段
    Ts = sto_data[:, 2].astype(int)    # 最小停机时段
    Tg = sto_data[:, 3].astype(int)    # 最小发电时段
    Ci = sto_data[:, 4]    # 初始容量(MWh)
    Cf = sto_data[:, 5]    # 终止容量(MWh)
    Cm = sto_data[:, 6]    # 最大容量(MWh)
    Ec = sto_data[:, 7]    # 充放电效率
    Pf = sto_data[:, 8]    # 抽水固定功率(MW)
    Pgmax = sto_data[:, 9] # 最大发电率(MW)
    Pgmin = sto_data[:, 10]# 最小发电功率(MW)

    sto_num = len(sto_data)  # 储能机组数量

    # 读取储能报价容量 stbidcapacity.txt
    sto_bid_data = pd.read_csv(rf'{folder_path}/stbidcapacity.txt', sep=r'\s+', header=None, encoding='gb2312', skiprows=1)
    sto_bid_data = sto_bid_data.iloc[:, 1:].astype(float).values

    # 读取储能报价 stbidprice.txt
    sto_price_data = pd.read_csv(rf'{folder_path}/stbidprice.txt', sep=r'\s+', header=None, encoding='gb2312', skiprows=1)
    sto_point_price = sto_price_data.iloc[:, 1:].astype(float).values
    
    # 计算储能分段价格
    sto_price = np.zeros((sto_num, 6))
    sto_price[:, 0] = sto_point_price[:, 0] / Pf  # 充电费用
    for j in range(1, 5):
        sto_price[:, j+1] = (sto_point_price[:, j+1] - sto_point_price[:, j]) / sto_bid_data[:, j]

    # 返回增加储能数据的元组
    return (load, T_num, unit_num, Pmax, Pmin, Ru, Rd, To, Tf, Co, U0, P0, Tc,
            bid_data, price, Plmax, sens, sum_sens,
            sto_num, Tp, Ts, Tg, Ci, Cf, Cm, Ec, Pf, Pgmax, Pgmin,
            sto_bid_data, sto_price, point, sto_point)

def build_scuc_model(data):
    """构建SCUC优化模型"""
    # 创建COPT环境和模型
    env = cp.Envr()
    m = env.createModel("scuc")
    
    # 解包数据
    unit_num = data['unit_num']
    T_num = data['T_num']
    sto_num = data['sto_num']
    load = data['load']
    Pmax = data['Pmax']
    Pgmax = data['Pgmax']
    Pgmin = data['Pgmin']
    Cm = data['Cm']
    Ci = data['Ci']
    Cf = data['Cf']
    Pf = data['Pf']
    Ec = data['Ec']
    sto_bid_data = data['sto_bid_data']
    sto_price = data['sto_price']
    Pmin = data['Pmin']
    Ru = data['Ru']
    Rd = data['Rd']
    To = data['To']
    Tf = data['Tf']
    Tg = data['Tg']
    Tp = data['Tp']
    Ts = data['Ts']
    Co = data['Co']
    U0 = data['U0']
    # P0 = data['P0']
    Tc = data['Tc']
    price = data['price']
    bid_data = data['bid_data']
    Plmax = data['Plmax']
    sens = data['sens']
    sum_sens = data['sum_sens']

    # 批量创建变量
    u = {(i,t): m.addVar(vtype=cp.COPT.BINARY, name=f'u_{i}_{t}') 
         for i in range(unit_num) for t in range(T_num)}
    v = {(i,t): m.addVar(vtype=cp.COPT.BINARY, name=f'v_{i}_{t}') 
         for i in range(unit_num) for t in range(T_num)}
    w = {(i,t): m.addVar(vtype=cp.COPT.BINARY, name=f'w_{i}_{t}') 
         for i in range(unit_num) for t in range(T_num)}
    P = {(i,t): m.addVar(lb=0, ub=Pmax[i], name=f'P_{i}_{t}') 
         for i in range(unit_num) for t in range(T_num)}
    p = {(i,j,t): m.addVar(lb=0, ub=bid_data[i,j], name=f'p_{i}_{j}_{t}')
         for i in range(unit_num) for j in range(5) for t in range(T_num)}

    # 新增  能相关变量
    u_sto = {(i,t): m.addVar(vtype=cp.COPT.BINARY, name=f'u_sto_{i}_{t}') 
             for i in range(sto_num) for t in range(T_num)}  # 放电状态
    v_sto = {(i,t): m.addVar(vtype=cp.COPT.BINARY, name=f'v_sto_{i}_{t}') 
             for i in range(sto_num) for t in range(T_num)}  # 充电状态
    w_sto = {(i,t): m.addVar(vtype=cp.COPT.BINARY, name=f'w_sto_{i}_{t}') 
             for i in range(sto_num) for t in range(T_num)}  # 停机状态
    
    P_sto = {(i,t): m.addVar(
        lb=0, 
        ub=min(Pgmax[i], Cm[i]), # 考虑容量限制
        name=f'P_sto_{i}_{t}'
    ) for i in range(sto_num) for t in range(T_num)}
    
    P_charge = {(i,t): m.addVar(
        lb=0,
        ub=min(Pf[i], Cm[i] - Ci[i]), # 考虑剩余可充电容量
        name=f'P_charge_{i}_{t}'
    ) for i in range(sto_num) for t in range(T_num)}
    
    e_sto = {(i,t): m.addVar(
        lb=max(0, Cf[i] - (T_num-t-1)*(Pf[i]*Ec[i])), # 考虑终止容量约束
        ub=min(Cm[i], Ci[i] + t*Pf[i]*Ec[i]), # 考虑最大充电量
        name=f'e_sto_{i}_{t}'
    ) for i in range(sto_num) for t in range(T_num)}
    
    p_sto = {(i,j,t): m.addVar(lb=0, ub=sto_bid_data[i,j], name=f'p_sto_{i}_{j}_{t}')
             for i in range(sto_num) for j in range(4) for t in range(T_num)}  # 储能分段发电

    # 优化目标函数构建
    obj = cp.quicksum(u[i,t] * Pmin[i] * price[i,0] + 
             cp.quicksum(price[i,j+1] * p[i,j,t] for j in range(5) if j < len(price[i])-1) + 
             v[i,t] * Co[i]
             for i in range(unit_num) for t in range(T_num))

    # 储能成本分开计算
    for i in range(sto_num):
        for t in range(T_num):
            # 发电成本
            obj += u_sto[i,t] * Pgmin[i] * sto_price[i,1]
            for j in range(4):
                obj += sto_price[i,j+2] * p_sto[i,j,t]
            
            # 充电成本：创建一个辅助变量来线性化 v_sto[i,t] * P_charge[i,t]
            # 因为P_charge已经在约束中定义为 P_charge[i,t] == v_sto[i,t] * data['Pf'][i]
            obj += sto_price[i,0] * P_charge[i,t]

    m.setObjective(obj, sense=cp.COPT.MINIMIZE)

    # 添加启停状态逻辑约束
    for i in range(unit_num):
        # 初始时刻
        m.addConstr(u[i,0] - U0[i] == v[i,0] - w[i,0])
        # 其他时刻
        for t in range(1, T_num):
            m.addConstr(u[i,t] - u[i,t-1] == v[i,t] - w[i,t])

    # 系统功率平衡约束
    for t in range(T_num):
        m.addConstr(cp.quicksum(P[i,t] for i in range(unit_num)) + 
                   cp.quicksum(P_sto[i,t] for i in range(sto_num)) - 
                   cp.quicksum(P_charge[i,t] for i in range(sto_num)) == load[t])
        
    # 最小开机时间约束
    for i in range(unit_num):
        # 完整时段的约束
        for t in range(T_num + 1 - To[i]):
            sum_u = cp.quicksum(u[i,k] for k in range(t, t+To[i]))
            m.addConstr(sum_u >= To[i] * v[i,t])
        
        # 末尾不足时段的特殊处理
        for t in range(T_num - To[i] + 1, T_num):
            sum_u = cp.quicksum(u[i,k] for k in range(t, T_num))
            remaining_periods = T_num - t
            m.addConstr(sum_u >= remaining_periods * v[i,t])
        
        # 初始时刻约束
        if U0[i] == 1 and Tc[i] < To[i]:
            sum_u = cp.quicksum(u[i,t] for t in range(To[i]-Tc[i]))
            m.addConstr(sum_u == To[i] - Tc[i])

    # 最小停机时间约束
    for i in range(unit_num):
        # 完整时段的约束
        for t in range(T_num + 1 - Tf[i]):
            sum_off = cp.quicksum(1-u[i,k] for k in range(t, t+Tf[i]))
            m.addConstr(sum_off >= Tf[i] * w[i,t])
        
        # 末尾  足时段的特殊处理
        for t in range(T_num - Tf[i] + 1, T_num):
            sum_off = cp.quicksum(1-u[i,k] for k in range(t, T_num))
            remaining_periods = T_num - t
            m.addConstr(sum_off >= remaining_periods * w[i,t])
        
        # 初始时刻约束
        if U0[i] == 0 and Tc[i] < Tf[i]:
            sum_off = cp.quicksum(1-u[i,t] for t in range(Tf[i]-Tc[i]))
            m.addConstr(sum_off == Tf[i] - Tc[i])

    # 添加系统备用约束
    SR = 0.1  # 备用容量系数
    penalty = 1e4  # 备用约束违反惩罚系数
    
    # 创建备用容量松弛变量
    reserve_slack = {t: m.addVar(lb=0, name=f'reserve_slack_{t}') 
                    for t in range(T_num)}
    
    # 修改后的备用容量约束
    for t in range(T_num):
        total_output = cp.quicksum(P[i,t] for i in range(unit_num))
        max_available = cp.quicksum(Pmax[i] * u[i,t] for i in range(unit_num))
        reserve_capacity = max_available - total_output
        
        # 添加带松弛变量的备用约束
        m.addConstr(reserve_capacity + reserve_slack[t] >= SR * load[t], 
                   name=f'reserve_constraint_{t}')
        
        # 在目标函数中添加惩罚项
        obj += penalty * reserve_slack[t]

    # 更新目标函数
    m.setObjective(obj, sense=cp.COPT.MINIMIZE)

    # 机组出力上下限约束
    for i in range(unit_num):
        for t in range(T_num):
            m.addConstr(P[i,t] >= Pmin[i] * u[i,t])
            m.addConstr(P[i,t] <= Pmax[i] * u[i,t])

    # 优化分段出力约束实现
    for i in range(unit_num):
        for t in range(T_num):
            # 总出力等于分段出力之和加最小出力
            m.addConstr(P[i,t] == cp.quicksum(p[i,j,t] for j in range(5)) + Pmin[i] * u[i,t])
            # 分段出力上下限
            for j in range(5):
                m.addConstr(p[i,j,t] >= 0)
                m.addConstr(p[i,j,t] <= bid_data[i,j] * u[i,t])  # 与机组状态关联

    # 优化爬坡约束实现
    M = max(Pmax) + 100  # 增加一些余量确保M足够大
    for i in range(unit_num):
        for t in range(1, T_num):
            z = m.addVar(vtype=cp.COPT.BINARY, name=f'z_{i}_{t}')
            
            # 线性化约束
            m.addConstr(z <= u[i,t])
            m.addConstr(z <= u[i,t-1])
            m.addConstr(z >= u[i,t] + u[i,t-1] - 1)
            
            # 爬坡约束
            m.addConstr(P[i,t] - P[i,t-1] <= Ru[i] * z + M * (1 - z))
            m.addConstr(P[i,t-1] - P[i,t] <= Rd[i] * z + M * (1 - z))

    # 优化断面约束
    for t in range(T_num):
        for s in range(len(Plmax)):
            flow = (cp.quicksum(sens[s,i] * P[i,t] for i in range(unit_num)) + 
                   cp.quicksum(sens[s,-1] * (P_sto[i,t] - P_charge[i,t]) 
                             for i in range(sto_num)) - sum_sens[s,t])
            m.addConstr(flow <= Plmax[s])
            m.addConstr(flow >= -Plmax[s])

    # 新增储能��关约束
    # 1. 状态互斥约束
    for i in range(sto_num):
        for t in range(T_num):
            m.addConstr(u_sto[i,t] + v_sto[i,t] + w_sto[i,t] == 1)

    # 2. 储能容量变化约束
    for i in range(sto_num):
        m.addConstr(e_sto[i,0] == Ci[i])  # 初始容量
        for t in range(T_num-1):
            m.addConstr(e_sto[i,t+1] == e_sto[i,t] - P_sto[i,t] + P_charge[i,t])
        m.addConstr(e_sto[i,T_num-1] - P_sto[i,T_num-1] + P_charge[i,T_num-1] >= Cf[i])

    # 3. 充放电功率约束
    for i in range(sto_num):
        for t in range(T_num):
            m.addConstr(P_charge[i,t] == v_sto[i,t] * Pf[i])
            m.addConstr(P_sto[i,t] >= Pgmin[i] * u_sto[i,t])
            m.addConstr(P_sto[i,t] <= Pgmax[i] * u_sto[i,t])

    # 储能最小持续时间约束
    for i in range(sto_num):
        # 初始状态约束
        # 最小放电时间
        m.addConstr(cp.quicksum(u_sto[i,t] for t in range(min(Tg[i], T_num))) >= 
                    Tg[i] * u_sto[i,0])
        
        # 最小充电时间
        m.addConstr(cp.quicksum(v_sto[i,t] for t in range(min(Tp[i], T_num))) >= 
                    Tp[i] * v_sto[i,0])
        
        # 最小停机时间
        m.addConstr(cp.quicksum(w_sto[i,t] for t in range(min(Ts[i], T_num))) >= 
                    Ts[i] * w_sto[i,0])

        # ���间时段约束
        for t in range(1, T_num):
            # 最小放电时间约束
            if t + Tg[i] - 1 <= T_num:
                # 修改这里，确保range不会超出T_num
                end_t = min(t + Tg[i], T_num)
                m.addConstr(cp.quicksum(u_sto[i,k] for k in range(t, end_t)) >= 
                          Tg[i] * (u_sto[i,t] - u_sto[i,t-1]))
            
            # 最小充电时间约束
            if t + Tp[i] - 1 <= T_num:
                # 修改这里，确保range不会超出T_num
                end_t = min(t + Tp[i], T_num)
                m.addConstr(cp.quicksum(v_sto[i,k] for k in range(t, end_t)) >= 
                          Tp[i] * (v_sto[i,t] - v_sto[i,t-1]))
            
            # 最小停机时间约束
            if t + Ts[i] - 1 <= T_num:
                # 修改这里，确保range不会超出T_num
                end_t = min(t + Ts[i], T_num)
                m.addConstr(cp.quicksum(w_sto[i,k] for k in range(t, end_t)) >= 
                          Ts[i] * (w_sto[i,t] - w_sto[i,t-1]))

        # 末尾时段约束
        # 放电状态
        for t in range(max(1, T_num - Tg[i] + 2), T_num):
            m.addConstr(cp.quicksum(u_sto[i,k] for k in range(t, T_num)) >= 
                       (T_num - t) * (u_sto[i,t] - u_sto[i,t-1]))
        
        # 充电��态
        for t in range(max(1, T_num - Tp[i] + 2), T_num):
            m.addConstr(cp.quicksum(v_sto[i,k] for k in range(t, T_num)) >= 
                       (T_num - t) * (v_sto[i,t] - v_sto[i,t-1]))
        
        # 停机状态
        for t in range(max(1, T_num - Ts[i] + 2), T_num):
            m.addConstr(cp.quicksum(w_sto[i,k] for k in range(t, T_num)) >= 
                       (T_num - t) * (w_sto[i,t] - w_sto[i,t-1]))

    # 5. 新增：充放电状态转换约束
    for i in range(sto_num):
        for t in range(1, T_num):
            # 发电后的下一个时刻不能立即充电
            m.addConstr(u_sto[i,t-1] + v_sto[i,t] <= 1)
            # 充电后的下一个时刻不能立即发电
            m.addConstr(v_sto[i,t-1] + u_sto[i,t] <= 1)

    # 修改求解参数设置
    # 1. 基础参数设置
    m.setParam(cp.COPT.Param.FeasTol, 1e-6)        # 降低可行性容差
    m.setParam(cp.COPT.Param.IntTol, 1e-6)         # 降低整数容差
    m.setParam(cp.COPT.Param.DualTol, 1e-6)        # 降低对偶容差
    m.setParam(cp.COPT.Param.BarIterLimit, 1000)   # 增加内点法迭代次数
    
    # 2. 预处理和缩放设置
    m.setParam(cp.COPT.Param.Presolve, 2)          # 加强预求解
    m.setParam(cp.COPT.Param.Scaling, 1)           # 启用矩阵缩放
    m.setParam(cp.COPT.Param.Dualize, 1)           # 启用对偶化
    
    # 3. 启发式算法设置
    m.setParam(cp.COPT.Param.HeurLevel, 1)         # 加强启发式搜索
    m.setParam(cp.COPT.Param.RoundingHeurLevel, 1) # 加强舍入启发式
    m.setParam(cp.COPT.Param.DivingHeurLevel, 1)   # 加强深优先启发式
    m.setParam(cp.COPT.Param.SubMipHeurLevel, 1)   # 加强子MIP启发式
    
    # 4. 分支定界设置
    m.setParam(cp.COPT.Param.StrongBranching, 1)   # 启用强分支
    m.setParam(cp.COPT.Param.ConflictAnalysis, 1)  # 启用冲突分析
    
    # 5. 割平面设置
    m.setParam(cp.COPT.Param.CutLevel, 3)          # 提高整体割平面强度
    m.setParam(cp.COPT.Param.RootCutLevel, 3)      # 在根节点生成更强割平面
    m.setParam(cp.COPT.Param.TreeCutLevel, 2)      # 在搜索树中启用割平面
    m.setParam(cp.COPT.Param.RootCutRounds, 100)   # 增加根节点割平面轮数
    m.setParam(cp.COPT.Param.NodeCutRounds, 50)    # 增加树节点割平面轮数
    
    # 6. 并行计算设置
    m.setParam(cp.COPT.Param.Threads, 8)           # 使用8个线程
    m.setParam(cp.COPT.Param.BarThreads, 8)        # 内点法线程数
    m.setParam(cp.COPT.Param.SimplexThreads, 8)    # 单纯形法线程数
    m.setParam(cp.COPT.Param.CrossoverThreads, 8)  # 交叉线程数
    
    # 7. 求解方法设��
    m.setParam(cp.COPT.Param.LpMethod, 2)          # 使用内点法
    m.setParam(cp.COPT.Param.BarHomogeneous, 1)    # 使用齐次自对偶内点法
    m.setParam(cp.COPT.Param.Crossover, 1)         # 启用交叉算法

    # 系统功率平衡约束添加松弛变量
    slack_plus = {t: m.addVar(lb=0, name=f'slack_plus_{t}') 
                 for t in range(T_num)}
    slack_minus = {t: m.addVar(lb=0, name=f'slack_minus_{t}') 
                  for t in range(T_num)}
    
    # 修改功率平衡约束
    for t in range(T_num):
        m.addConstr(cp.quicksum(P[i,t] for i in range(unit_num)) + 
                   cp.quicksum(P_sto[i,t] for i in range(sto_num)) - 
                   cp.quicksum(P_charge[i,t] for i in range(sto_num)) + 
                   slack_plus[t] - slack_minus[t] == load[t])
    
    # 在目标函数中添加惩罚项
    M = 1e6  # 大数惩罚系数
    obj += M * cp.quicksum(slack_plus[t] + slack_minus[t] for t in range(T_num))

    return m, env

def solve_scuc():
    """求解SCUC问题"""
    # 修改数据文件夹路径
    folder_path = 'Competition_model_2024\Question1-1\model'
    solution_path = 'Competition_model_2024\Question1-1\solution'
    
    # 确保结果文件夹存在
    os.makedirs(solution_path, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # 读取数据
        data_tuple = read_scuc_data(folder_path)
        data = convert_tuple_to_dict(data_tuple)
        
        # 构建并求解模型
        model, env = build_scuc_model(data)
        
        # 设置求解参数
        model.setParam(cp.COPT.Param.TimeLimit, 3600)  # 时间限制1小时
        model.setParam(cp.COPT.Param.RelGap, 0.003)   # 相对间隙0.3%
        
        # 直接求解模型
        model.solve()
        
        # 保存结果
        solution_file = os.path.join(solution_path, 'solution.sol')
        save_solution(model, data, solution_file)
        
        # 保存求解日志
        log_file = os.path.join(solution_path, 'solve.log')
        save_log(model, data, log_file, time.time() - start_time)
        
        print(f"\n结果已保存到: {solution_path}")
            
    except Exception as e:
        print(f"求解过程出错: {str(e)}")
        raise
    
    finally:
        if 'env' in locals():
            env.close()

def save_solution(model, data, filename):
    """保存solution.sol文件，并根据决策变量重新计算目标函数值"""
    # 获取所有决策变量值
    T_num = data['T_num']
    unit_num = data['unit_num']
    sto_num = data['sto_num']
    
    # 预先获取所有需要的变量值
    u_vals = np.zeros((unit_num, T_num))
    v_vals = np.zeros((unit_num, T_num))
    p_vals = np.zeros((unit_num, T_num))
    u_sto_vals = np.zeros((sto_num, T_num))
    ps_sto_vals = np.zeros((sto_num, T_num))
    
    # 获取机组变量值
    for i in range(unit_num):
        for t in range(T_num):
            u_vals[i,t] = round(model.getVarByName(f'u_{i}_{t}').x)
            v_vals[i,t] = round(model.getVarByName(f'v_{i}_{t}').x)
            p_vals[i,t] = model.getVarByName(f'P_{i}_{t}').x
    
    # 获取储能变量值
    for i in range(sto_num):
        for t in range(T_num):
            u_sto_vals[i,t] = round(model.getVarByName(f'u_sto_{i}_{t}').x)
            if u_sto_vals[i,t] > 0.5:  # 发电状态
                ps_sto_vals[i,t] = model.getVarByName(f'P_sto_{i}_{t}').x
            else:  # 充电状态
                ps_sto_vals[i,t] = -model.getVarByName(f'P_charge_{i}_{t}').x

    # 获取分段出力值
    p_seg_vals = np.zeros((unit_num, 5, T_num))  # 常规机组分段出力
    p_sto_seg_vals = np.zeros((sto_num, 4, T_num))  # 储能分段出力
    
    for i in range(unit_num):
        for j in range(5):
            for t in range(T_num):
                p_seg_vals[i,j,t] = model.getVarByName(f'p_{i}_{j}_{t}').x
    
    for i in range(sto_num):
        for j in range(4):
            for t in range(T_num):
                p_sto_seg_vals[i,j,t] = model.getVarByName(f'p_sto_{i}_{j}_{t}').x

    # 计算目标函数（成本）
    total_cost = 0
    
    # 常规机组成本
    for t in range(T_num):
        for i in range(unit_num):
            # 启动成本
            total_cost += v_vals[i,t] * data['Co'][i]
            # 最小出力成本
            total_cost += data['price'][i,0] * u_vals[i,t] * data['Pmin'][i]
            # 分段出力成本
            for j in range(5):
                if j < len(data['price'][i])-1:
                    total_cost += data['price'][i,j+1] * p_seg_vals[i,j,t]

    # 储能成本
    for t in range(T_num):
        for i in range(sto_num):
            if ps_sto_vals[i,t] < 0:  # 充电状态
                total_cost += data['sto_price'][i,0] * ps_sto_vals[i,t]
            else:  # 发电状态
                if u_sto_vals[i,t] > 0.5:
                    # 最小发电成本
                    total_cost += data['sto_price'][i,1] * u_sto_vals[i,t] * data['Pgmin'][i]
                    # 分段发电成本
                    for j in range(4):
                        total_cost += data['sto_price'][i,j+2] * p_sto_seg_vals[i,j,t]

    # 写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Objective value = {total_cost:.8e}\n")
        
        # 写入储能结果
        for i in range(sto_num):
            for t in range(T_num):
                if ps_sto_vals[i,t] > 1e-6:  # 发电
                    f.write(f"storage{i+1}_s_{t} 1\n")
                    f.write(f"storage{i+1}_p_{t} {ps_sto_vals[i,t]:.8e}\n")
                elif ps_sto_vals[i,t] < -1e-6:  # 充电
                    f.write(f"storage{i+1}_s_{t} -1\n")
                    f.write(f"storage{i+1}_p_{t} {ps_sto_vals[i,t]:.8e}\n")
                else:  # 停机
                    f.write(f"storage{i+1}_s_{t} 0\n")
                    f.write(f"storage{i+1}_p_{t} 0.00000000e+00\n")
        
        # 写入机组结果
        for i in range(unit_num):
            for t in range(T_num):
                f.write(f"unit{i+1}_s_{t} {int(u_vals[i,t])}\n")
                if abs(p_vals[i,t]) > 1e-6:
                    f.write(f"unit{i+1}_p_{t} {p_vals[i,t]:.8e}\n")
                else:
                    f.write(f"unit{i+1}_p_{t} 0.00000000e+00\n")

def save_log(model, data, filename, solve_time):
    """保存求解日志"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"求解时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
        f.write(f"求解状态: {model.status}\n")
        f.write(f"计算时间: {solve_time:.2f}秒\n")
        f.write(f"目标函数值: {model.objVal:.6f}\n")
        f.write(f"节点数: {model.getAttr('NodeCnt')}\n")

def convert_tuple_to_dict(data_tuple):
    """将数据元组转换为字典"""
    return {
        'load': data_tuple[0],
        'T_num': data_tuple[1],
        'unit_num': data_tuple[2],
        'Pmax': data_tuple[3],
        'Pmin': data_tuple[4],
        'Ru': data_tuple[5],
        'Rd': data_tuple[6],
        'To': data_tuple[7],
        'Tf': data_tuple[8],
        'Co': data_tuple[9],
        'U0': data_tuple[10],
        'P0': data_tuple[11],
        'Tc': data_tuple[12],
        'bid_data': data_tuple[13],
        'price': data_tuple[14],
        'Plmax': data_tuple[15],
        'sens': data_tuple[16],
        'sum_sens': data_tuple[17],
        'sto_num': data_tuple[18],
        'Tp': data_tuple[19],
        'Ts': data_tuple[20],
        'Tg': data_tuple[21],
        'Ci': data_tuple[22],
        'Cf': data_tuple[23],
        'Cm': data_tuple[24],
        'Ec': data_tuple[25],
        'Pf': data_tuple[26],
        'Pgmax': data_tuple[27],
        'Pgmin': data_tuple[28],
        'sto_bid_data': data_tuple[29],
        'sto_price': data_tuple[30],
        'point': data_tuple[31],
        'sto_point': data_tuple[32]
    }

if __name__ == "__main__":
    solve_scuc() 