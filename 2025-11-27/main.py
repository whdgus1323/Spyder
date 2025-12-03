import time
import numpy as np
import matplotlib.pyplot as plt

def build_config():
    C = {}
    
    C['OCWmin'] = 7
    C['OCWmax'] = 8
    C['Mode_UORA'] = 0



    C['N_unassoc_In'] = 5
    C['T_unassoc_In'] = 0.0
    
    C['N_unassoc_Out'] = 2
    C['T_unassoc_Out'] = 4.0
    
    C['N_STAs_Assoc'] = 50
    
    
    
    
    C['N_RU'] = 9
    C['N_RA_RU_0'] = 8
    C['N_RA_RU_2045'] = C['N_RU'] - C['N_RA_RU_0']
    
    C['N_slot_total'] = 3
    
    return C

def init_sim(C) :
    S = {}
    
    S['i'] = 1
    
    S['N_STAs_existing'] = int(C['N_STAs_Assoc'])
    S['N_STA'] = S['N_STAs_existing'] + int(C['N_unassoc_In'])
    
    
    S['U_slot'] = 9e-6
    S['SIFS'] = 16e-6
    
    S['L_MPDU'] = np.full(S['N_STA'], 2000.0, dtype=float)
    
    S['Next_T_unassoc_In'] = 0
    
    S['CWmin'] = C['OCWmin']
    S['CWmax'] = C['OCWmax']
    
    S['T_unassoc_In_slots'] = int(np.ceil(C['T_unassoc_In'] / S['U_slot']))
    S['T_unassoc_Out_slots'] = int(np.ceil(C['T_unassoc_Out'] / S['U_slot']))
    S['T_Period_InOut'] = S['T_unassoc_In_slots'] + S['T_unassoc_Out_slots']
    
    S['STA_Type'] = np.zeros(S['N_STA'], dtype=int)
    S['STA_Type'][:S['N_STAs_existing']] = 1
    S['STA_Type'][S['N_STAs_existing']:] = 0
    
    S['STA_Type'][(S['N_STAs_existing'] + int(C['N_unassoc_In'])) - 2 :] = -1
    
    return S

def step_in_out(C, S) :
    print('A', end=', ')
    if S['Next_T_unassoc_In'] <= S['i'] and C['N_unassoc_In'] > 0:
        STA_ID_Disable = np.where(S['STA_Type'] == -1)[0]
        print(STA_ID_Disable, ' : ', C['N_unassoc_In'])
        
        if len(STA_ID_Disable) == C['N_unassoc_In'] - 3:
            S['STA_Type'][STA_ID_Disable] = 0
            S['Next_T_unassoc_In'] = S['i'] + S['T_Period_InOut']
            print(S['Next_T_unassoc_In'], ' / ', S['i'], ' / ', S['T_Period_InOut'])
    

def step_obo_decrement(C, S) :
    print('B', end=', ')
    
def step_tf_and_ru_assign(C, S) :
    print('C', end=', ')
    
def step_ru_outcomes(C, S) :
    print('D', end=', ')
    
def step_time_and_assoc_success(C, S) :
    print('E', end=', ')
    
def step_observation_and_tail(C, S):
    S['i'] += 1
    print('F')


def run_one_slot(C, S) :
    step_in_out(C, S)
    step_obo_decrement(C, S)
    step_tf_and_ru_assign(C, S)
    step_ru_outcomes(C, S)
    step_time_and_assoc_success(C, S)
    step_observation_and_tail(C, S)
    
def compute_metrics_and_plot(C, S):
    print('compute_metrics_and_plot')
    print(S['STA_Type'])

def run_once_split(C):
    S = init_sim(C)
    
    while S['i'] <= C['N_slot_total'] :
        run_one_slot(C, S)
    out = compute_metrics_and_plot(C, S)
        
    return out


if __name__ == '__main__' :
    plt.figure(figsize=(9, 5))
    
    runs = [{'label' : 'UORA_STD(7, 31)', 'marker' : '.', 'Mode_UORA' : 0, 'OCWmin' : 7, 'OCWmax' : 31}, 
            {'label' : 'UORA_STD(15, 255)', 'marker' : '.', 'Mode_UORA' : 0, 'OCWmin' : 15, 'OCWmax' : 255}, 
            {'label' : 'UORA_STD(31, 1023)', 'marker' : '.', 'Mode_UORA' : 0, 'OCWmin' : 31, 'OCWmax' : 1023}]
    
    for cfg in runs:
        C = build_config()
        
        C['OCWmin'] = cfg['OCWmin']
        C['OCWmax'] = cfg['OCWmax']
        C['Mode_UORA'] = cfg['Mode_UORA']
        
        out = run_once_split(C)
        
        
        