import pandas as pd
import numpy as np
def A_B_count(playerA, playerB):
    count = 0
    df = pd.read_csv('./2020_Problem_D_DATA/passingevents.csv')
    player_dic = {}
    for i in range(len(passing)):
        if passing['TeamID'][i] == 'Huskies':
            if passing['OriginPlayerID'][i] == playerA and passing['DestinationPlayerID'][i] == playerB:
                count += 1
    return count

def Huskies_passing_table():
    df = pd.read_csv('./2020_Problem_D_DATA/passingevents.csv')
    player_dic = {}
    for i in range(len(passing)):
        if passing['TeamID'][i] == 'Huskies':
            if passing['OriginPlayerID'][i] not in player_dic:
                player_dic[passing['OriginPlayerID'][i]] = [1, 0]
            else:
                player_dic[passing['OriginPlayerID'][i]][0] += 1
            if passing['DestinationPlayerID'][i] not in player_dic:
                player_dic[passing['DestinationPlayerID'][i]] = [0, 1]
            else:
                player_dic[passing['DestinationPlayerID'][i]][1] += 1
    return player_dic

def match_i_Opponent_passing_table(match_i):
    df = pd.read_csv('./2020_Problem_D_DATA/passingevents.csv')
    player_dic = {}
    for i in range(len(passing)):
        if passing['MatchID'][i] == match_i:
            if passing['TeamID'][i] != 'Huskies':
                if passing['OriginPlayerID'][i] not in player_dic:
                    player_dic[passing['OriginPlayerID'][i]] = [1, 0]
                else:
                    player_dic[passing['OriginPlayerID'][i]][0] += 1
                if passing['DestinationPlayerID'][i] not in player_dic:
                    player_dic[passing['DestinationPlayerID'][i]] = [0, 1]
                else:
                    player_dic[passing['DestinationPlayerID'][i]][1] += 1
    return player_dic

def match_i_Huskies_passing_table(match_i):
    df = pd.read_csv('./2020_Problem_D_DATA/passingevents.csv')
    player_dic = {}
    for i in range(len(passing)):
        if passing['MatchID'][i] == match_i:
            if passing['TeamID'][i] == 'Huskies':
                if passing['OriginPlayerID'][i] not in player_dic:
                    player_dic[passing['OriginPlayerID'][i]] = [1, 0]
                else:
                    player_dic[passing['OriginPlayerID'][i]][0] += 1
                if passing['DestinationPlayerID'][i] not in player_dic:
                    player_dic[passing['DestinationPlayerID'][i]] = [0, 1]
                else:
                    player_dic[passing['DestinationPlayerID'][i]][1] += 1
    return player_dic

def sub_pitch_passing_table(match_i):
    df = pd.read_csv('./2020_Problem_D_DATA/passingevents.csv')
    df = df[df['MatchID'] == match_i]
    table = {(x,y):[0,0]for x in range(100) for y in range(100)}
    for i in range(len(df)):
        sub_pitch_x_o, sub_pitch_y_o = cal_sub_pitch_pos(df['EventOrigin_x'][i], df['EventOrigin_y'][i])
        sub_pitch_x_d, sub_pitch_y_d = cal_sub_pitch_pos(df['EventDestination_x'][i], df['EventDestination_y'][i])
        table[(sub_pitch_x_o, sub_pitch_y_o)][0] += 1
        table[(sub_pitch_x_d, sub_pitch_y_d)][1] += 1    
    return table
def cal_sub_pitch_pos(x, y):
    return x//11, int(y //6.5)
