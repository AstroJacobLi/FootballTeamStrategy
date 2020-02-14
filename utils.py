import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def match_i_Huskies_passing_table(filename, match_i):
    '''
    Match i-th Huskies players passing table
    
    Return: {playername: [origin, destination]}
    '''
    passing = pd.read_csv(filename)
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