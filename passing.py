def A_B_count(playerA, playerB):
    count = 0
    df = pd.read_csv('passingevents.csv')
    player_dic = {}
    for i in range(len(passing)):
        if passing['TeamID'][i] == 'Huskies':
            if passing['OriginPlayerID'][i] == playerA and passing['DestinationPlayerID'][i] == playerB:
                count += 1
    return count
print(A_B_count('Huskies_M1', 'Huskies_F2'))
print(A_B_count('Huskies_F2', 'Huskies_M1'))

import pandas as pd
def Huskies_passing_table():
    df = pd.read_csv('passingevents.csv')
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
# example
Huskies_passing_table_ = Huskies_passing_table()
print(Huskies_passing_table_)

def match_i_Opponent_passing_table(match_i):
    df = pd.read_csv('passingevents.csv')
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
# example
match_1_Opponent_passing_table = match_i_Opponent_passing_table(1)
print(match_1_Opponent_passing_table)

#match_i is a number from 1 to 38
import pandas as pd
def match_i_Huskies_passing_table(match_i):
    df = pd.read_csv('passingevents.csv')
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
# example
match_1_Huskies_passing_table = match_i_Huskies_passing_table(1)
print(match_1_Huskies_passing_table)


