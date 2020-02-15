import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def mcounter(gr, mo, weight_dict):
    """
    Counts motifs in a directed graph
    From: https://gist.github.com/tpoisot/8582648
    
    :param gr: A ``DiGraph`` object
    :param mo: A ``dict`` of motifs to count
    :param weight_dict: ``dict`` of weight of each edge in graph ``gr``
    
    :returns: A ``dict`` with the number of each motifs, with the same keys as ``mo``
    This function is actually rather simple. It will extract all 3-grams from
    the original graph, and look for isomorphisms in the motifs contained
    in a dictionary. The returned object is a ``dict`` with the number of
    times each motif was found.::
        >>> print mcounter(gr, mo)
        {'S1': 4, 'S3': 0, 'S2': 1, 'S5': 0, 'S4': 3}
    """
    import itertools
    #This function will take each possible subgraphs of gr of size 3, then
    #compare them to the mo dict using .subgraph() and is_isomorphic
    
    #This line simply creates a dictionary with 0 for all values, and the
    #motif names as keys

    mcount = dict(zip(mo.keys(), list(map(int, np.zeros(len(mo))))))
    mweight = dict(zip(mo.keys(), list(map(int, np.zeros(len(mo))))))
    nodes = gr.nodes()

    #We use iterools.product to have all combinations of three nodes in the
    #original graph. Then we filter combinations with non-unique nodes, because
    #the motifs do not account for self-consumption.

    triplets = list(itertools.product(*[nodes, nodes, nodes])) # all permutaions of 3 nodes
    triplets = [trip for trip in triplets if len(list(set(trip))) == 3] # remove permutations with duplicate nodes
    triplets = map(list, map(np.sort, triplets)) 
    u_triplets = []
    [u_triplets.append(trip) for trip in triplets if not u_triplets.count(trip)]

    #The for each each of the triplets, we (i) take its subgraph, and compare
    #it to all fo the possible motifs

    for trip in u_triplets:
        sub_gr = gr.subgraph(trip)
        weight_sum = np.sum([weight_dict[edge] for edge in sub_gr.edges])
        mot_match = list(map(lambda mot_id: nx.is_isomorphic(sub_gr, mo[mot_id]), mo.keys()))
        match_keys = [list(mo.keys())[i] for i in range(len(mo)) if mot_match[i]]
        if len(match_keys) == 1:
            mcount[match_keys[0]] += 1
            mweight[match_keys[0]] += weight_sum

    return mcount, mweight


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

def match_i_passing_table(filename, team_id, match_i):
    '''
    Match i-th {TeamID} players passing table
    
    Return: {playername: [origin, destination]}
    '''
    passing = pd.read_csv(filename)
    player_dic = {}
    for i in range(len(passing)):
        if passing['MatchID'][i] == match_i:
            if passing['TeamID'][i] == team_id:
                if passing['OriginPlayerID'][i] not in player_dic:
                    player_dic[passing['OriginPlayerID'][i]] = [1, 0]
                else:
                    player_dic[passing['OriginPlayerID'][i]][0] += 1
                if passing['DestinationPlayerID'][i] not in player_dic:
                    player_dic[passing['DestinationPlayerID'][i]] = [0, 1]
                else:
                    player_dic[passing['DestinationPlayerID'][i]][1] += 1
    return player_dic

def match_i_Opponent_passing_table(filename, match_i):
    '''
    Match i-th Opponent players passing table
    
    Return: {playername: [origin, destination]}
    '''
    passing = pd.read_csv(filename)
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


def Huskies_passing_table(filename):
    passing = pd.read_csv(filename)
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