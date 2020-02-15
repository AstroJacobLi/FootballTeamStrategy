import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import utils

def build_network(df, team, matchid):
    # import data
    if matchid != 'all':
        df = df[df['MatchID'] == matchid]
    df = df[df['TeamID'] == team]
    
    # calculate average position of each player
    uniq_player = np.unique(
        np.union1d(df['OriginPlayerID'], df['DestinationPlayerID']))
    x_mean = [
        np.mean(
            list(df[df['OriginPlayerID'] == name]['EventOrigin_x']) +
            list(df[df['DestinationPlayerID'] == name]['EventDestination_x']))
        for name in uniq_player
    ]
    y_mean = [
        np.mean(
            list(df[df['OriginPlayerID'] == name]['EventOrigin_y']) +
            list(df[df['DestinationPlayerID'] == name]['EventDestination_y']))
        for name in uniq_player
    ]

    pos = {
        name: np.array([x_mean[i], y_mean[i]])
        for i, name in enumerate(uniq_player)
    }

    # Build Directional graph
    G = nx.from_pandas_edgelist(df, source='OriginPlayerID', 
                                target='DestinationPlayerID', 
                                edge_attr=True, 
                                create_using=nx.DiGraph())

    # Calculate degree centrality
    deg_centrality = nx.algorithms.centrality.degree_centrality(G)
    deg_centrality = np.array([deg_centrality[node] for node in list(G.nodes())])

    # Calculate betweenness centrality
    between_centrality = nx.algorithms.centrality.betweenness_centrality(G)
    between_centrality = np.array([between_centrality[node] for node in list(G.nodes())])

    # Calculate eigenvector centrality
    eigen_centrality = nx.algorithms.centrality.eigenvector_centrality(G)
    eigen_centrality = np.array([eigen_centrality[node] for node in list(G.nodes())])
    
    centrality_dict = {'deg_centrality': deg_centrality, 
                       'between_centrality': between_centrality, 
                       'eigen_centrality': eigen_centrality}
    
    # Calculate total (in + out) passes through a node
    team_passes = utils.match_i_passing_table('./2020_Problem_D_DATA/passingevents.csv', 
                                              team_id=team, match_i=matchid)
    node_passes = np.array([np.sum(team_passes[node]) for node in list(G.nodes())])

    # Calculate geometrical distance between nodes (as a weight)
    geometrical_dist = {}
    from scipy.spatial import distance
    for edge in G.edges:
        geometrical_dist[edge] = distance.euclidean(pos[edge[0]], pos[edge[1]])


    # Calculate passes from A to B (as a weight)
    H = nx.from_pandas_edgelist(df[['OriginPlayerID', 'DestinationPlayerID']], 
                                source='OriginPlayerID', 
                                target='DestinationPlayerID', edge_attr=True, 
                                create_using=nx.MultiDiGraph()) ### Multi-Directional graph
    unidirection_pass = {}
    for edge in H.edges:
        unidirection_pass[(edge[0], edge[1])] = H.number_of_edges(edge[0], edge[1])

    # Weight 
    weight_dict = {}
    for edge in G.edges:
        if geometrical_dist[edge] == 0:
            dist = 0
        else:
            dist = 100 / geometrical_dist[edge]
            dist /= 10
        passes = unidirection_pass[edge] / 5
        weight = dist + passes
        weight_dict[edge] = weight
        
    return G, pos, centrality_dict, geometrical_dist, unidirection_pass, weight_dict


def plot_network(df, team, matchid, ax=None, savefig=False):
    
    G, pos, centrality_dict, geometrical_dist, unidirection_pass, weight_dict = build_network(df, 'Huskies', 1)

    eigen_centrality = centrality_dict['eigen_centrality']
    between_centrality = centrality_dict['between_centrality']
    
    # Plot the network
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        ax = ax
        
    node_size = np.exp(eigen_centrality / np.percentile(eigen_centrality, 15))
    node_size = node_size / np.nanmean(node_size) * 1500

    node_color = np.log10(between_centrality)
    
    edge_width = np.array(list(weight_dict.values())) * 2
    
    nx.draw_networkx(G, pos, ax=ax, 
                     node_size=node_size, 
                     node_color=node_color, 
                     edgelist=list(unidirection_pass.keys()), 
                     width=edge_width,
                     edge_color='gray', cmap='RdBu_r', 
                     vmin=node_color[~np.isinf(node_color)][-1], 
                     vmax=max(node_color), arrowsize=15, arrowstyle='->')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=node_color[~np.isinf(node_color)][-1], 
                                                                 vmax=max(node_color)))
    cbar = plt.colorbar(sm, ax=ax, extend='both')
    cbar.set_label('log( Betweenness Centrality )')

    plt.tick_params(direction='in')
    if savefig:
        plt.savefig('./{0}_match{1}.png'.format(team, matchid), bbox_inches='tight')
    if ax is None:
        plt.show()
        return
    else:
        return ax