import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import utils

## We define each S* motif as a directed graph in networkx
## up=1, down=3, right=2
motifs_3 = {
    'S1': nx.DiGraph([(2,1),(2,3)]),
    'S2': nx.DiGraph([(3,2),(2,1)]),
    'S3': nx.DiGraph([(3,2),(2,3),(2,1)]),
    'S4': nx.DiGraph([(3,1),(2,1)]),
    'S5': nx.DiGraph([(3,1),(2,1),(2,3)]),
    'S6': nx.DiGraph([(3,1),(2,1),(2,3),(3,2)]),
    'S7': nx.DiGraph([(3,2),(2,3),(1,2)]),
    'S8': nx.DiGraph([(3,2),(2,3),(2,1),(1,2)]),
    'S9': nx.DiGraph([(1,2),(2,3),(3,1)]),
    'S10': nx.DiGraph([(1,2),(2,3),(3,1),(2,1)]),
    'S11': nx.DiGraph([(1,2),(2,1),(3,2),(3,1)]),
    'S12': nx.DiGraph([(1,2),(2,1),(3,2),(2,3),(3,1)]),
    'S13': nx.DiGraph([(1,2),(2,1),(2,3),(3,2),(3,1),(1,3)])
    }

motifs_2 = {
    'S1': nx.DiGraph([(1,2)]),
    'S2': nx.DiGraph([(1,2),(2,1)])
    }


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
    '''
    for edge in G.edges:
        if geometrical_dist[edge] == 0:
            dist = 0
        else:
            dist = 100 / geometrical_dist[edge]
            dist /= 10
        passes = unidirection_pass[edge] / 5
        weight = passes + dist
        weight_dict[edge] = weight
    '''

    for edge in G.edges:
        if geometrical_dist[edge] == 0:
            dist = np.inf
        else:
            dist = geometrical_dist[edge]
        passes = unidirection_pass[edge]
        weight = passes / np.sqrt(dist)
        weight_dict[edge] = weight
    
    df['weight'] = np.array([weight_dict[(row['OriginPlayerID'], row['DestinationPlayerID'])] for ind, row in df.iterrows()])

    # Rebuild the graph, with weights
    G = nx.from_pandas_edgelist(df, source='OriginPlayerID', 
                                target='DestinationPlayerID', 
                                edge_attr=['weight'], 
                                create_using=nx.DiGraph())
    G.remove_edges_from(nx.selfloop_edges(G))

    return G, pos, centrality_dict, geometrical_dist, unidirection_pass, weight_dict


def plot_network(df, team, matchid, ax=None, savefig=False, **kwds):
    
    G, pos, centrality_dict, geometrical_dist, unidirection_pass, weight_dict = build_network(df, team, matchid)

    eigen_centrality = centrality_dict['eigen_centrality']
    between_centrality = centrality_dict['between_centrality']
    
    # Plot the network
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7.5))
    else:
        ax = ax
        
    node_size = np.exp(eigen_centrality / np.percentile(eigen_centrality, 15))
    node_size = node_size / np.nanmean(node_size) * 1000 # 1500

    node_color = np.log10(between_centrality)
    
    edge_width = np.array(list(weight_dict.values())) / 5 # *2
    
    from matplotlib.colors import ListedColormap
    # Choose colormap
    my_cmap = plt.cm.RdBu_r
    '''
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    '''
    nx.draw_networkx(G, pos, ax=ax, 
                     node_size=node_size, 
                     node_color=node_color, 
                     edgelist=list(unidirection_pass.keys()), 
                     width=edge_width,
                     edge_color='lightgray', cmap=my_cmap, 
                     #font_color='lawngreen', 
                     vmin=node_color[~np.isinf(node_color)][-1], 
                     vmax=max(node_color), arrowsize=8,  # 15
                     arrowstyle='->', **kwds)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=node_color[~np.isinf(node_color)][-1], 
                                                                 vmax=max(node_color)))
    cbar = plt.colorbar(sm, ax=ax, extend='both')
    cbar.set_label(r'log (Betweenness Centrality)')
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([20, 40, 60, 80, 100])

    #plt.xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.tick_params(direction='in', left=True, bottom=True, labelleft=True, labelbottom=True)

    if savefig:
        plt.savefig('./{0}_match{1}.png'.format(team, matchid), bbox_inches='tight')
    if ax is None:
        plt.show()
        return
    else:
        return ax


def calc_network_params(graph):
    '''
    Calculate many indicators and structual parameters of the network.
    
    Graph must have weight!
    '''
    clustering_coeff = nx.algorithms.average_clustering(graph, weight='weight')
    # Shortest path length is undefined if the graph is disconnected.
    if not nx.is_weakly_connected(graph):
        shortest_path = np.inf
    else:
        shortest_path = nx.algorithms.average_shortest_path_length(graph, weight='weight')
    A = nx.adjacency_matrix(graph, weight='weight')
    e = np.linalg.eigvals(A.todense())
    largest_eigenvalue = np.real(max(e))
    # The paper-1 doesn't use normalized algebraic connectivity
    algebraic_conn = nx.algebraic_connectivity(graph.to_undirected(), weight='weight') 
    eg_cen = list(nx.algorithms.eigenvector_centrality(graph, weight='weight').values())
    eigen_centrality_dict = {'max': np.max(eg_cen), 'std': np.std(eg_cen, ddof=1)}
    
    network_params = {}
    network_params['clustering_coeff'] = clustering_coeff
    network_params['shortest_path'] = shortest_path
    network_params['largest_eigenvalue'] = largest_eigenvalue
    network_params['algebraic_conn'] = algebraic_conn
    network_params['eigen_cen_max'] = eigen_centrality_dict['max']
    network_params['eigen_cen_std'] = eigen_centrality_dict['std']
    
    return network_params

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

    num_of_nodes = len(list(mo.values())[0].nodes)
    #We use iterools.product to have all combinations of three nodes in the
    #original graph. Then we filter combinations with non-unique nodes, because
    #the motifs do not account for self-consumption.

    triplets = list(itertools.product(*[nodes, nodes, nodes])) # all permutaions of 3 nodes
    triplets = [trip for trip in triplets if len(list(set(trip))) == num_of_nodes] # remove permutations with duplicate nodes
    triplets = map(list, map(np.sort, triplets)) 
    u_triplets = []
    [u_triplets.append(trip) for trip in triplets if not u_triplets.count(trip)]

    #The for each each of the triplets, we (i) take its subgraph, and compare
    #it to all fo the possible motifs

    for trip in u_triplets:
        sub_gr = gr.subgraph(trip).to_directed()
        sub_gr.remove_edges_from(nx.selfloop_edges(sub_gr)) # remove self-looops
        weight_sum = np.sum([weight_dict[edge] for edge in sub_gr.edges])
        mot_match = list(map(lambda mot_id: nx.is_isomorphic(sub_gr, mo[mot_id]), mo.keys()))
        match_keys = [list(mo.keys())[i] for i in range(len(mo)) if mot_match[i]]
        if len(match_keys) == 1:
            mcount[match_keys[0]] += 1
            mweight[match_keys[0]] += weight_sum

    return mcount, mweight

