import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import re

from data.visualise import final_pos_to_csv

GRAPH_FOLDER = "../data/graphs" 
POS_FOLDER = "../data/positions" 

def graph_to_csv(G, id, type = "random", folder = GRAPH_FOLDER):
    """Function for writing graph data to a .csv file

    The files first line includes the id of the graph, and the number of nodes,
    the following lines contain the edge list in the form of node pairs
    """
    node1 = [id]
    node2 = [G.number_of_nodes()]
    for u, v in G.edges:
        node1.append(u)
        node2.append(v)
    df = pd.DataFrame({'node1':node1, 'node2': node2})
    df.to_csv(f"{folder}/{type}_{id}.csv", header=False, index=False)

def graph_from_csv(file, folder = GRAPH_FOLDER, get_pos = True):
    """
    Function for reading graph data from a .csv file
    :param file: relative path to file from base folder
    """
    # get data from file name
    file = file.split('.')[0]
    type, id = file.split('_')
    file = f"{folder}/{file}.csv"
    # read the data
    df = pd.read_csv(file, header=None)
    num_nodes = df.iloc[0,1]
    id = df.iloc[0,0]
    # construct graph
    g = nx.Graph()
    g.add_nodes_from(range(0,num_nodes))    
    g.add_edges_from(list(df.iloc[1:].itertuples(index=False,name=None)))
    attrs_g = {'id': id, 'type': type}
    g.graph.update(attrs_g)
    # get pos data
    if get_pos:
        pos_df = pd.read_csv("../data/positions/final_pos.csv",sep=';')
        algorithm = pos_df[pos_df['id']==id]['algorithm']
        pos_dict = pos_df[pos_df['id']==id]['pos_dict']
        if len(algorithm) > 0:
            g.graph['algorithm'] = algorithm.iloc[0]
        if len(pos_dict) > 0:
            pos_dict = eval(pos_dict.values[0])
            for node in pos_dict.keys():
                g.nodes[node]['pos'] = pos_dict[node]
    return g

def get_last_id(folder = GRAPH_FOLDER):
    """Finds the largest graph id in a folder"""
    files = os.listdir(folder)
    max_id = -1
    for file in files:
        id = int(re.search(r'\d+', file).group())
        if (id > max_id):
           max_id = id
    return max_id

def read_all_graphs(folder=GRAPH_FOLDER, get_pos = True):
    """Reads all the graphs from the folder into a list"""
    graph_files = re.compile(r'^(star_|random_|grid_)')
    files = os.listdir(folder)
    graphs = []
    for file in files:
        if graph_files.search(file):
            g = graph_from_csv(file, get_pos= get_pos)
            graphs.append(g)
    return graphs

def read_all_star_graphs(folder=GRAPH_FOLDER, get_pos = True):
    """Reads all the star graphs from the folder into a list"""
    star_files = re.compile(r'^star_')
    graph_files = os.listdir(folder)
    graphs = []
    for file in graph_files:
        if star_files.search(file):
            g = graph_from_csv(file, get_pos= get_pos)
            graphs.append(g)
    return graphs

def read_all_grid_graphs(folder=GRAPH_FOLDER, get_pos = True):
    """Reads all the grid graphs from the folder into a list"""
    grid_files = re.compile(r'^grid_')
    graph_files = os.listdir(folder)
    graphs = []
    for file in graph_files:
        if grid_files.search(file):
            g = graph_from_csv(file, get_pos= get_pos)
            graphs.append(g)
    return graphs

def read_all_random_graphs(folder=GRAPH_FOLDER, get_pos = True):
    """Reads all the random graphs from the folder into a list"""
    random_files = re.compile(r'^random_')
    graph_files = os.listdir(folder)
    graphs = []
    for file in graph_files:
        if random_files.search(file):
            g = graph_from_csv(file, get_pos= get_pos)
            graphs.append(g)
    return graphs


def generate_random_graphs(num, min_node = 50, max_nodes = 500, old_graphs = None):
    """generates and saves random graphs
    :param num: number of graphs to generate
    :param min_node: minimum number of nodes in the graphs
    :param max_node: maximum number of nodes in the graphs
    :param old_graphs: previous graphs, so they don't have to parsed again
    :returns new_graphs: a list of the generated graphs"""
    # read the previuous graphs for checking isomorphism
    last_id = get_last_id()
    if old_graphs == None:
        graphs = read_all_graphs()
    else:
        graphs = old_graphs

    # edge ratios
    edge_ratios = np.random.beta(2, 1000, num)
    idx = 0
    new_graphs = []
    while len(new_graphs) < num:
        g = nx.fast_gnp_random_graph(np.random.randint(min_node, max_nodes),edge_ratios[idx])
        # check for isomorphism
        is_unique = all(not nx.is_isomorphic(g, graph) for graph in graphs + new_graphs)
        if is_unique:
            # update attributes and save
            idx = idx + 1
            last_id = last_id + 1
            attrs_g = {'id': last_id, 'type': 'random'}
            g.graph.update(attrs_g)
            new_graphs.append(g)
            graph_to_csv(g, last_id, 'random')
    return new_graphs

def generate_star_graphs(num, min_nodes = 5, max_nodes = 100, old_graphs = None):
    """Genarates star graphs, each having one more node than the last starting with 6 nodes
    :param num: number of graphs to generate
    :param old_graphs: previous graphs, so they don't have to parsed again
    :returns new_graphs: a list of the generated graphs"""
    # read the previuous graphs for checking isomorphism
    last_id = get_last_id()
    if old_graphs == None:
        graphs = read_all_graphs()
    else:
        graphs = old_graphs
    
    idx = 0
    new_graphs = []
    while len(new_graphs) < num:
        g = nx.star_graph(np.random.randint(min_nodes, max_nodes))
        # check for isomorphism
        is_unique = all(not nx.is_isomorphic(g, graph) for graph in graphs + new_graphs)
        if is_unique:
            # update attributes and save
            idx = idx + 1
            last_id = last_id + 1
            attrs_g = {'id': last_id, 'type': 'star'}
            g.graph.update(attrs_g)
            new_graphs.append(g)
            graph_to_csv(g, last_id, 'star')
    return new_graphs

def generate_grid_graphs(num, min_size = 5, max_size = 100, old_graphs = None):
    """Generates grid graphs
    :param num: number of graphs to generate
    :param min_size: minimum dimension of the grid
    :param max_size: maximum dimension of the grid
    :param old_graphs: previous graphs, so they don't have to parsed again
    :returns new_graphs: a list of the generated graphs"""
    # read the previuous graphs for checking isomorphism
    last_id = get_last_id()
    if old_graphs == None:
        graphs = read_all_graphs()
    else:
        graphs = old_graphs
    
    idx = 0
    new_graphs = []
    while len(new_graphs) < num:
        # nx.grid_2d_graph labels the nodes with their position in a grid layout so it needs to be relabeled
        h = nx.grid_2d_graph((np.random.randint(min_size, max_size)),(np.random.randint(min_size, max_size)))
        # mapping for relabeling
        mapping = {}
        for idx, node in enumerate(h.nodes):
            mapping[node] = idx
        g = nx.relabel_nodes(h, mapping)
        # saving the pos information
        for i, pos in enumerate(list(h.nodes)):
            g.nodes[i]['pos'] = pos
        # check for isomorphism
        is_unique = all(not nx.is_isomorphic(g, graph) for graph in graphs + new_graphs)
        if is_unique:
            # update attributes and save
            idx = idx + 1
            last_id = last_id + 1
            attrs_g = {'id': last_id, 'type': 'grid'}
            g.graph.update(attrs_g)
            new_graphs.append(g)
            graph_to_csv(g, last_id, 'grid')
            final_pos_to_csv(g)
    return new_graphs

def generate_graphs(num_random = 900, num_star = 50, num_grid= 50):
    """Generates graphs
    :param num_random: Number of random graphs to generate
    :param num_star: Number of star graphs to generate
    :param num_grid: Number of grid graphs to generate
    :returns graphs: a list of the generated graphs"""
    # get all graphs for isomorphism checking
    graphs = read_all_graphs()
    random_graphs = generate_random_graphs(num_random, old_graphs=graphs)
    print("Generating random graphs finished!")
    # for generating different star graphs
    star_graphs = read_all_star_graphs()
    max_node = 0
    for graph in star_graphs:
        if graph.number_of_nodes() > max_node:
            max_node = graph.number_of_nodes()
    star_graphs = generate_star_graphs(num_star, min_nodes=max(5,max_node), old_graphs = graphs + random_graphs)
    print("Generating star graphs finished!")
    grid_graphs = generate_grid_graphs(num_grid, old_graphs = graphs + random_graphs + star_graphs)
    print("Generating grid graphs finished!")
    return random_graphs + star_graphs + grid_graphs