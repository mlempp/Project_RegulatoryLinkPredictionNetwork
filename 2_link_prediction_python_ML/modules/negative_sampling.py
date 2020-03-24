import numpy as np

def negative_sampling(all_edges, negative_rate):
    
    met_nodes   = np.unique(all_edges[0,:])
    rxn_nodes   = np.unique(all_edges[1,:])
    edge        = np.array([[],[]])
    edge_tuple  = [tuple(x) for x in np.transpose(edge)]
    exclusions  = [tuple(x) for x in np.transpose(np.array(all_edges[0:2,:]))]
    

    while edge.shape[1] < np.round(negative_rate * all_edges.shape[1]):
        rnd_met_id = np.random.randint(len(met_nodes), size = 1)
        rnd_rxn_id = np.random.randint(len(rxn_nodes), size = 1)
        
        source_node = met_nodes[rnd_met_id]
        target_node = rxn_nodes[rnd_rxn_id]
        tentative_edge = tuple((source_node[0],target_node[0]))   
        if tentative_edge not in edge_tuple and exclusions:
           edge_tuple.append(tentative_edge)
           edge = np.array(np.transpose(edge_tuple))
    
    negative_edges = edge
    edge_class = np.zeros((1, negative_edges.shape[1])).astype(int)
    negative_edges = np.concatenate((negative_edges,edge_class), axis = 0)
    
    pos = np.ones((1, all_edges.shape[1])).astype(int)
    neg = np.zeros((1, negative_edges.shape[1])).astype(int)
    
    all_edges       = np.concatenate((all_edges, pos), axis = 0)
    negative_edges  = np.concatenate((negative_edges, neg), axis = 0)
    
    edges = np.concatenate((all_edges,negative_edges), axis = 1)
    return edges