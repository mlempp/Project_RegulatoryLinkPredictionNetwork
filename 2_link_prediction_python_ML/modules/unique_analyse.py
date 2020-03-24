import numpy as np

def unique_analyse(uniques, p, data_set):
    
    edge_set    = [tuple(x) for x in np.transpose(np.array(data_set.edge_index))]
    edge_unique = [tuple(x) for x in np.transpose(np.array(uniques))]
    inter = list(set(edge_set) & set(edge_unique))
    ind = [edge_set.index(a) for a in inter]
    unique_pred = p[ind]
    if np.array(unique_pred.sum()) > 0:
        answer = 'yes'
    else:
        answer = 'no'
    return answer