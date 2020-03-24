import numpy as np

def val_miss_class(target,prediction,edges):
    prediction = np.array(prediction)
    target = np.array(target)
    edges = np.array(edges)
    pos_pred = prediction == 1
    neg_pred = prediction == 0
    
    pos_pred = np.reshape(pos_pred,(pos_pred.shape[0]))
    
    edges_pos = edges[:,pos_pred]
    edges_neg = edges[:,neg_pred]

    fp_id = target[pos_pred] == 0
    tp_id = target[pos_pred] == 1
    fn_id = target[neg_pred] == 1
    
    fp_edges = edges_pos[:,fp_id]
    fn_edges  = edges_neg[:,fn_id]
    tp_edges = edges_pos[:,tp_id]
    return fp_edges, fn_edges, tp_edges
