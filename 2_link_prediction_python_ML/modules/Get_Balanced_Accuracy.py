from sklearn.metrics import confusion_matrix

def Get_Balanced_Accuracy(target, prediction): #is the same as AUC
    tn, fp, fn, tp= confusion_matrix(target,prediction).ravel()
    recall = tp/(tp+fn)
    TNR = tn/(tn+fp)
    Balanced_Accuracy = (recall + TNR)/2 #equals AUC
    criterium2 = recall-(1-TNR)
    precision = tp/(tp+fp)
    F1 = 2*precision*recall/(precision+recall)
    return Balanced_Accuracy, criterium2, precision, F1,recall