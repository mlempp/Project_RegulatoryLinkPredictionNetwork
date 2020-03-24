from sklearn.metrics import roc_curve, auc, confusion_matrix


def evaluate_results(target, prediction,caze):
    
    fpr, tpr, _ = roc_curve(target, prediction)
    roc_auc = auc(fpr, tpr)
    tn, fp, fn, tp= confusion_matrix(target,prediction).ravel()
    recall = tp/(tp+fn)
    TNR = tn/(tn+fp)
    FNR = 1-recall
    FPR = 1-TNR
    Balanced_Accuracy = (recall+TNR)/2
    precision = tp/(tp+fp)
    f1 = 2*precision*recall/(precision+recall)
    if caze == 1:
       print('Train Results')
    elif caze == 2:
        print('Test Results')
    else:
        print('Validation Results')        

    print('Recall/ True Positive Rate: {:.4f}'.format(recall))
    print('Specificity/ True Negative Rate: {:.4f}'.format(TNR))
    print('False Negative Rate: {:.4f}'.format(FNR))
    print('False Positive Rate: {:.4f}'.format(FPR))
    print('ROC_AUC: {:.4f}'.format(Balanced_Accuracy))
    return recall, TNR, FNR, FPR, Balanced_Accuracy, precision, f1
