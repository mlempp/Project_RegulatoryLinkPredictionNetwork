import numpy as np
from matplotlib.colors import ListedColormap
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def decision_plot (data_set, x2, epoch, epochs):
    yks = data_set.y.cpu().numpy()# +  data_set.y2.cpu().numpy()
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    colors= ['cyan', 'red', 'blue'] 
    X=x2.detach().numpy()          
    ax1=plt.subplot(1, epochs/500, (epochs/500)/(epochs/epoch))
    h = .02
    x_min, x_max = -2,3 #X[:, 0].min() - 0.1, X[:, 0].max() + 0.1 
    y_min, y_max = -2,3 #X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    newdata = np.c_[xx.ravel(), yy.ravel()]
    XX = torch.Tensor(newdata)
    yhat = F.log_softmax(XX, dim=1)
    _,hsmf2 = yhat.max(dim=1)
    yhat = hsmf2.numpy().reshape(xx.shape)
    #plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    ax1.plot([-3, 3], [-3,3], 'k-')
    ax1.scatter(X[:,0], X[:,1], c=yks, cmap=ListedColormap(colors), s= 1)
    ax1.set_ylim([-3,3])
    ax1.set_xlim([-1.5,3])
