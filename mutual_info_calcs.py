import networkentropy as nent
import numpy as np
import matplotlib.pyplot as plt

#generates the graph from an edge list file
graph = nent.graph_from_file('network_data/fb-pages-tvshow.edges')
nent.reduce_and_relabel(graph)
#sets up the initial parameters and intervals for the swapping procedure
swaps = [0,100,200,400,800,1600,3200]
iterations = 100
successCount = 0
#initialises the lists for recording critical fraction and mutual information values
randCritAve = []
randError = []
targCritAve = []
targError = []
standardMutuals = []
clusterMutuals = []

#iterates over edge swaps until reaching a cutoff number of swaps
while successCount <= max(swaps)+1:
    #at set intervals, records mutual information and critical fraction values
    if successCount in swaps:
        #informs the user when measurements are being taken
        print('Successful swaps = ' + str(successCount))
        standardMutuals.append(graph.mutual_info(clusterAdjust=False))
        clusterMutuals.append(graph.mutual_info(clusterAdjust=True))
        #simulates the critical fraction measurement multiple times in order to obtain an average
        randCrits = [nent.sim_crit_frac(graph) for i in range(iterations)]
        targCrits = [nent.sim_crit_frac(graph,targeting=True) for i in range(iterations)]
        randCritAve.append(np.average(randCrits))
        randError.append(np.std(randCrits))
        targCritAve.append(np.average(targCrits))
        targError.append(np.std(targCrits))
    #performs a correlation preserving swap on the graph
    nent.correlation_preserve_swap(graph,stayConnected=True)
    successCount += 1

#plots random critical fraction against swaps
plt.figure()
plt.errorbar(y = randCritAve,x=swaps,yerr = randError,fmt ='s',capsize=2,color='green',label='Random Failure')
plt.errorbar(y = randCritAve,x=swaps,color='green',linestyle='dashed')
plt.xlim(-100,3500)
plt.ylim(0.84,0.94)
plt.legend()
plt.ylabel(r'$f_c$',fontdict = {'fontsize':16},rotation = 0)
plt.xlabel('Number of Edge Swaps',fontdict = {'fontsize':12})

#plots mutual information with clustering against random critical fraction
plt.figure()
plt.errorbar(x = randCritAve,y=clusterMutuals,xerr = randError,fmt ='s',capsize=2,color='green',label='Random Failure')
plt.errorbar(x = randCritAve,y=clusterMutuals,color='green',linestyle='dashed')
plt.ylim(0.3,1.2)
plt.xlim(0.84,0.94)
plt.legend()
plt.ylabel('Mutual Information with Clustering',fontdict = {'fontsize':12})
plt.xlabel(r'$f_c$',fontdict = {'fontsize':16})

#plots targeted critical fraction against swaps
plt.figure()
plt.errorbar(y = targCritAve,x=swaps,yerr = targError,fmt ='o',capsize=2,color='purple',label='Targeted Attack')
plt.errorbar(y = targCritAve,x=swaps,color='purple',linestyle='dashed')
plt.xlim(-100,3500)
plt.ylim(0.34,0.5)
plt.legend()
plt.ylabel(r'$f_c$',fontdict = {'fontsize':12},rotation = 0)
plt.xlabel('Number of Edge Swaps',fontdict = {'fontsize':12})

#plots mutual information with clustering against targeted critical fraction
plt.figure()
plt.errorbar(x = targCritAve,y=clusterMutuals,xerr = targError,fmt ='o',capsize=2,color='purple',label='Targeted Attack')
plt.errorbar(x = targCritAve,y=clusterMutuals,color='purple',linestyle='dashed')
plt.ylim(0.3,1.2)
plt.xlim(0.34,0.5)
plt.legend()
plt.ylabel('Mutual Information with Clustering',fontdict = {'fontsize':12})
plt.xlabel(r'$f_c$',fontdict = {'fontsize':16})