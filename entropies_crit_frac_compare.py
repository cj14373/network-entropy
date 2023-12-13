import networkentropy as nent
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

maxDegree = 1000
expectedDegree = 10

powerDegEnts = []
powerRemEnts = []
powerCrits = []
minDegree = 1
#iterates power law distributions over minimum degree values
while minDegree < 10:
    #finds value for alpha
    alpha = nent.alpha_finder(minDegree,expectedDegree)
    #generates power law degree distribution and measures degree distribution entropy
    powerProbDist = nent.power_law_distribution(alpha,minDegree)
    powerDegEnts.append(entropy(powerProbDist))
    #finds first and second moments and calculates Molloy-Reed critical fraction
    powerLawExpect = sum(k*p for k,p in zip(range(minDegree,maxDegree),powerProbDist))
    powerLawExpectSquare = sum((k**2)*p for k,p in zip(range(minDegree,maxDegree),powerProbDist))
    powerCrits.append(1-1/(powerLawExpectSquare/powerLawExpect-1))
    #measures power law remaining degree entropy
    powerRemEnts.append(entropy([k*p/powerLawExpect for k,p in zip(range(minDegree,maxDegree),powerProbDist)]))
    #increases minimum degree value for next iteration
    minDegree += 1

logDegEnts = []
logRemEnts = []
logCrits = []
sigma = 0.2
#iterates numerical log normal distributions over sigma values
while sigma < 3:
    #finds value for mu
    mu = nent.mu_finder(sigma,expectedDegree)
    #generates log normal degree distribution and measures degree distribution entropy
    logProbDist = nent.log_normal_distribution(mu,sigma)
    logDegEnts.append(entropy(logProbDist))
    #finds first and second moments and calculates Molloy-Reed critical fraction
    logNormExpect = sum(k*p for k,p in zip(range(maxDegree),logProbDist))
    logNormExpectSquare = sum((k**2)*p for k,p in zip(range(maxDegree),logProbDist))
    logCrits.append(1-1/(logNormExpectSquare/logNormExpect-1))
    #measures log normal remaining degree entropy
    logRemEnts.append(entropy([k*p/logNormExpect for k,p in zip(range(maxDegree),logProbDist)]))
    #increases sigma value for next iteration
    sigma += 0.2
    
theoryDegEnts = []
theoryRemEnts = []
theoryCrits = []
theorySigma = 0.01
#iterates theoretical log normal distributions over sigma values
while theorySigma < 3:
    #calculates entropy values and Molloy Reed critical fraction
    theoryDegEnts.append(0.5*(1-theorySigma**2) + np.log(expectedDegree*theorySigma*(2*np.pi)**0.5))
    theoryRemEnts.append(0.5*(1+theorySigma**2) + np.log(expectedDegree*theorySigma*(2*np.pi)**0.5))
    theoryCrits.append(1-1/(expectedDegree*np.exp(theorySigma**2)-1))
    #increases sigma value for next iteration
    theorySigma += 0.01
    
#plots degree distribution entropy against Molloy-Reed critical fraction
plt.figure()
plt.xlim(1,3.5)
plt.ylim(0.88,1)
plt.plot(powerDegEnts,powerCrits,'^',color = 'blue',label = 'Power-Law\n(numerical)')
plt.plot(logDegEnts,logCrits,'s',color = 'red',label = 'Log-Normal\n(numerical)')
plt.plot(theoryDegEnts,theoryCrits, linestyle = 'dashed', color = 'black',label = 'Log-Normal\n(theoretical)')
plt.xlabel(r'$H(p)$',fontdict = {'fontsize':16})
plt.ylabel(r'$f_c$',fontdict = {'fontsize':16},rotation=0)
plt.legend(loc = 'upper left')

#plots remaining degree entropy against Molloy-Reed critical fraction
fig = plt.figure()
ax = fig.add_subplot()
plt.xlim(1,7)
plt.ylim(0.88,1)
plt.plot(powerRemEnts,powerCrits,'^',color = 'blue',label = 'Power-Law\n(numerical)')
plt.plot(logRemEnts,logCrits,'s',color = 'red',label = 'Log-Normal\n(numerical)')
plt.plot(theoryRemEnts,theoryCrits, linestyle = 'dashed', color = 'black',label = 'Log-Normal\n(theoretical)')
ax.set_xlabel(r'$H(q)$',fontdict = {'fontsize':16})
ax.set_ylabel(r'$f_c$',fontdict = {'fontsize':16},rotation=0)
plt.legend(loc = 'upper left')