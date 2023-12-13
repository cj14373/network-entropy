import numpy as np
import networkx as nx
from scipy.special import erf
from scipy.optimize import minimize_scalar
import re

#this class acts as an extension of the nx.Graph class, adding several new functions for measuring the properties of a graph
class extendGraph(nx.Graph):
    
    #calculates degree distribution
    def degree_dist(self):
        return [degree/self.number_of_nodes() for degree in nx.degree_histogram(self)]
    
    #calculates average degree
    def expected_degree(self):
        return 2*self.number_of_edges()/self.number_of_nodes()
    
    #calculates average degree squared
    def expected_degree_square(self):
        degreeDist = self.degree_dist()
        return sum((degree**2)*degreeDist[degree] for degree in range(len(degreeDist)))
    
    #calculates remaining degree distribution
    def remain_dist(self):
        degreeDist = self.degree_dist()
        return [(degree+1)*degreeDist[degree+1]/self.expected_degree() for degree in range(len(degreeDist)-1)]
    
    #organises nodes into groups based on their degree values
    def degree_groups(self):
        degreeDist = self.degree_dist()
        degreeGroups = {degree:[] for degree in range(len(degreeDist)) if degreeDist[degree] != 0}
        for node in self.nodes():
            degreeGroups[self.degree(node)].append(node)
        return degreeGroups
    
    #calculates the molloy reed critical fraction
    def molloy_reed(self):
        return 1-1/(self.expected_degree_square()/self.expected_degree() - 1)
    
    #calculates joint distribution
    def joint_dist(self,clusterAdjust=False):
        #if adjusting for clustering, degree values of nodes are reduced according to how many clusters they are in
        if clusterAdjust == True:
            degreeList = []
            for edge in self.edges():
                degreeAdjust = len([neigh for neigh in nx.common_neighbors(self,edge[0],edge[1])])
                degreeList.append(sorted([self.degree(edge[0])-degreeAdjust,self.degree(edge[1])-degreeAdjust]))
        #if clustering is not adjusted for, degree values are recorded without adjustment
        else:
            degreeList = [sorted([self.degree(edge[0]),self.degree(edge[1])]) for edge in self.edges()]
        #creates a matrix recording how many pairs of given degrees exist
        remDist = self.remain_dist()
        jointDegreeMatrix = np.array([[0 for q in remDist] for r in remDist])
        for degree in degreeList:
            jointDegreeMatrix[degree[0]-1][degree[1]-1] += 1
            jointDegreeMatrix[degree[1]-1][degree[0]-1] += 1
        #normalises and returns the joint distribution
        return jointDegreeMatrix/sum(sum(row) for row in jointDegreeMatrix)
    
    #calculates mutual information
    def mutual_info(self,clusterAdjust):
        remDist = self.remain_dist()
        productDist = np.array([q*np.array(remDist) for q in remDist])        
        jointDist = self.joint_dist(clusterAdjust)
        return sum(sum([jointDist[i][j]*(np.log(jointDist[i][j])-np.log(productDist[i][j])) for i in range(len(jointDist)) 
                        if productDist[i][j] != 0 and jointDist[i][j] != 0]) for j in range(len(jointDist)))
    
#alters a graph in place, reducing it to its largest connected components and relabelling its nodes
def reduce_and_relabel(graph):
    largestComp = sorted([list(c) for c in nx.connected_components(graph)],key=len)[-1]
    graph.remove_nodes_from([node for node in graph.nodes() if node not in largestComp])
    mapping = dict(list(zip(sorted(graph),list(range(graph.number_of_nodes()+1)))))
    nx.relabel_nodes(graph,mapping,copy=False)

#reads network files from a directory and creates a graph
def graph_from_file(path):
    with open(path) as inputfile:
        lines = inputfile.readlines()
    inputfile.close()
    edgeList = [re.split('[,|\t| |\n]',l)[:2] for l in lines if not l.startswith(('%','#'))]
    graph = extendGraph(nx.Graph())
    graph.add_edges_from(edgeList)
    return graph
    
#performs a correlation preserving edge swap on a graph
def correlation_preserve_swap(graph,stayConnected=False,maxDepth=1000):
    #chooses a group of nodes according to their degree values
    chosenGroup = graph.degree_groups()[np.random.choice(range(len(graph.degree_dist())), p = graph.degree_dist())]
    #initialises parameters for checking whether a successful swap has occurred, and how many failures have occurred
    successfulSwap = False
    depth = 0
    #attempts swaps until successful or until the maximum number of allowable failures occurs
    while successfulSwap == False:
        if depth < maxDepth:
            if len(chosenGroup) > 1:
                #if the degree group has more than two members, chooses two nodes u and v from the group
                #choses edges (u,x) and (v,y), removes these from the graph and adds (u,y) and (v,x)
                try:
                    nodeU,nodeV = np.random.choice(chosenGroup,size = 2,replace = False)
                    nodeX = np.random.choice([neigh for neigh in graph[nodeU] if neigh not in graph[nodeV]])
                    nodeY = np.random.choice([neigh for neigh in graph[nodeV] if neigh not in graph[nodeU]])
                    graph.remove_edges_from([(nodeU, nodeX),(nodeV,nodeY)])
                    graph.add_edges_from([(nodeU,nodeY),(nodeV,nodeX)])
                    #ensures the graph remains connected after the swap, if this is required
                    if stayConnected == True:
                        if nx.is_connected(graph) == True:
                            successfulSwap = True
                        #if the graph is disconnected, reverses the swap and tries again
                        else:
                            graph.add_edges_from([(nodeU, nodeX),(nodeV,nodeY)])
                            graph.remove_edges_from([(nodeU,nodeY),(nodeV,nodeX)])
                            chosenGroup = graph.degree_groups()[np.random.choice(range(len(graph.degree_dist())), p = graph.degree_dist())]
                            depth += 1
                    else:
                        successfulSwap = True
                #if x and y cannot be chosen such that both (u,y) and (v,x) edges do not already exist, a new degree group is chosen
                except ValueError:
                    chosenGroup = graph.degree_groups()[np.random.choice(range(len(graph.degree_dist())), p = graph.degree_dist())]
                    depth += 1
            #if the chosen degree group has only one member, a new degree group is chosen
            else:
                chosenGroup = graph.degree_groups()[np.random.choice(range(len(graph.degree_dist())), p = graph.degree_dist())]
                depth += 1
        #if too many failures occur, an exception is raised
        else:
            raise Exception("Maximum recursion depth reached without finding suitable swap candidates.")

#calculates the critical fraction for random node removal via simulation
def sim_crit_frac(graph,targeting=False,criticalPercent=0.01):
    #if targeting by degree value, creates a target list in ascending degree order, randomised within degree groups
    if targeting == True:
        targetList = []
        for group in graph.degree_groups().values():
            targetList.extend(list(np.random.choice(group,len(group),replace=False)))
    #if not targeting by degree value, creates a target list in random order
    else:
        targetList = [n for n in graph.nodes()]
        np.random.shuffle(targetList)
    #intialises variables corresponding to an empty graph
    activeNodes = []
    largestComp = 0
    trees = {n:n for n in graph.nodes()}
    sizes = {n:1 for n in graph.nodes()}
    # iteratively adds nodes to the empty graph until the largest component reaches a specified critical point
    while largestComp < round(graph.number_of_nodes()*criticalPercent):
        #selects the next node
        node = targetList[len(activeNodes)]
        #searches over all neighbours present in the graph to determine which component nodes belong to
        for neigh in graph.neighbors(node):
            if neigh in activeNodes:
                #finds the "roots" of the selected node and neighbour via a recursive search
                nodeRoot = trees[node]
                neighRoot = trees[neigh]
                while nodeRoot != trees[nodeRoot]:
                    trees[nodeRoot] = trees[trees[nodeRoot]]
                    nodeRoot = trees[nodeRoot]
                while neighRoot != trees[neighRoot]:
                    trees[neighRoot] = trees[trees[neighRoot]]
                    neighRoot = trees[neighRoot]
                #if the selected node and neighbour do not already have a common root, the labels and component sizes are updated
                if nodeRoot != neighRoot:
                    if sizes[nodeRoot] >= sizes[neighRoot]:
                        trees[neighRoot] = trees[nodeRoot]
                        sizes[nodeRoot] += sizes[neighRoot]
                        sizes[neighRoot] = 0
                    else:
                        trees[nodeRoot] = trees[neighRoot]
                        sizes[neighRoot] += sizes[nodeRoot]
                        sizes[nodeRoot] = 0
        # records the largest component size and updates the active nodes in the graph
        largestComp = max(sizes.values())
        activeNodes.append(node)
    return 1 - len(activeNodes)/graph.number_of_nodes()
    
#the "zed" function for the truncated normal distribution
def zed_func(mu,sigma):
    return 0.5*(erf(mu/(sigma*(2**0.5))) + 1)

#the "phi" function for the truncated normal distribution
def phi_func(mu,sigma):
    return (2*np.pi)**-0.5 * np.exp(-((mu/sigma)**2)/2)

#calculates ratio between first and second moments of the truncated normal distribution and then uses this to calculate Molloy-Reed critical fraction
def trunc_crit(mu,sigma):
    kappa = (mu**2 + sigma**2 + mu*sigma*(phi_func(mu,sigma) / zed_func(mu,sigma)))/(mu + (sigma * (phi_func(mu,sigma) / zed_func(mu,sigma))))
    return 1 - float(1)/(kappa - 1)

#degree distribution entropy of the truncated normal distribution
def trunc_entropy(mu,sigma):
    truncEnt = np.log(((2*np.pi*np.exp(1))**0.5) * sigma * zed_func(mu,sigma)) - ((mu/(2*sigma)) * (phi_func(mu,sigma)/zed_func(mu,sigma)))
    return truncEnt

#power law probability distribution
def power_law_distribution(alpha,minDegree,maxDegree=1000):
    norm = sum((nVal + minDegree)**(-alpha) for nVal in range(maxDegree))
    return [(k**(-alpha))/norm for k in range(minDegree,maxDegree)]

#log normal probability distribution
def log_normal_distribution(mu,sigma,maxDegree=1000):
    rawDist = [np.exp(-((np.log(k)-mu)**2)/(2*sigma**2)) for k in range(1,maxDegree)]
    return [0]+[p/sum(rawDist) for p in rawDist]

#given an expected degree and minimum degree value, finds the appropriate value of alpha for the power law distribution
def alpha_finder(minDegree,expectedDegree,maxDegree=1000):
    
    def alpha_min(alphaGuess,maxDegree=maxDegree,expectedDegree=expectedDegree,minDegree=minDegree):
        probDist = power_law_distribution(alphaGuess,minDegree,maxDegree)
        calcExpect = sum(k*p for k,p in zip(range(minDegree,maxDegree),probDist))
        return np.abs(calcExpect-expectedDegree)

    alpha = minimize_scalar(alpha_min).x
    return alpha

#given an expected degree and sigma value, finds the appropriate value of mu for the log normal distribution
def mu_finder(sigma,expectedDegree,maxDegree=1000):
    
    def mu_min(muGuess,sigma=sigma,expectedDegree=expectedDegree,maxDegree=maxDegree):
        probDist = log_normal_distribution(muGuess,sigma,maxDegree)
        calcExpect = sum(k*p for k,p in zip(range(maxDegree),probDist))
        return np.abs(calcExpect-expectedDegree)
    
    mu = minimize_scalar(mu_min).x
    return mu