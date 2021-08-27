#
__author__ = "Chandrashish Prasad"
__license__ = "Feel free to copy, I appreciate if you learn from here"

#READING

class jaccard_kmeans:
    def __init__(self, collection, vocab=None):
        self.collection = collection
        self.vocab_txt = vocab
        self.vocab = None
        self.inertia = []
        
        self.doc={}
        self.jac_ind_matrix = {}
        self.doc_count = 0
        self.vocab_count = 0
        self.entries = 0
        self.dissimilar_seed = None
        
        self.make_doc()
        self.jac_mat()
        self.dissimilarity_index()
        if self.vocab_txt!=None:
            self.build_vocab()
    
    def build_vocab(self):
        self.vocab = [None]   
        fh = open(self.vocab_txt, 'r')
        lines = fh.readlines()
        for line in lines:
            self.vocab.append(line.strip())
        fh.close()
        
    def jaccard(self, set1, set2):
        index = float(len(set1.intersection(set2)))/float(len(set1.union(set2)))
        return index, 1-index
    
    def dissimilarity_index(self):
        self.dissimilar_seed = [0]
        for i in range(1,self.doc_count+1):
            temp = 0
            for j in range(1,self.doc_count+1):
                temp += self.jac_ind_matrix[i][j]
            self.dissimilar_seed.append(temp)
        
    def make_doc(self):
        fh = open(self.collection, 'r')
        self.doc_count = int(fh.readline().strip())
        self.vocab_count = int(fh.readline().strip())
        self.entries = int(fh.readline().strip())
        lines = fh.readlines()
        for line in lines:
            a = list(map(int, line.strip().split()))
            if len(a)>1:
                doc_id, word_id, freq = a
                if doc_id not in self.doc.keys():
                    self.doc[doc_id] = set([word_id])
                else:
                    self.doc[doc_id].add(word_id)
        fh.close()
        return
        
    def jac_mat(self):
        for i in range(1,self.doc_count+1):
            if i not in self.jac_ind_matrix.keys():
                self.jac_ind_matrix[i]= {}
            for j in range(i,self.doc_count+1):
                if j not in self.jac_ind_matrix.keys():
                        self.jac_ind_matrix[j] = {}
                jac_ind, jac_dist = self.jaccard(self.doc[i], self.doc[j])
                self.jac_ind_matrix[i][j] = jac_ind
                self.jac_ind_matrix[j][i] = jac_ind
        return
    
    def find_seeds(self):
        if self.seed_state=="random":
            self.seed = list(np.random.choice(range(1,self.doc_count+1), self.k, replace=False))
        else:
            self.seed = list(np.argsort(self.dissimilar_seed)[1:self.k+1])
     
    def initialise(self):      
        self.clusters = {}
        self.rev_clusters = {}
        for i in range(1,self.doc_count+1):
            self.rev_clusters[i] = -1
        for i in range(self.k):
            self.clusters[i] = set([self.seed[i]])
            self.rev_clusters[self.seed[i]] = i 
        for i in range(1,self.doc_count+1):
            if i in self.seed:
                continue
            temp_clust = -1
            temp_ind = -1
            for j in range(self.k):
                ind = self.jac_ind_matrix[i][self.seed[j]]
                if ind>temp_ind:
                    temp_ind = ind
                    temp_clust = j
            self.clusters[temp_clust].add(i)
            self.rev_clusters[i] = temp_clust
        return
            
    def find_clusters_by_avg(self):
        clusters = {}
        rev_clusters = {}
        self.inertia = 0
        for i in range(1, self.doc_count+1):
            temp_check = -1
            temp_clust = -1
            for j in range(self.k):
                temp = 0
                for m in self.clusters[j]:
                    temp += self.jac_ind_matrix[m][i]
                temp = temp/len(self.clusters[j])
                if(temp>temp_check):
                    temp_check = temp
                    temp_clust = j
            self.inertia += temp_check
            if temp_clust not in clusters.keys():
                clusters[temp_clust] = set([i])
            else:
                clusters[temp_clust].add(i)
            rev_clusters[i] = temp_clust
        return clusters, rev_clusters
    
    def converge(self, k, max_iter=5, jaccard_index_cluster_average=True, seed=None, seed_state = "random"):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.seed_state = seed_state
        
        if self.seed == None:
            self.find_seeds()
        self.initialise()
        
        if jaccard_index_cluster_average:   
            clusters, rev_clusters = self.find_clusters_by_avg()
        else:                              
            clusters, rev_clusters = self.find_clusters_by_arith_mean()
        self.clusters = copy.deepcopy(clusters)
        self.rev_clusters = copy.deepcopy(rev_clusters)
        
        iters = 1
        while(iters<=self.max_iter):
            if jaccard_index_cluster_average:
                clusters, rev_clusters = self.find_clusters_by_avg()
            else:
                clusters, rev_clusters = self.find_clusters_by_arith_mean()
            iters += 1
            if self.rev_clusters != rev_clusters:
                self.clusters = copy.deepcopy(clusters)
                self.rev_clusters = copy.deepcopy(rev_clusters)
            else:
                print("complete convergence at {} iterations".format(iters-1))
                break
        print("Convergence attempt for {} ietrations done".format(self.max_iter))
        return
    
    def cluster_top_words(self, count):
        cluster_words = {}
        for i in range(self.k):           
            word_freq = [0]*(self.vocab_count+1)
            for j in self.clusters[i]:    
                for m in self.doc[j]:    
                    word_freq[m]+=1
            topk = np.argsort(word_freq)[-count:]
            cluster_words[i] = set()
            for w_id in topk:
                cluster_words[i].add(self.vocab[w_id])
        return cluster_words
