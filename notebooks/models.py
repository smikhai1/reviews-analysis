import numpy as np
import scipy as scp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from collections import Counter
from nltk.corpus import stopwords

class Word2Vec(object):
    
    def __init__(self, sentences):
        """
        sentences -- preprocessed sentences of reviews
        vocab -- vocabulary of a corpus words; {words: index}
        D -- word-context co-occurence matrix
        W -- matrix of words embeddings
        C -- matrix of contexts embeddings
        d -- dimension of words and reviews embeddings
        """

        self.sentences = sentences
        self.vocab = None
        self.D = None
        self.W = None
        self.C = None
        self.d = 200
        self.Wt = None
        self.Ct = None
    
    ###################### DATA PROCESSING ######################
    
    ###### Create vocabulary from given sentences ######
    
    def create_vocabulary(self, r=200):
        """
        r -- word occurence treshold
        """
        
        self.vocab = dict()
        word_count = dict()
        idx = 0
        
        print('Creating vocabulary')
        for sentence in self.sentences:
            for word in sentence:
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1

        for word, count in word_count.items():
            if word_count[word] >= r:
                self.vocab[word] = idx
                idx += 1
                
        
    
    ###### Create word-context co-occurence matrix ######
    
    def create_corpus_matrix(self, L=2):
        """
        L -- size of the sliding window
        """
        
        print('Creating corpus matrix')
        # initialization
        words_counts = Counter()
        for sentence_index, sentence in enumerate(self.sentences):
            for word_index, word in enumerate(sentence):
                if word in self.vocab:
                    around_indexes = [i for i in range(max(word_index - L, 0), 
                                                       min(word_index + L + 1, len(sentence))) 
                                      if i != word_index]
                    for occur_word_index in around_indexes:
                            occur_word = sentence[occur_word_index]
                            if occur_word in self.vocab:
                                skipgram = (word, occur_word)
                                if skipgram in words_counts:
                                    words_counts[skipgram] += 1
                                else:
                                    words_counts[skipgram] = 1
        rows = list()
        cols = list()
        values = list()


        for (word_1, word_2), sharp in words_counts.items():                                            
            rows.append(self.vocab[word_1])
            cols.append(self.vocab[word_2])
            values.append(sharp)

        self.D = scp.sparse.csr_matrix((values, (rows, cols)))
        
        
    ###################### AUXILARY FUNCTIONS ######################
    
    ###### Sigmoid ######
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    ##### Loss function #####
    def loss(self, k):
        wc_ = self.D.sum()
        w_ = np.array(self.D.sum(axis=1))
        c_ = np.array(self.D.sum(axis=0))
        loss = self.D.toarray() * np.log(self.sigmoid(self.X)) + (k * w_ * c_ / wc_) * np.log(self.sigmoid(-self.X))
        return np.sum(loss)
    
    ###### Gradient of the objective function ######
    def grad(self, k):
        wc_ = self.D.sum()
        w_ = np.array(self.D.sum(axis=1))
        c_ = np.array(self.D.sum(axis=0))
        gr = self.D.toarray() * self.sigmoid(-self.X) - (k * w_ * c_ / wc_) * self.sigmoid(self.X)
        return gr

    ###################### DIFFERENT METHODS FOR WORD EMBEDDINGS COMPUTATION ######################
    
    ###### Create matrix of words embeddings by IMF ######    
    def compute_embedds_IMF(self, k, alpha=.5):
        """
        k -- negative sampling hyperparameter
        alpha -- hyperparameter for matrix representation
        """
        print('Computing of words embeddings')
        all_observations = self.D.sum()

        rows = []
        cols = []
        sppmi_values = []

        sum_over_words = np.array(self.D.sum(axis=0)).flatten()
        sum_over_contexts = np.array(self.D.sum(axis=1)).flatten()

        for word_index_1, word_index_2 in zip(self.D.nonzero()[0], 
                                                  self.D.nonzero()[1]):
            sg_count = self.D[word_index_1, word_index_2]

            pwc = sg_count
            pw = sum_over_contexts[word_index_1]
            pc = sum_over_words[word_index_2]

            spmi_value = np.log2(pwc * all_observations / (pw * pc * k))
            sppmi_value = max(spmi_value, 0)

            rows.append(word_index_1)
            cols.append(word_index_2)
            sppmi_values.append(sppmi_value)

        sppmi_mat = scp.sparse.csr_matrix((sppmi_values, (rows, cols)))
        U, S, V = scp.sparse.linalg.svds(sppmi_mat, self.d)
        self.W = U @ np.diag(np.power(S, alpha))
        self.C = np.diag(np.power(S, alpha)) @ V
        self.X = self.W @ self.C
        
        # SGNS objective
        print("Value of the SGNS's objective: ", self.loss(k))
    
    ###### Create matrix of words embeddings by Riemannian optimization ######
    def compute_embedds_riem(self, k, step=5e-5, max_iter=20, alpha=.5):
        self.X = self.W @ self.C
        U, S, Vt = np.linalg.svd(self.X, full_matrices=False)
        U, S, Vt = U[:, :self.d], S[:self.d], Vt[:self.d, :]
        
        for i in range(max_iter):
            print(f"Value of the SGNS's objective on the {i} iteration: \n {self.loss(k)}")
            grad_step = self.X + step * self.grad(k)
            U, S = np.linalg.qr(grad_step @ Vt.T)
            V, St = np.linalg.qr(grad_step.T @ U)
            self.X = U @ S @ V.T
            
        U_, S_, Vt_ = np.linalg.svd(self.X)
        U_, S_, Vt_ = U_[:, :self.d], S_[:self.d], Vt_[:self.d, :]
        self.W = U_ @ np.power(np.diag(S_), alpha)
    
     ###### Create matrix of words embeddings by EMF (AMEMF) ######    
    def compute_embedds_EMF(self, k, step=1e-3, max_iter=50, eps=1e-3, iters=20):
        """
        k -- negative sampling hyperparameter
        """
        # initialization
        m = len(self.vocab)
        self.Wt = np.random.rand(self.d, m)
        self.Ct = np.random.rand(self.d, m)
        Wt_prvs = np.zeros(self.Wt.shape)
        Ct_prvs = np.zeros(self.Ct.shape)
        
        wc_ = self.D.sum()
        w_ = np.array(self.D.sum(axis=1))
        c_ = np.array(self.D.sum(axis=0))
        Q = self.D.toarray() + k * w_ * c_ / wc_
        
        error = lambda M, M_prvs: np.linalg.norm(M - M_prvs)
        
        for i in range(max_iter):
            print(f'{i} iteration')
            # minimize over C
            
            for j in range(iters):
                Wt_prvs = self.Wt
                E = Q * self.sigmoid(self.Ct.T @ self.Wt)
                self.Wt = self.Wt - step * self.Ct @ (E - self.D.toarray())
                print(error(self.Wt, Wt_prvs))
                
                """if error(self.Wt, Wt_prvs) <= eps:
                    break"""
            print('First loop finished')
            
            # minimize over W
            
            for j in range(iters):
                Ct_prvs = self.Ct
                E = Q * self.sigmoid(self.Ct.T @ self.Wt)
                self.Ct = self.Ct - (step * (E - self.D.toarray()) @ self.Wt.T).T
                
                """if error(self.Ct, Ct_prvs) > eps:
                    break"""
            print('Second loop finished')
        
    ###### Get vector embedding for a given word ######
    def get_word_embedding(self, word):
        if word in self.vocab:
            idx = self.vocab[word]
            return self.W[idx, :]
        else:
            print('This word is not in the vocabulary')
    
    ###### Get vector embedding for a given word ######
    def get_word_embedding2(self, word):
        if word in self.vocab:
            idx = self.vocab[word]
            return self.Wt.T[idx, :]
        else:
            print('This word is not in the vocabulary')
    
    ###################### REVIEW EMBEDDINGS ######################
    
    ##### Compute review embeddings #####
    def get_review_embedding(self, review):
        """
        review -- current review to be embedded
        """

        review_vec = np.zeros(self.d)
        words_count = 0
        stops = set(stopwords.words("english"))

        for word in review:
            if (word in self.vocab) and not (word in stops):
                review_vec += self.get_word_embedding(word)
                words_count += 1
        review_vec /= words_count
        return review_vec
    
    ##### Compute review embeddings #####
    def get_review_embedding2(self, review):
        """
        review -- current review to be embedded
        """

        review_vec = np.zeros(self.d)
        words_count = 0
        stops = set(stopwords.words("english"))

        for word in review:
            if (word in self.vocab) and not (word in stops):
                review_vec += self.get_word_embedding2(word)
                words_count += 1
        review_vec /= words_count
        return review_vec
    
    ##### Create matrix 'embeddings-reviews' #####
    def get_features_matrix(self, reviews):
        """
        reviews -- the whole collection of reviews
        """
        X = np.zeros((len(reviews), self.d))
        for idx, review in enumerate(reviews):
            X[idx, :] = self.get_review_embedding(review)
        return X 
    
    def get_features_matrix2(self, reviews):
        """
        reviews -- the whole collection of reviews
        """
        X = np.zeros((len(reviews), self.d))
        for idx, review in enumerate(reviews):
            X[idx, :] = self.get_review_embedding2(review)
        return X 