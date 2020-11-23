import numpy as np
import kmeans
import common
import naive_em
import em

# = np.loadtxt("toy_data.txt")

#For netflix incomplete data

X = np.loadtxt("netflix_incomplete.txt")

X_gold = np.loadtxt("netflix_complete.txt")

#X = np.loadtxt("netflix_incomplete.txt")

########## Begin: kMeans vs EM (and BIC) #############
#K = [1, 2, 3, 4]    # Clusters to try
K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  
seeds = [0, 1, 2, 3, 4]     # Seeds to try

costs_KMeans = [0, 0, 0, 0, 0]
costs_EM = [0, 0, 0, 0, 0]

best_seed_kMeans = [0, 0, 0, 0]
best_seed_EM = [0, 0, 0, 0]


mixtures_kMeans = [0, 0, 0, 0, 0]
mixtures_EM = [0, 0, 0, 0, 0]


posts_kMeans = [0, 0, 0, 0, 0]
posts_EM = [0, 0, 0, 0, 0]

# BIC score of cluster
bic = [0., 0., 0., 0.]

for k in range(len(K)):
    for i in range(len(seeds)):

        # Run K-Means    
        mixtures_kMeans[i], posts_kMeans[i], costs_KMeans[i] = \
        kmeans.run(X, *common.init(X, K[k], seeds[i]))

        # Run Naive EM
        # mixtures_EM[i], posts_EM[i], costs_EM[i] = \
        # naive_em.run(X, *common.init(X, K[k], seeds[i]))

        # Run EM

        mixtures_EM[i], posts_EM[i], costs_EM[i] = \
        em.run(X, *common.init(X, K[k], seeds[i]))

        print("==================Clusters: ", k+1, "=============")
        print("Lowest cost using k-Means: ", np.min(costs_KMeans))
        print("Highest log likelihood using EM: ", np.max(costs_EM))

        #BIC score for EM
    #bic[k] = common.bic(X, mixtures_EM[best_seed_EM[k]], np.max(costs_EM))
    
# Print the best K based on BIC
print("================= BIC ====================")
print("Best K is:", np.argmax(bic)+1)
print("BIC for the best K is:", np.max(bic))

# X_predicted = em.fill_matrix(X, mixtures_EM[[best_seed_EM[12], np.max(costs_EM)]])

# print("==================RMSE=======================")
# print("RMSE: ", common.rmse(X_gold, X_predicted))