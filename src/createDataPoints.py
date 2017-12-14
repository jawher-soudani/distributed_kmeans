import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs


class Generator(object):
 
    def generateData(self,num, lbl,outputFile):
        """this function helps generate any number of points 
        for clustering using two arguments

        instances: the number of points
        outputFile: the output file
        """
        #the real centers of the clusters
        centers = [[-30, -30], [-10, -10], [-1, -1], [10, 10], [30, 30]]

        
        #Generate isotropic Gaussian blobs for clustering.
        X, labels = make_blobs(n_samples=num,
                                    centers=centers, cluster_std=3.9,
                                    n_features=2)
        
        #save points to dataframe for easy manipulation
        df = pd.DataFrame(X)
        #enregistrer les labels dans un fichier pour calculer la précision aprés 
        lb = pd.DataFrame(labels)
        #save points to output file for our mapreduce prgoram 
        df.to_csv(outputFile, header=False, index=False, sep=" ")
        lb.to_csv(lbl, header=False, index=False, sep=" ")

        #plotting the points with different colors according to each center
        #to compare it later to the clustering of our program
        plt.figure(figsize=(9,8))
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("outputFile", type=str,help="the file where do you want to save the data points")
    parser.add_argument("lbl", type=str,help="the file where do you want to save the real labels")
    parser.add_argument("instances", type=int,help="number of point you want to create for clustering")

    args = parser.parse_args()
    instanceGenerator = Generator()
    instanceGenerator.generateData(args.instances, args.lbl, args.outputFile)
