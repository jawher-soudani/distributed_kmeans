
# coding: utf-8

# In[ ]:
import numpy as np
from mrjob.util import to_lines
from mrjob.job import MRJob
from finalmapper import MRFinalKMeans
from math import sqrt
from kmeans import MRKMeans
import sys
import os.path
import shutil
import re


input_centroids = "/home/djo/Desktop/kmeans_mapreduce/centroids.txt"

temp_centroids_file="/tmp/centroids.txt"

def get_centroids(job, runner):
 #get the output centroid from the mrjob using runner.cat
    c = []
    for key, value in job.parse_output(runner.cat_output()):    
        value=value[:1]+key+','+value[1:]       
        c.append(value)
    return c

def get_first_c(fname):
 #get the first centroids from the file input and not from the mrjob output
    f = open(fname,'r')
    centroids=[]
    for line in f.read().split('\n'):
        if line:
                newline = re.search("\[(.*?)\]", line)
                newline = newline.group()
                newline = newline.replace("[", "").replace("]", "")
                newline.strip()
                k,x,y = newline.split(',')
                centroids.append([float(k),float(x),float(y)])
    f.close()
    return centroids



def distance_vec(d,centroid):
        #calculate the ditance between two vectors (in two dimensions)
        #np.linalg.norm(data_point - centroid)
        diff=[d1 - c2 for d1,c2 in zip(d,centroid)]
        return np.linalg.norm(diff)

def diff(k,centroids1,centroids2):
    # calculate the distance between the old centroids and the new centroids
    max_distance = 0.0
    for i in range(k):
        dist = distance_vec(centroids1[i],centroids2[i])

        if dist > max_distance:
            max_distance = dist

    return max_distance


def write_centroids(centroids):
 #write centroids to disk file to test them and use them as initial centroids

    f = open(temp_centroids_file, "w")


    centroids.sort()
    for c in centroids:
        f.write(c)
    f.close()


 
if __name__ == '__main__':

    args = sys.argv[1:]

    shutil.copy(input_centroids,temp_centroids_file)

    old_centroids = get_first_c(input_centroids)

 
    while True:



        mr_job=MRKMeans(args=[args[0],'--c',temp_centroids_file,'--k',args[1]])
 
        with mr_job.make_runner() as runner:
        

            runner.run()

           
            centroids = get_centroids(mr_job,runner)

        write_centroids(centroids)
        new_centroids = get_first_c(temp_centroids_file)

        max_d = diff(int(args[1]),new_centroids,old_centroids)
        #print "dist max = "+str(max_d)
        if max_d < 0.001:
            break
        else:
            old_centroids = new_centroids





    mr_job=MRFinalKMeans(args=[args[0],'--c',temp_centroids_file,'--k',args[1]])
 
    with mr_job.make_runner() as runner:

        runner.run()
       
        for key, value in mr_job.parse_output(runner.cat_output()):
            print(key+','+value)


