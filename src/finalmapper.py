import re
import numpy as np
import mrjob
from mrjob.job import MRJob
import math



class MRFinalKMeans(MRJob):

    SORT_VALUES = True
    OUTPUT_PROTOCOL = mrjob.protocol.RawProtocol
    def distance_vec(self,d,c):
        #calculate the ditance between two vectors (in two dimensions)
        #np.linalg.norm(data_point - centroid)
        diff=[d1 - c2 for d1,c2 in zip(d,c)]
        return np.linalg.norm(diff)
 
    def configure_args(self):
        super(MRFinalKMeans, self).configure_args()
        #add the number of cluster after the k argument
        self.add_passthru_arg("--k",  default=4, help="Number of clusters.")

        #add the centoird file after the c argument
        self.add_file_arg('--c')

    def file_to_centroids(self):
    #gzt the centoids from the file (self.option.c) and transform them to a list
        f = open(self.options.c,'r')
        output_data=f.read().split('\n')
        centroids=[]
        for line in output_data:
            if line:
                newline = re.search("\[(.*?)\]", line)
                newline = newline.group()
                newline = newline.replace("[", "").replace("]", "")
                newline.strip()
                k,x,y = newline.split(',')
                centroids.append([float(k),float(x),float(y)])
        f.close()
        return centroids
 
    def mapper(self, _, lines):
       
        centroids = self.file_to_centroids()
        for l in lines.split('\n'):
            x,y = l.split(' ')
            p = [float(x),float(y)]
            min_dist=math.inf
            classe = 0
            # we iterate over the centroids , we have k centroids ( we get the number of centroids from the k argument in command line)
            for i in range(int(self.options.k)):
                dist = self.distance_vec(p,centroids[i][1:])
            #dist = [distance_vec(p - c) for c in centroids]
            #classe = np.argmin(dist)
                if dist < min_dist:
                    min_dist = dist
                    classe = centroids[i][0]
            
        yield str(classe), str(p)
 


 
if __name__ == '__main__':

 #just run mapreduce !
    import sys
 #just run mapreduce !
    args = sys.argv[1:]
    mr_job=MRFinalKMeans(args=[args[0],'--c',args[1],'--k',args[2]])
 
    with mr_job.make_runner() as runner:

        runner.run()
       
        for key, value in mr_job.parse_output(runner.cat_output()):
            print(key+','+value)





