import numpy as np
# from jqmcvi import base
class PerformClustering:
    def initializeCentroidsUsingKmeansplusplus(self,k):
        print('In kmeans++')
        dimensions=len(self.data)
        self.centroids=np.zeros((dimensions,k),order='F',dtype=np.float16)
        self.centroids[:,0]=self.data[:,np.random.randint(len(self.data[0]))]
        for nxtCentroid in range(k-1):
            dist=float('-inf')
            dataIndx=-1
            for i in range(len(self.data[0])):
                # print('centroid',nxtCentroid,'doc',i)
                d=float('inf')
                for j in range(nxtCentroid+1):
                    d=min(d,np.linalg.norm(self.centroids[:,j]-self.data[:,i]))
                if d>dist:
                    dist,dataIndx=d,i
            # if dataIndx!=-1:
            self.centroids[:,nxtCentroid+1]=self.data[:,dataIndx]

    def performKmeans(self,data,k,clusterfilename):
        self.data=data
        dimensions=len(self.data)
        print(self.data.shape)
        PerformClustering.initializeCentroidsUsingKmeansplusplus(self,k)
        docs=len(self.data[0])
        clusterAssign=[0]*docs
        tmpClusterAssign=[-1]*docs
        prevIntraCluster=float('inf')
        while True:
            tempIntraCluster=0
            ## Cluster assignment when cluster centroids are fixed
            for docId in range(docs):
                tmp=float('inf')
                fcentroidId=-1
                for centroidId in range(k):
                    val=np.linalg.norm(self.data[:,docId]-self.centroids[:,centroidId])
                    if val <tmp:
                        tmp=val
                        fcentroidId=centroidId
                # if fcentroidId!=-1:
                    # clusterAssign[docId]=fcentroidId+1
                tmpClusterAssign[docId]=fcentroidId
                tempIntraCluster+=tmp
            
            ## Cluster centroid update when clusters are fixed
            self.centroids=np.zeros((dimensions,k),order='F',dtype=np.float16)
            clusterCnt=[0]*k
            for docId in range(docs):
                # self.centroids[:,clusterAssign[docId]-1]+=self.data[:,docId]
                self.centroids[:,tmpClusterAssign[docId]]+=self.data[:,docId]
                # clusterCnt[clusterAssign[docId]-1]+=1
                clusterCnt[tmpClusterAssign[docId]]+=1
            for centroidId in range(k):
                self.centroids[:,centroidId]/=clusterCnt[centroidId]
            print(prevIntraCluster,tempIntraCluster)
            # if tmpClusterAssign==clusterAssign:
                # break
            if all(i==j for i,j in zip(tmpClusterAssign,clusterAssign)):
                break
            clusterAssign=tmpClusterAssign.copy()
            # if prevIntraCluster-tempIntraCluster <=0.0000001:
                # break
            prevIntraCluster=tempIntraCluster
        self.clusterAssign=clusterAssign
        
        print('calculating Dunn index')
        # dindx=base.dunn_fast(self.data.transpose(),[i-1 for i in clusterAssign])
        dindx=self.calculateDunnIndex(k)
        print(dindx)
        fdes=open(clusterfilename,"a+")
        for docId in range(docs):
            fdes.write("%d %d\n" % (docId+1,clusterAssign[docId]+1))
        fdes.close()

    ## calculates Dunn Index. Alternatives are DB (Davis-Bouldin) Index, Silhouette Index
    def calculateDunnIndex(self,k):
        minInterClusterDist=float('inf')
        maxIntraClusterDist=float('-inf')
        docs=len(self.data[0])
        ##calculate IntraCluster
        # for cluster in range(k):
        cdist=[0]*k
        cnt=[0]*k
        for docId in range(docs):
            cluster=self.clusterAssign[docId]
            cdist[cluster]+=np.linalg.norm(self.centroids[:,cluster]-self.data[:,docId])
            cnt[cluster]+=1
        for i in range(k):
            maxIntraClusterDist=max(maxIntraClusterDist,cdist[i]/cnt[i])
        print('maxIntraCluster',maxIntraClusterDist)
        ##calculateIntercluster
        for i in range(k):
            for j in range(i+1,k):
                val=np.linalg.norm(self.centroids[:,i]-self.centroids[:,j])
                minInterClusterDist=min(minInterClusterDist,val)
        print('minInterCluster',minInterClusterDist)
        return minInterClusterDist/maxIntraClusterDist
