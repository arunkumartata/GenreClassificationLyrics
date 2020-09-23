import numpy as np
from scipy.sparse import csr_matrix
import sys
class PerformDimensionReduction:
    # Using Orthogonal projection
    def reduceDimensionUsingPCA(self,data,dimensionToReduce):
        print("In PCA")
        originalDimensions=len(data)
        # print(originalDimensions)
        # numOfDataPoints=len(data[0])
        # sparsedmatrix=csr_matrix(data)
        # print(np.var(data,axis=0))
        data=csr_matrix(data)
        sparseMatrixTranspose=data.transpose()
        # sparseMatrixTranspose=data.transpose()
        theta=np.random.rand(originalDimensions,dimensionToReduce).astype(np.float16)
        # DocCluster.theta=np.random.normal(loc=0.0,scale=1.0,size=(DocCluster.mRow,k))
        cnt=1
        while True:
            # temp=sparseMatrixTranspose @ theta
            # temp=sparsedmatrix @ temp
            temp= data @ (sparseMatrixTranspose @ theta)
            tmp,_=np.linalg.qr(temp)
            # tmp,_=np.linalg.qr(sparsedmatrix @ (sparseMatrixTranspose @ theta))
            if np.linalg.norm(theta)-np.linalg.norm(tmp)<=0.00000000001:
                break
            theta=tmp
            print(cnt)
            cnt+=1
        reducedMatrix= np.transpose(theta) @ data
        # print(np.var(reducedMatrix,axis=0))
        return reducedMatrix.astype(np.float16)

    # data= U D V (SVD) and result will be U^T * data
    def reduceDimensionsUsingSVD(self,data,dimensionToReduce):
        print("In SVD PCA")
        data=data-data.mean(axis=0)
        # data=csr_matrix(data-data.mean(axis=0))
        data=csr_matrix(data)
        # print(sys.getsizeof(data))
        U,_,_=np.linalg.svd(data)
        return U[:,0:dimensionToReduce+1].transpose() @ data
