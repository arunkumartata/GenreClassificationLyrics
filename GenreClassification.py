import re
import os
import collections
from os import path
import configparser
from SongsDataPreProcessing import SongsDataPreProcessing
from PerformDimensionReduction import PerformDimensionReduction
from scipy.sparse import csr_matrix
import sys
from PerformClustering import PerformClustering

if __name__ == "__main__":
    configsFile='configs/configs.cfg'
    if not path.exists(configsFile):
        print("configs.dat is missing!!")
    configs=configparser.ConfigParser()
    configs.read(configsFile)
    dataFile=configs.get('general','datafiletoreadfrom')
    dictFileName=configs.get('general','dictionaryfilename')
    wordCountFileName=configs.get('general','preprocesseddatfilename')
    colNumber=configs.get('general','dataFileColNumb')
    if len(colNumber)>0:
        colNumber=int(colNumber)
    else:
        colNumber=3
    songsPreProcess=SongsDataPreProcessing()
    _,dataMatrix=songsPreProcess.readData(dataFile,dictFileName,wordCountFileName,colNumber)
    # print(dataMatrix.shape,dataMatrix.nbytes)
    # print(bagOfWords.shape,dataMatrix.nbytes)
    del songsPreProcess
    # print(dataMatrix.shape,dataMatrix.nbytes)
    # print(bagOfWords.shape,bagOfWords.nbytes)
    # print(sys.getsizeof(csr_matrix(dataMatrix)))
    # dataMatrix=csr_matrix(dataMatrix)
    # print(songsPreProcess.bagOfWords)
    # print(songsPreProcess.dataMtrx.shape)
    # dataMatrix=songsPreProcess.dataMtrx
    # print(configs.has_option('general','isPCArequired'))
    # print(configs.get('general','isPCArequired'))
    if configs.has_option('general','isPCArequired') and configs.get('general','isPCArequired').lower()=='yes':
        reducedDimension=configs.get('general','requireddimensionsfrompca')
        reducedDimension=(1000,int(reducedDimension))[len(reducedDimension)>0]
        dataMatrix=PerformDimensionReduction().reduceDimensionUsingPCA(dataMatrix,reducedDimension)
        # dataMatrix=PerformPCA().reduceDimensionsUsingSVD(dataMatrix,reducedDimension)
    # print(dataMatrix)
    print(dataMatrix.shape)
    clusters=configs.get('general','numberofclusters')
    clusterfileName=configs.get('general','finalclusterfilename')
    if clusters:
        clusters=int(clusters)
    else:
        clusters=5
    PerformClustering().performKmeans(dataMatrix[:,0:5000],clusters,clusterfileName)

