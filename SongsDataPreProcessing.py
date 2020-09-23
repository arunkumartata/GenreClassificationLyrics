import numpy as np
from nltk.corpus import stopwords
import csv
import re
import os
import string
import collections
from os import path
class SongsDataPreProcessing:
    def constructWordDictionary(self,fileName,dataFile,colNumb):
        if len(fileName)<=0:
            print("Invalid filename given!!")
            return
        if len(dataFile)<=0:
            print("Invalid datafile given to read the data from!!")
            return
        if not path.exists(dataFile):
            print("Given dataFile doesn't exists!!")
            return
        if not path.exists(fileName):
            print("In if dict")
            dictinry=set()
            stopWords=set(stopwords.words('english'))
            flst=[]
            punctuation='?\";,:.!()-#@&?'
            translationTable=str.maketrans('','',punctuation)
            with open(dataFile,'r') as csvfile:
                data=csv.reader(csvfile)
                # indx=0
                for row in data:
                    tlst=[]
                    # print("dataPoint: ",indx)
                    if(row[colNumb]=='text'):
                        continue
                    for strval in re.split('\s+',row[colNumb]):
                        strval=strval.translate(translationTable)
                        strval=strval.strip().lower()
                        strval=SongsDataPreProcessing.removeSingleQuotes(self,strval)
                        if len(strval)>0 and ('[' not in strval) and (']' not in strval) and (not re.search('\d+',strval)) and strval not in stopWords:
                            if strval not in dictinry:
                                dictinry.add(strval)
                            # tlst.append(strval)
                    flst.append(tlst[:])
            bagOfWordsFileDes=open(fileName,"a+")
            ndict=sorted(dictinry)
            self.bagOfWords=ndict[:]
            np.savetxt(bagOfWordsFileDes,ndict,fmt='%s')
            bagOfWordsFileDes.close()
        else:
            # print("In else dict")
            self.bagOfWords=np.loadtxt(fileName,dtype=np.dtype(str))
    
    def constructWordCountPerSong(self,fileName,dataFile):
        if len(fileName)<=0 or len(dataFile)<=0:
            print("Invalid fileName or invalid dataFile given!!")
        if not path.exists(dataFile):
            print("Given dataFile doesn't exists!!")
        if not path.exists(fileName):
            # print("In if wordcount")
            stopWords=set(stopwords.words('english'))
            self.flst=[]
            punctuation='?\";,:.!()-#@&?'
            translationTable=str.maketrans('','',punctuation)
            with open(dataFile,'r') as csvfile:
                data=csv.reader(csvfile)
                # indx=0
                for row in data:
                    tlst=[]
                    # print("dataPoint: ",indx)
                    if(row[3]=='text'):
                        continue
                    for strval in re.split('\s+',row[3]):
                        strval=strval.translate(translationTable)
                        strval=strval.strip().lower()
                        strval=SongsDataPreProcessing.removeSingleQuotes(self,strval)
                        if len(strval)>0 and ('[' not in strval) and (']' not in strval) and (not re.search('\d+',strval)) and strval not in stopWords:
                            tlst.append(strval)
                    self.flst.append(tlst[:])
            rows=len(self.bagOfWords)
            cols=len(self.flst)
            # print(rows,cols)
            self.dataMtrx=np.zeros((np.int(rows),np.int(cols)),dtype=np.dtype('i2'))
            songsDataFileDes=open(fileName,"a+")
            for i in range(cols):
                # print("Prossessing: ",i)
                cur=self.flst[i]
                curDct=collections.Counter(cur)
                for j in range(rows):
                    if self.bagOfWords[j] in curDct:
                        self.dataMtrx[j][i]=curDct[self.bagOfWords[j]]
                        songsDataFileDes.write("%d %d %d\n" % (i+1, j+1 ,self.dataMtrx[j][i]))
            songsDataFileDes.close()
        else:
            # print("in else wordcount")
            fileData=np.loadtxt(fileName,dtype ={'names': ('songIdx', 'wordIdx', 'count'),'formats': ('i4', 'i4', 'i4')})
            numDocs,numWords=0,0
            for songId,wordId,_ in fileData:
                numDocs=max(numDocs,songId)
                numWords=max(numWords,wordId)
            self.dataMtrx=np.zeros((np.int(numWords),np.int(numDocs)),dtype=np.dtype('i2'))
            for docId,wordId,cnt in fileData:
                self.dataMtrx[wordId-1][docId-1]=cnt



    def readData(self,dataFile,dictFile,wordCountFile,colNum):
        # dataFile='songdata.csv'
        # dictFile='songsdata_dict.txt'
        # wordCountFile='songsdata_data.txt'
        SongsDataPreProcessing.constructWordDictionary(self,dictFile,dataFile,colNum)
        SongsDataPreProcessing.constructWordCountPerSong(self,wordCountFile,dataFile)
        return self.bagOfWords,self.dataMtrx
        # print(self.bagOfWords)
        # print(self.dataMtrx.shape)


    def removeSingleQuotes(self,removeFrom: str):
        # print("Received string: ", removeFrom)
        if  len(removeFrom)<=0 and '\'' not in removeFrom:
            return removeFrom
        while len(removeFrom)>0 and removeFrom[-1]=='\'':
            removeFrom=removeFrom[:len(removeFrom)-1]
            # print("After removal: ",removeFrom )
        while len(removeFrom)>0 and removeFrom[0]=='\'':
            removeFrom=removeFrom[1:]
        
        return removeFrom


# SongsDataPreProcessing().readData()