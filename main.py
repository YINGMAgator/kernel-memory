from tomita import trainlab,strain,testlab,stest,mask,maskv
import kaarma1
import KAARMA
import numpy as np

# print (len(trainp),len(trainn),len(testp),len(testn))
# print (trainn)
# l=min(len(trainp),len(trainn))
# Trainn=trainn[0:l]  # make the number of positive and negtive samples the same
# Traindata=trainp[0:l]+Trainn[0:l]
# Trainlabel = [[[1]]] * l+[[[0]]] * l
# print (Traindata[1])
# Traindata=trainp+trainn
# Trainlabel=trainlabelp+trainlabeln

# print(Trainlabel)
#
# Testdata=testp+testn
# Testlabel=testlabelp+testlabeln
# Testlabel=[[[1]]] * len(testp)+[[[0]]] * len(testn)


# print (Traindata[0][0,:])
# print (Traindata[0][0])
# KAARMA_TOMITA=kaarma1.KAARMA(Traindata,Trainlabel,Testdata,Testlabel)
# KAARMA_TOMITA.train()
# print (strain[0][0,:])
# print (strain[0][0])
# mask=[1,2,3]
print(stest)
print(testlab)
print(maskv)
# KAARMA_TOMITA=kaarma1.KAARMA(strain,trainlab,mask,stest,testlab,maskv)
# KAARMA_TOMITA.train()


# import tensorflow as tf


























# trainp = np.asarray(trainp)
# Trainp = np.expand_dims(trainp, axis=2)
#
# trainn = np.asarray(trainn)
# Trainn = np.expand_dims(trainn, axis=2)
#
# testn = np.asarray(testn)
# Testn = np.expand_dims(testn, axis=2)
#
# testp = np.asarray(testp)
# Testp = np.expand_dims(testp, axis=2)
# KAARMA_TOMITA=KAARMA.KAARMA(Trainp,Trainp,Trainn,Trainn)
# KAARMA_TOMITA.train()
