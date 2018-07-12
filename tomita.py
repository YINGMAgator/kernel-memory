# tomita grammar generator
import numpy as np
strain=[]
trainlab=[]
mask=[]
for i in range (10):
    l=np.random.randint(8)+3
    l=10
    sappend=np.random.randint(2, size=l)
    strain.append(np.expand_dims(sappend, axis=1))

    # maskappend=np.zeros((l,1),dtype=int)
    # maskappend[l-1]=1
    # mask.append(maskappend)
    # aaa=np.where((sappend[:-2] == 0) & (np.roll(sappend,-1)[:-2] == 0)& (np.roll(sappend,-2) [:-2]== 0))[0]
    # print(sappend,aaa)
    # if len(aaa) == 0:
    #     trainlab.append(np.zeros((l,1),dtype=int))
    # else:
    #     trainlab.append(np.ones((l,1),dtype=int))
    maskappend=np.ones((l,1),dtype=int)
    mask.append(maskappend)

    aaa=np.where((sappend[:-2] == 0) & (np.roll(sappend,-1)[:-2] == 0)& (np.roll(sappend,-2) [:-2]== 0))[0]
    if len(aaa) == 0:
        trainlab.append(np.zeros((l,1),dtype=int))
    else:
        la=np.zeros((l,1),dtype=int)
        laindex = aaa[0]+2
        la[int(laindex):]=1
        trainlab.append(la)
# print(strain,trainlab)

# trainp = [a1 for a1, c1 in zip(strain,trainlab) if c1[0] == 1]  # seperate positive and negtive training samples
# trainn = [a1 for a1, c1 in zip(strain,trainlab) if c1[0] == 0]
# trainlabelp = [c1 for c1 in trainlab if c1[0] == 1]
# trainlabeln = [c1 for c1 in trainlab if c1[0] == 0]


stest=[]
testlab=[]
maskv=[]
for i in range (10):
    l=np.random.randint(6)+10
    # l=10
    sappend=np.random.randint(2, size=l)
    stest.append(np.expand_dims(sappend, axis=1))

    maskappend=np.ones((l,1),dtype=int)
    maskv.append(maskappend)

    aaa=np.where((sappend[:-2] == 0) & (np.roll(sappend,-1)[:-2] == 0)& (np.roll(sappend,-2) [:-2]== 0))[0]
    if len(aaa) == 0:
        testlab.append(np.zeros((l,1),dtype=int))
    else:
        la=np.zeros((l,1),dtype=int)
        laindex = aaa[0]+2
        la[int(laindex):]=1
        testlab.append(la)
    # maskvappend=np.zeros((l,1),dtype=int)
    # maskvappend[l-1]=1
    # maskv.append(maskvappend)
    # aaa=np.where((sappend[:-2] == 0) & (np.roll(sappend,-1)[:-2] == 0)& (np.roll(sappend,-2) [:-2]== 0))[0]
    # if len(aaa) == 0:
    #     testlab.append(np.zeros((l,1),dtype=int))
    # else:
    #     testlab.append(np.ones((l,1),dtype=int))

# testp = [a1 for a1, c1 in zip(stest,testlab) if c1[0] == 1]
# testn = [a1 for a1, c1 in zip(stest,testlab) if c1[0] == 0]
# testlabelp = [c1 for c1 in testlab if c1[0] == 1]
# testlabeln = [c1 for c1 in testlab if c1[0] == 0]
