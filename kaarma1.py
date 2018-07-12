# KAARMA revised by Ying Ma
import numpy as np
import sys


# In[1]:
class KAARMA:
    def __init__(self,X,
                 Y,
                 mask,
                 Xv,
                 Yv,
                 maskv,
                 nx = 4,
                 center_max=10000,
                 As = 1/5.0,
                 Au = 1/100.0,
                 qThresh = .25,
                 nSkip = 1,
                 trackTrain = False,
                 trackValid = True):

        self.X = X # Training data

        self.Y = Y
        self.mask=mask
        self.Xv = Xv # Test data
        self.Yv = Yv
        self.maskv=maskv
        # print(self.mmmmm)
        # xShape = np.shape(X)
        self.nSeq = len(self.X) # Number of sequences in training
        self.nSeq_valid = len(self.Xv)
        self.order = 0 # Length of sequence
        self.nu = len(self.X[0][0]) # Input dimension

        self.nx = nx # Free state dimension
        self.ny = len(self.Y[0][0]) # Output dimension
        self.ns = self.nx+self.ny # State dimension

        self.center_max=center_max # max number of centers
        self.As = As # State kernel width
        self.Au = Au # Data kernel width
        self.qThresh = qThresh # Quantization threshold

        self.nSkip = nSkip
        self.trackTrain = trackTrain
        self.trackValid = trackValid

    def train(self, eta = .1,
              nIter = 1):

        # nAdd = int((self.order**2+self.order)/2) # Dictionary update per iteration
        # print(self.maskv[0][1],self.mmmmm)
        self.A = np.zeros([self.center_max,self.ns])
        self.S = np.zeros([self.center_max,self.ns])
        self.U = np.zeros([self.center_max,self.nu])

        # Set A(i-1), S(i-1),  U(i-1)
        rAmp = 0.1
        self.A[0,:] = rAmp*np.random.normal(0, 1, [1,self.ns])
        self.S[0,:] = rAmp*np.random.normal(0, 1, [1,self.ns])
        self.U[0,:] = rAmp*np.random.normal(0, 1, [1,self.nu])

        # Set fixed s0
        self.s0 = rAmp*np.random.normal(0, 1, [1,self.ns])
        sPrev = self.s0
        # print (self.s0 )
        # Create matrices for usage
        II = np.zeros([self.ny, self.ns])
        II[:,self.nx:] = np.identity(self.ny)
        nsEye = np.identity(self.ns)

        self.m = 1 # Initialize dictionary length

        nLoops = 0
        if self.trackTrain:
            self.trainLoss = np.zeros([int(self.nSeq*nIter/self.nSkip)+1])
            self.trainClass = np.zeros([int(self.nSeq*nIter/self.nSkip)+1])
            itError, itClass = self.evaluate(self.X, self.Y)
            self.trainLoss[nLoops] = itError
            self.trainClass[nLoops] = itClass

        if self.trackValid:
            self.validLoss = np.zeros([int(self.nSeq*nIter/self.nSkip)+1])
            self.validClass = np.zeros([int(self.nSeq*nIter/self.nSkip)+1])
            itError, itClass = self.evaluate(self.Xv, self.Yv)
            self.validLoss[nLoops] = itError
            self.validClass[nLoops] = itClass

        # Loop through nIter
        for it in range(nIter):
            # Randomly Scramble
            # rperm = np.random.permutation(self.nSeq)

            # Loop through data
            for permID in range(self.nSeq):
                seqID = permID
                sys.stdout.write("\rseqID: %i" % nLoops)
                sys.stdout.flush()
                nLoops += 1
                self.order=len(self.X[seqID])
                # print(self.maskv)
                # Loop through sequence
                for t in range(self.order):
                    # Preallocate A, S, U matrices
                    # print(self.mask[seqID][t])
                    if self.mask[seqID][t][0]==1:
                        # print ("yes")
                        S_ = np.zeros([t+1, self.ns])
                        U_ = np.zeros([t+1, self.nu])
                        qInd = np.zeros([t+1]).astype(int)

                        V = [np.hstack((np.reshape(nsEye[:,ni],(self.ns,1)),np.zeros([self.ns,t])))    for ni in range(self.ns)]
                        sprev = self.s0; # Set first state in sequence to fixed s0
                        for i in range(t+1):
                            # u = np.reshape(self.X[seqID,i,:],(1,self.nu))
                            u = np.reshape(self.X[seqID][i],(1,self.nu))
                            LAM, qIndSamp, s = self.cpuTrainKernel(sprev, u, i)

                            if i > 0:
                                qInd[i-1] = qIndSamp
                            if i == t:
                                qInd[t] = self.quantize(s = s, u = u)

                                # qInd[i] = self.quantize(s = s, u = u) for easy of the code

                            S_[i,:] = sprev
                            U_[i,:] = u
                            sprev = s
                            if i>0:
                                for ni in range(self.ns):
                                    V[ni][:,0:i] = np.dot(LAM,V[ni][:,0:i])
                                    V[ni][:,i] = nsEye[:,ni]

                        # Find error
                        d = np.reshape(self.Y[seqID][t], (1, self.ny))
                        e = self.costDerivative(d,s)
                        # print (s)

                        # Find new weights
                        addA = np.zeros([t+1,self.ns])
                        for ni in range(self.ns):
                            addACol = eta*np.dot(e,np.dot(II,V[ni])).T
                            addA[:,ni] = np.squeeze(addACol)

                        # Quantize Vector
                        for aID in range(t+1):
                            if qInd[aID]>=0:
                                self.A[qInd[aID],:] += addA[aID,:]

                            else:
                                self.A[self.m,:] = addA[aID,:]
                                self.S[self.m,:] = S_[aID,:]
                                self.U[self.m,:] = U_[aID,:]
                                self.m += 1

                if self.trackTrain and (nLoops)%self.nSkip==0:
                    itError, itClass = self.evaluate(self.X, self.Y)
                    self.trainLoss[int((nLoops-1)/self.nSkip)] = itError
                    self.trainClass[int((nLoops-1)/self.nSkip)] = itClass
                if self.trackValid and (nLoops)%self.nSkip==0:
                    itError, itClass = self.evaluate(self.Xv, self.Yv)
                    print("It: ", it, ", Error: ", itError, ", Class: ", itClass)
                    self.validLoss[int((nLoops-1)/self.nSkip)] = itError
                    self.validClass[int((nLoops-1)/self.nSkip)] = itClass

        sys.stdout.write("\nCenters Created: %i" % self.m)

    def costDerivative(self, d, s):
        e = np.subtract(d,s[0,self.nx:])
        return e

    def cpuTrainKernel(self, s, u, it):
        Ds = np.subtract(self.S[0:self.m,:],s)
        Du = np.subtract(self.U[0:self.m,:],u)

        qVals = np.add(self.As*np.sum(Ds**2,axis=-1),self.Au*np.sum(Du**2,axis=-1))
        Ksu = np.exp(-1*qVals)

        AbyK = self.A[0:self.m,:].T*Ksu
        snext = np.reshape(np.sum(AbyK,axis=1),(1,self.ns))
        LAM = 2.0*self.As*np.dot(AbyK,Ds)

        minInd = -1
        if it>0:
            minInd = self.quantize(qVals = qVals)

        return LAM, minInd, snext

    def quantize(self, qVals = None, s = None, u = None):
        if qVals is  None:
            Ds = np.subtract(self.S[0:self.m,:],s)
            Du = np.subtract(self.U[0:self.m,:],u)
            qVals = np.add(self.As*np.sum(Ds**2,axis=-1),self.Au*np.sum(Du**2,axis=-1))

        minInd = np.argmin(qVals)
        if qVals[minInd]>self.qThresh:
            minInd = -1

        return minInd

    def evaluate(self, X, Y):
        e = 0.0
        nCorrect = 0.0
        bcorrect = 0.0
        sprev = self.s0
        nSeq = len(X)
        for seqID in range(nSeq):
            self.order=len(X[seqID])
            seqpre=[]
            for i in range(self.order):
                d = np.reshape(Y[seqID][i,:], (1, self.ny))
                u = np.reshape(X[seqID][i,:],(1,self.nu))
                s = self.cpuTestKernel(sprev, u)
                sprev = s
                seqpre.append(s)

            e += np.sum(np.subtract(d,s[0,self.nx:])**2)*(1.0/self.ny)

            if np.argmax(s[0,self.nx:]) == np.argmax(d):
                nCorrect += 1.0
        print (X[seqID],Y[seqID],seqpre)
        return e/nSeq, nCorrect/nSeq

    def getEvalTrain(self):
        return self.trainLoss, self.trainClass
    def getEvalValid(self):
        return self.validLoss, self.validClass

    def cpuTestKernel(self, s, u):
        Ds = np.subtract(self.S[0:self.m,:],s)
        Du = np.subtract(self.U[0:self.m,:],u)

        qVals = np.add(self.As*np.sum(Ds**2,axis=-1),self.Au*np.sum(Du**2,axis=-1))

        Ksu = np.exp(-1*qVals)
        AbyK = self.A[0:self.m,:].T*Ksu
        snext = np.reshape(np.sum(AbyK,axis=1),(1,self.ns))

        return snext

    def cpuTestSequence(self, U, D):
        e_tot = 0.0
        # pred = np.zeros((np.shape(D)[0],self.order,self.ns))
        pred =[]
        sprev = self.s0
        L=0
        for seqID in range(len(U)):
            self.order=len(U[0])
            pred_oneseuqnce=np.zeros(self.order,self.ns)
            for i in range(self.order):
                d = np.reshape(D[seqID][i], (1, self.ny))
                u = np.reshape(U[seqID][i],(1,self.nu))
                s = self.cpuTestKernel(sprev, u)
                sprev = s
                pred_oneseuqnce[i,:] = s[0]

                e_tot += np.sum(np.subtract(d,s[0,self.nx:])**2)*(1.0/self.ny)
            pred.append(pred_oneseuqnce)
            L=L+self.order
        e_tot = e_tot/L

        return pred, e_tot

    def getA(self):
        return self.A


# In[ ]:
