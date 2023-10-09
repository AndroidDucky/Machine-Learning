##### >>>>>> Jared Adams 865234


# Various tools for data manipulation. 



import numpy as np
import math

class MyUtils:

    
    def z_transform(X, degree):
        if(degree==1):
            return X
        n,d=X.shape

        B=np.arange(degree)
        #print("Size of B: "+ str(len(B)))
        for i in range(degree):
            B[i]=math.comb(i+d,d-1)
        dSum = np.sum(B)
        Z = X.copy()

        #for geting my L
        l = np.arange(dSum)
        #print(l)

        q=0
        p=d
        g=d
        for i in range(1,degree):
            #print("In I")
            for j in range(q,p):
                #print("In J")
                for k in range((l[j]),d):
                    temp=np.multiply(Z[:,j],X[:,k])
                    Z = np.append(Z, temp.reshape(-1,1),axis=1)
                    l[g]=k
                    g=g+1
            q=p
            p=p+B[i]
        #print("Final Z value: ")
        #print(Z)
        return Z
    
    
