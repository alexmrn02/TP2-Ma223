import numpy as np
import time
import math as ma
import matplotlib.pyplot as plt


def ResolutionSystTriginf(L,B):

    Taug=np.c_[L,B]
    n,m = Taug.shape
    if m!=n+1:
        print('pas une matrice augmentée')
        return
    x=np.zeros(n)
    for i in range(n):
        somme=0
        for K in range(n):
            somme=somme+x[K]*Taug[i,K]
        x[i]=(Taug[i,-1]-somme)/Taug[i,i]
    return x


def ResolutionSystTriSup(U,Y):

    Taug = np.c_[U,Y]
    n,m = Taug.shape
    if m!=n+1:
        return('pas une matrice augmentée')
    x=np.zeros(n)
    for i in range(n-1,-1,-1):
        somme=0
        for k in range(i+1,n):
            somme=somme+x[k]*Taug[i,k]
        x[i]=(Taug[i,-1]-somme)/Taug[i,i]
    return x

def Cholesky(A):
    s=0
    nl, nc = np.shape(A)
    L = np.zeros([nl, nl])
    try:
        for k in range(0, nc):
            s = 0
            for j in range(0, k):
                s += L[k,j]**2
            L[k,k] = ma.sqrt(A[k,k] - s)
            for i in range (k+1,nl):
                s = 0        
                for l in range(0, k):
                    s += L[i,l]*L[k,l]
                L[i,k] = (A[i,k] - s)/L[k,k]
        verif = 1             
        L_T = np.transpose(L)
        return L, L_T, verif
    except:
        L_T = np.transpose(L)
        verif = 0
        return L, L_T, verif


X = [] 

def ResolCholesky(A,B):
    t1 = time.time()
    nl, nc = np.shape(A)

    L , LT, verif = Cholesky(A)
    
    if verif == 0:
        print('Cholesky impossible.')
        t2=time.time()
        tps_calcul = t2-t1
        IndicesChol.append(tps_calcul) 
    else:
        Y = ResolutionSystTriginf(L,B)
        X = ResolutionSystTriSup(LT,Y)
        erreur = np.linalg.norm(A.dot(X) - np.ravel(B))
        ErreurChol.append(erreur)
        t2=time.time()
        tps_calcul = t2-t1
        IndicesChol.append(tps_calcul)
    return X




ErreurChol = []
IndicesChol = []
IndicesLU=[]
ErreurLU=[]
IndicesLinalg = []
ErreurLinalg=[]
IndicesLinalgChol =[]
ErreurLinalgChol = []

def DecompositionLU(A):

    n,m = A.shape 
    L = np.eye(n)
    U = np.copy(A)
    for i in range(0,n-1):
        for k in range(i+1,n):
            pivot = U[k,i]/U[i,i]
            L[k,i]= pivot
            for j in range(i,n):
                U[k,j]=U[k,j]-pivot*U[i,j]
    return L,U

def resolution_LU(A,B):

    t1 = time.time()
    n,m = np.shape(A)
    X=np.zeros(n)
    L,U=DecompositionLU(A)
    Y = ResolutionSystTriginf(L,B)
    X = ResolutionSystTriSup(U,Y)
    X = np.asarray(X)
    erreur = np.linalg.norm(A.dot(X) - np.ravel(B))
    print("LU:" ,erreur)
    ErreurLU.append(erreur)
    t2 = time.time()
    tps_calcul = t2-t1
    IndicesLU.append(tps_calcul)
    return X

def linalg_solve(A,B):
 
    t1=time.time()
    X = np.linalg.solve(A,B)
    t2=time.time()
    tps_calcul = t2-t1 
    IndicesLinalg.append(tps_calcul)
    erreur = np.linalg.norm(A.dot(X) - np.ravel(B))
    print("Linalg:", erreur)
    ErreurLinalg.append(erreur)
    return tps_calcul

def linalg_cholesky(A,B):

    t1=time.time()
    L  = np.linalg.cholesky(A)
    LT = np.transpose(L)
    Y = ResolutionSystTriginf(L,B)
    X = ResolutionSystTriSup(LT,Y)
    erreur = np.linalg.norm(A.dot(X) - np.ravel(B))
    print(erreur)
    t2=time.time()
    tps_calcul = t2-t1 
    IndicesLinalgChol.append(tps_calcul)
    ErreurLinalgChol.append(erreur)
    return tps_calcul



for n in range(50, 500, 50):
    X.append(n)
    A = np.random.rand(n, n)
    B = np.random.rand(n)
    A = A.dot(np.transpose(A))
    resolution_LU(A,B)
    linalg_solve(A,B)
    ResolCholesky(A, B)
    linalg_cholesky(A, B)



plt.ylabel("temps(seconde)")
plt.xlabel("dimension")
plt.title("Temps en fonction de la dimension de la matrice")
plt.plot(X,IndicesLU, color = 'green', label='LU')
plt.plot(X,IndicesLinalg, color = 'blue', label='Linalg.solve')
plt.plot(X,IndicesChol, color = 'orange', label='Cholesky')
plt.plot(X,IndicesLinalgChol, color = 'red', label='Linalg.Cholesky')
plt.legend()
plt.show()


plt.ylabel("normes(erreurs)")
plt.xlabel("dimension")
plt.title("Erreur en fonction de la dimension de la matrice")
plt.plot(X,ErreurLU, color = 'green', label='LU')
plt.plot(X,ErreurLinalg, color = 'blue', label='Linalg.solve')
plt.plot(X,ErreurChol, color = 'orange', label='Cholesky')
plt.plot(X,ErreurLinalgChol, color = 'red', label='Linalg.Cholesky')
plt.legend()
plt.show()
