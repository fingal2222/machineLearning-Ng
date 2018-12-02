import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt
import  matplotlib.pyplot as plt
import  math
from numpy.linalg import *

def loadData(filePath):
    return loadmat(filePath)

def ex8_cofi(Y,R):
    print(np.mean(Y[0,R[0,:]]))

# Collaborative Filtering
    data2=loadmat("/Users/zhaofengjun/Documents/python/machineLearning-Ng/ex8/ex8_movieParams.mat")
#     reduce the data set size so that this
    X=data2["X"]
    Theta=data2["Theta"]
    num_users=4
    num_movies=5
    num_features=3
    X=X[0:num_movies,0:num_features]
    Theta=Theta[0:num_users,0:num_features]
    Y=Y[0:num_movies,0:num_users]
    R=R[0:num_movies,0:num_users]
    params=np.r_[np.ravel(X),np.ravel(Theta)]
    J,grad=cofiCostFunc(params,Y,R,num_users,num_movies,num_features,0)
    print("cost at loaded parameters: ")
    print(J)

    #checkgradient

#     Collaborative Filtering Cost Regularization
    params=np.r_[np.ravel(X),np.ravel(Theta)]
    J,grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)
    print("J at lam=1.5 is:  ")
    print(J)

    n=1682
    movieList=loadMovieList()
    my_rating=np.zeros((n,1))
    my_rating[0,:]=4
    my_rating[97,:]=2
    my_rating[6,:]=3
    my_rating[11,:]=5
    my_rating[53,:]=4
    my_rating[63,:]=5
    my_rating[65,:]=3
    my_rating[68,:]=5
    my_rating[182,:]=4
    my_rating[225,:]=5
    my_rating[354,:]=5
    for i in range(n):
        if(my_rating[i,:]>0):
            print(my_rating[i,:],movieList[i])
            print("\n")

    #Learning Movie Ratings
    dt=loadmat("/Users/zhaofengjun/Documents/python/machineLearning-Ng/ex8/ex8_movies.mat")
    Y=np.c_[my_rating,dt["Y"]]
    R=np.c_[(my_rating!=0)+0,dt["R"]]
    Ymean,Ynorm=normalizeRatings(Y,R)

    num_movies,num_users=np.shape(Y)
    num_features=10
    X=np.random.randint(0,5,size=(num_movies,num_features))
    Theta=np.random.rand(num_users,num_features)
    initial_nn_params=np.array(np.r_[X.ravel(),Theta.ravel()])
    costFunc=costFunction(Y,R,num_users,num_movies,num_features,0)
    [params, cost] = opt.fmin_cg(costFunc, x0=initial_nn_params)
    X = np.reshape(params[0:num_movies * num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_features * num_movies:], (num_users, num_features))

    #Recommendation for you
    predictions=np.dot(X,Theta.T)
    my_pred=predictions[:,0]+Ymean
    indx= np.argsort(my_pred)
    lastIndex=len(indx)-1
    for i in range(10):
        print("movieName:",movieList[lastIndex-i])
        print("rating is:",my_pred[lastIndex-i,0])
        print("\n")

    for i in range(n):
        if(my_rating[i,:]>0):
            print(my_rating[i,:],movieList[i])
            print("\n")





def costFunction(Y,R,num_users,num_movies,num_features,lam):
    def nnCostFunction2(nn_params):
        return cofiCostFunc(nn_params,Y,R,num_users,num_movies,num_features,lam)
    return nnCostFunction2

def normalizeRatings(Y,R):
    m,n=np.shape(Y)
    Ymean=np.zeros((m,1))
    Ynorm=np.zeros(np.shape(Y))
    for i in range(m):
        idx=np.where(R[i,:]==1)
        Ymean[i,:]=np.mean(Y[i,idx])
        Ynorm[i,idx]=Y[i,idx]-Ymean[i,:]
    return Ymean,Ynorm






def loadMovieList():
    movieList = []
    filename = "/Users/zhaofengjun/Documents/python/machineLearning-Ng/ex8/movie_ids.txt"
    for data in open(filename):
        movieList.append(data[data.find(" ") + 1:])

    return np.array(movieList,dtype=str)


    #checkGradient(1.5)


def cofiCostFunc(params,Y,R,num_users,num_movies,num_features,lam):

    J=0
    X=np.reshape(params[0:num_movies*num_features],(num_movies,num_features))
    Theta=np.reshape(params[num_features*num_movies:],(num_users,num_features))
    X_grad=np.zeros((np.shape(X)))
    Theta_grad=np.zeros(np.shape(Theta))
    J_temp=(np.dot(X,Theta.T)-Y)**2
    J=np.sum(J_temp[R==1])/2+lam/2*(np.sum(Theta**2)+np.sum(X**2))
    X_grad=np.dot(((np.dot(X,Theta.T)-Y)*R),Theta)+lam*X
    Theta_grad=np.dot(((np.dot(X,Theta.T)-Y)*R).T,X)+lam*Theta
    grad=np.r_[X_grad.ravel(),Theta_grad.ravel()]
    return J,np.array(grad)

if __name__=='__main__':
    data=loadData("/Users/zhaofengjun/Documents/python/machineLearning-Ng/ex8/ex8_movies.mat")
    Y=data["Y"]
    R=data["R"]
    ex8_cofi(Y,R)
    loadMovieList()

