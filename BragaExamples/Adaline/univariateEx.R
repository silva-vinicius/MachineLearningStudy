#Adaline Example 
#Book: Introdução a Ciência de Dados - Uma perspectiva de redes neurais artificias e reconhecimento de padrões



#xin -> matrix containing the data (without the labels)
#yd -> labels of the data
#eta -> training step (η)
#tol -> tolerable error
#maxepocas -> maximum number of training iterations (number of times it will iterate over the complete xin)
#par -> indicates whether or not the xin matrix contains an artificailly introduced column of 1s. 
#           1 means that we should include it ourselves and 0 that it's already there
trainAdaline <- function(xin, yd, eta, tol, maxepocas, par){
  
  dimXin<-dim(xin)
  N <- dimXin[1] # number of X lines 
  n <- dimXin[2] # number of X columns
  
  #initializing the weights matrix. Initially they will contain random numbers (runif -> uniform distribution)
  if(par==1){
    wt <- as.matrix(runif(n+1)-0.5)
    xin<-cbind(1, xin)
  }else{
    wt <- as.matrix(runif(n)-0.5)
  }
  
  nepocas <- 0
  eepoca <- tol+1 #eepoca is the error found in one epoch
  
  #we will store the error of every epoch in this variable evec
  evec <- matrix(nrow=1, ncol=maxepocas)
  
  #training each epoch
  while((nepocas < maxepocas) && (eepoca>tol)){
    
    #squared error
    ei2 <- 0
    xseq <- sample(N)
    
    #iterating over the dataset
    for(i in 1:N){
      
      irand <- xseq[i]
      yhati <- 1.0*((xin[irand,] %*% wt))
      ei <- yd[irand] - yhati
      
      #creating the delta w vector
      dw <- eta*ei*xin[irand,]
      
      #adjusting the weights for the next iterations
      wt <- wt+dw
      
      ei2 <- ei2 + ei*ei
    }
    
    nepocas <- nepocas+1
    evec[nepocas] <- ei2/N
    
    eepoca <- evec[nepocas]
    
  }
  
  #returning a list containing the weights and the error on each epoch 
  retlist <- list(wt, evec[1:nepocas])
  
  return (retlist)
}

approximate <- function(value,trainedModel){
  
  #Fixed: Had forgotten to use cbind()
  return ( cbind(1,value) %*% trainedModel)
}


#Testing the Adaline function

#we will try to approximate the function f(x) = 4x+2,
#where x = sin(t)


#---------collecting test data-----------
t<-matrix(seq(0, 2*pi, 0.1*pi), ncol=1)
x<-sin(t)
y<-matrix((4*x)+2, ncol=1)
#----------------------------------------

trainedModel <-trainAdaline(x,y, 0.01, 0.01, 50, 1)
w <- trainedModel[[1]]
error<-trainedModel[[2]]

plot(error, type="l", xlab="Epoch", ylab="Error")

print(w)

#yhat <- numeric(length=length(x))


yhat <- as.numeric(lapply(x, approximate, trainedModel=w))


#compare yhat to y
plot(t, y, xlab = "x", ylab = "f(x)", type="n")
lines(t, y, type = "o", col="black", pch=21, cex=0.6)
lines(t, t(yhat), type = "o", col="red", pch=21, cex=0.6)
legend("top", legend=c("f(x)", "Approx f(x)"),
       col=c("black", "red"), lty=1:1, cex=0.8)





