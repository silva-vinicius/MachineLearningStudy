#Hebbian Linaer Aproximation example 
#Book: Introdução a Ciência de Dados - Uma perspectiva de redes neurais artificias e reconhecimento de padrões

t <- seq(0, 5*pi, 0.5*pi) #creates a sequence of random numbers from 0 to 20Pi, using a step of 0.5Pi

#creates a labeled data sample 
x<-as.matrix(sin(t))
y<-as.matrix(-2*x) #function to be approximated: -2sin(t)


x<-cbind(x,1) #adding a column of 1's to the matrix x


#Normalizing
dK<-diag(x %*% t(x)) #generates the diagonal of  the K matrix
x[,1] <- x[,1]/dK
x[,2] <- x[,2]/dK


w<-t(x) %*% y 


#------------------------------------------------
#In order to obtain the approximation, we should multply H by w
yhat <- x %*% w #multiplies by the coefficients

#compare yhat to y
plot(t, y, xlab = "x", ylab = "f(x)", type="n")
lines(t, y, type = "o", col="black", pch=21, cex=0.6)
lines(t, yhat, type = "o", col="red", pch=21, cex=0.6)
#legend("top", col=c("black", "red"), legend = c("f(x)", "Aprox f(x)") )
legend("top", legend=c("f(x)", "Aprox f(x)"),
       col=c("black", "red"), lty=1:1, cex=0.8)