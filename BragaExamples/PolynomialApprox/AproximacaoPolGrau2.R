#Polinomial aproximation example 
#Book: Introdução a Ciência de Dados - Uma perspectiva de redes neurais artificias e reconhecimento de padrões

library(corpcor) #library that allows us to calculate pseudoinverses of matrixes

#----------------------------------------------------------------------
#Function that we will try to approximate: 1/2 (x^2) +3x + 10
fgx <- function(xin){
  0.5*(xin^2) + 3*xin + 10
}
#----------------------------------------------------------------------


#Generating the dataset containg the samples we will use to approximate the function
X <- runif(n=20, min=-15, max=10) #Generates 20 random points within the interval from -15 to 10 (uniform distribution)
Y <- fgx(X) + 10*rnorm(length(X)) #Generate the values by applying the function and adding some noise to them

#We will use a quadratic aprroximation
H <- cbind(X^2, X, 1) #Generates the matrix H |X^2  X  1|
w <- pseudoinverse(H) %*% Y #Determining the polynomial coefficients for the approximation function 

#Now we should compare the results obtained by both the original function and its approximation

#creating an interval in order to represent the x-axis, verying from -15 to 10 with a 0.1 step
xgrid<-seq(-15, 15, 1)

#Applying the original function to the interval
ygrid <- (0.5*(xgrid^2) + (3*xgrid) + 10 )

#In order to obtain the approximation, we should multply H by w
Hgrid <- cbind(xgrid^2, xgrid, 1) #applies x^2 + x + k
yhatgrid <- Hgrid %*% w #multiplies by the coefficients

#compare yhatgrid to ygrid
plot(xgrid, ygrid, xlab = "x", ylab = "f(x)", type="n")
lines(xgrid, ygrid, type = "o", col="black", pch=21, cex=0.6)
lines(xgrid, yhatgrid, type = "o", col="red", pch=21, cex=0.6)
#legend("top", col=c("black", "red"), legend = c("f(x)", "Aprox f(x)") )
legend("top", legend=c("f(x)", "Aprox f(x)"),
       col=c("black", "red"), lty=1:1, cex=0.8)
