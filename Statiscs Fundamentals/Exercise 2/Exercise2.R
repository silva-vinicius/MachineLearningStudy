set.seed(123)
data <- data.frame(matrix(rnorm(10000, mean=3), ncol=25, dimnames=list(NULL, paste("X", 1:25, sep="."))))

x = c()
for(i in 1:(dim(data)[2])){
  x[i] = paste("Var", i, sep="")
}
colnames(data) = x
print(colnames(data))