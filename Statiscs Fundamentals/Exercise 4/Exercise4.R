#Lista 04 - Fundamentos Estatísticos para Ciência dos Dados
#Vinícius de Oliveira Silva - Matrícula:2013007820

#Questão 1

#A)
x <- 0:10
px <- dbinom(x, 20, 0.15) #vetor com as probabilidades de X=0, 1, 2, ... 10
par(mfrow=c(1,2)) # janela grafica com uma linha de 2 plots
plot(x, px, type = "h") # para usar linhas verticais até os pontos (x,px)
Fx <- pbinom(x, 20, 0.15) #vetor com as probabilidades acumuladas P(X<0), P(X<1), ... P(X<10)
plot(x, Fx, type = "s") # o argumento "s"


#B)

#O valor de x onde a probabilidade é máxima é 3. O valor dessa probabilidade é 0.2428

#C)
#Uma faixa onde P(X) seja próxima de 1 é [0 6]

#D)
#nƟ = 3 -> Ao redor de 3, a função tem o seu máximo

#E)
#0.01 foi subtraído para que o intervalo seja fechado em 5.
#Neste caso, devemos subtrair 0.01 de 0 para que o intervalo seja fechado em 0
print(pbinom(6, 20, 0.15) - pbinom(0-0.01, 20, 0.15))

#F)
print(qbinom(0.95, 20, 0.15))

#G)
print(pbinom(6, 20, 0.15))

#H)
randVals <- rbinom(1000, 20, 0.15)
foundVals <- randVals[ randVals >= 0 & randVals <= 6]
percentage <- length(foundVals)/length(randVals)
cat("A porcentagem de valores que cairam na faixa escolhida foi", percentage*100, "%")

#I)
#?
  
#Questão 2

#lambda = 0.73

#A)
x <- 0:10
px <- dpois(x, 0.73)
par(mfrow=c(1,2)) # janela grafica com uma linha de 2 plots
plot(x, px, type = "h") 
Fx <- ppois(x, 0.73)
plot(x, Fx, type = "s") # o argumento "s"

#B)
# Determinando k onde P(X=k) é maximo:
cat("O valor e maximo quando k=", x[which(px == max(px))], "\n")
# P(X=k) é maximo quando k=0, por isso devemos calcular P(X=0):
cat("P(X=0) = ",px[1], " que é relativamente proximo ao lambda = 0.73")

#C)
#Um intervalo razoavel é (0, 3)

#D)
cat("P(0 <= X <= 3) = ", ppois(3, 0.73) - ppois(0-0.01, 0.73))

#E)
randPoisVals <- rpois(200, 0.73)

#F)

for (i in 0:6) {
  #P(X=i)
  cat("A probabilidade P(X=",i,") = ", px[i+1], " -- A frequencia relativa é: ", (sum(randPoisVals == i) / length(randPoisVals) ) * 100, "% \n", sep = "")
}

#Questão 2

#lambda = 10

#A)
x <- 0:20
px <- dpois(x, 10)
par(mfrow=c(1,2)) # janela grafica com uma linha de 2 plots
plot(x, px, type = "h") # para usar linhas verticais ate os pontos (x,px)
Fx <- ppois(x, 10)
plot(x, Fx, type = "s") # o argumento "s"

#B)
# Determinando k onde P(X=k) é maximo:
cat("O valor e maximo quando k=", x[which(px == max(px))], "\n")
# P(X=k) é maximo quando k=9 ou k=10, calculemos P(X=9):
cat("P(X=9) = ",px[10], " não é nada proximo ao lambda = 10")

#C)
#Observando o gráfico, um intervalo razoavel é (3,17)

#D)
cat("P(3 <= X <= 17) = ", ppois(17, 10) - ppois(3-0.01, 10))

#E)
randPoisVals <- rpois(200, 10)

#F)

for (i in 0:6) {

  cat("A probabilidade P(X=",i,") = ", px[i+1], " -- A frequencia relativa é: ", (sum(randPoisVals == i) / length(randPoisVals) ) * 100, "% \n", sep = "")
}



#Questão 3

#A)
x <- 1:20

#alpha = 0.5
pxAlpha0 = numeric(20)

for(i in x){
  pxAlpha0[i] = (1/2.612) / (i ^ (1+ 0.5))
}

# alpha = 1
pxAlpha1 = numeric(20)
for(i in x){
  pxAlpha1[i] = (1/1.645) / (i ^ (1+ 1))
}

#alpha = 2
pxAlpha2 = numeric(20)
for(i in x){
  pxAlpha2[i] = (1/1.202) / (i ^ (1+ 2))
}

#Desenhando os graficos:
par(mfrow=c(1,3)) 
plot(x, pxAlpha0, type = "h") 
plot(x, pxAlpha1, type = "h") 
plot(x, pxAlpha2, type = "h")

#Obtendo a soma cumulativa
cumSum0 = cumsum(pxAlpha0)
cumSum1 = cumsum(pxAlpha1)
cumSum2 = cumsum(pxAlpha2)

cat("Prob Acumulada alpha=0.5:", cumSum0 )
cat("Prob Acumulada alpha=1:", cumSum1 )
cat("Prob Acumulada alpha=2:", cumSum2 )


#B)

getRazaoVetor <- function(vetor){
  razao = numeric(19)
  for (i in 1:19){
    razao[i] = vetor[i+1] / vetor[i]
  }
  return(razao)
}

getRazaoK <- function(alpha){
  
  razoes = numeric(19)
  for(k in 1:19){
      razoes[k] = ((k/(k+1)) ^ (1+alpha))
  }
  return(razoes)
}

razaoPalpha0 = getRazaoVetor(pxAlpha0)
razaoPalpha1 = getRazaoVetor(pxAlpha1)
razaoPalpha2 = getRazaoVetor(pxAlpha2)

razaoKalpha0 = getRazaoK(0.5)
razaoKalpha1 = getRazaoK(1)
razaoKalpha2 = getRazaoK(2)

if(all.equal(razaoKalpha0, razaoPalpha0)){
  cat("Para alpha=0.5, a igualdade e satisfeita")
}

if(all.equal(razaoKalpha1, razaoPalpha1)){
  cat("Para alpha=1, a igualdade e satisfeita")
}

if(all.equal(razaoKalpha2, razaoPalpha2)){
  cat("Para alpha=2, a igualdade e satisfeita")
}

#C)
#Espero encontrar probabilidades em que P(X=K+1) sejam muito maiores do que P(X=K), ou seja, a velocidade com que a função cai aumenta

#D)
x<-1:20

par(mfrow=c(1,3)) # janela grafica com uma linha de 2 plots
#alpha = 1/2
plot(log(x), log(pxAlpha0), type = "h", main = "P(X=k), alpha = 1/2, (Escala Log)")
abline(log(1/2.612), -(1 + 0.5))
#alpha = 1
plot(log(x), log(pxAlpha1), type = "h", main = "P(X=k), alpha = 1, (Escala Log)")
abline(log(1/1.645), -(1 + 1))
#alpha = 2
plot(log(x), log(pxAlpha2), type = "h", main = "P(X=k), alpha = 2, (Escala Log)")
abline(log(1/1.202), -(1 + 2))

#E)
rzipf = function(nsim = 1, alpha = 1, Cte = 1/1.645){
  res = numeric(nsim)
  for(i in 1:nsim){
    x = -1
    k = 1
    F = p = Cte
    U = runif(1)
    while( x == -1){
      if(U < F) x = k
      else{
        p = p * (k/(k+1))^(1+alpha)
        F = F + p
        k = k+1
      }
    }
    res[i] = x
  }
  res
}

cat(rzipf(400, alpha = 1/2, Cte = 1/2.612))
cat(rzipf(400, alpha = 1, Cte = 1/1.645))
cat(rzipf(400, alpha = 2, Cte = 1/1.202))
