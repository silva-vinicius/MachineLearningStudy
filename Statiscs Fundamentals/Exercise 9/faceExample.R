library(imager)
setwd("~/Documents/FundEst/Lista9/YaleFaces")
#================================GETTING STARTED================================
im <- load.image('Faces3/s5.jpg')
plot(im)
im #image we just read -> it contains the same image represented in 3 different colour channels.'''

# As our images are black and white, we can discard two of these channels -> They are identical
im_mat <- im[,,1]

#just double checking that im_mat is indeed a matrix
is.matrix(im_mat)

#checking the matrix's dimension
dim(im_mat)

#im_mat is not an image anymore
plot(im_mat)


#generating a heatmap of the matrix
image(im_mat)

#Gray shade histogram
hist(im_mat)

#sampling a few pixels from the matrix
sample(im_mat,5)
#================================================================================


#THE REAL DATA ANALYSIS STARTS HERE


photos <- list()
for(i in 1:15){
  
  photos[[i]]<-list() #each element of 'photos' corresponds to an individual with 11 pictures, that's why each element is also a list
  
  
  for (j in 1:11){
    
    photos[[i]][[j]] <- load.image(sprintf("Faces%i/s%i.jpg", i,j))
    
  }
}

#just for fun, let's print the 7th photo of the 9th individual:
plot(photos[[9]][[7]])


#let's visualize some random faces:
opar <- par() #just saving the dafault plotting settings
par(mfrow=c(4,5), mar=c(0,0,0,0))

#plotting the first 5 pictures of some guys
for(i in c(1,3,8,10)){
  for(j in 1:5){
    plot(photos[[i]][[j]], axes=FALSE)
  }
}

#restoring the default plotting settings
par(opar)


#converting the photos to matrixes
photosmat <- list()
for(i in 1:15){
  photosmat[[i]] <- list()
  
  for(j in 1:11){
    photosmat[[i]][[j]] <- photos[[i]][[j]][ , , 1] 
  }
}

#checking if all the 11 pictures of individual 5 are matrixes
sapply(photosmat[[5]], is.matrix)

#checking whether these matrixes dimensions are 100x100:
sapply(photosmat[[5]], dim)

#we need to transform each image into a single matrix column:
mat_pixels <- matrix (0, nrow = (100*100), ncol = (11*15))

for(i in 1:15){
  for(j in 1:11){                                                        #1 means that we are collecting the first column returned by the stack function
    mat_pixels[,j+(i-1)*11] <- stack(as.data.frame(photosmat[[i]][[j]]))[,1]
  }
}

#spliting the dataset into training and testing
set.seed(123)

#getting the random indexes of the pics who are going constitute our test set:
ind <- sample(15, 1:11, replace=TRUE)

indcol <- ind + ((1:15)-1)*11

mat_test = mat_pixels[,indcol]
dim(mat_test)

#removing the test samples from the training data
mat_pixels <- mat_pixels[,-indcol]

#we need to center the matrix around its mean. 
#we can do that by computing the "average photo" and subtracting it from each photo in the training dataset
mat_mean <- apply(mat_pixels, MARGIN=1, FUN=mean)
mat_centered <- mat_pixels - mat_mean


#we can visualize this average picture
avg_pic <- as.cimg(mat_mean, x=100, y=100)
par(mfrow=c(1,1))
plot(avg_pic, axes=FALSE)

#we should now obtain the principal components of our training dataset

#this wont work because mat_centered transposed has more columns than rows
#pca_pixels <- princomp(t(mat_centered))

pca_pixels <- prcomp(t(mat_centered))

summary(pca_pixels)

dim(pca_pixels$rot)

plot(pca_pixels)

eigenvalues <- (pca_pixels$sdev)^2
barplot(cumsum(eigenvalues))

aux <- cumsum(eigenvalues)/sum(eigenvalues)
barplot(aux)

#we can detail this plot a bit more:
barplot(aux[1:30], ylim = c(0,1))

#we are going to use 20 eigenvectors
eigenvectors <-pca_pixels$rot[,1:20]

#we are going to unstack our 20 eigenvectors and create eigenfaces!
eigen_faces <- list()

for(i in 1:20){
  eigen_faces[[i]] <- as.cimg(pca_pixels$rot[,i], x=100, y=100)
}

#we can visualize these images: They're quite scary, aren't they?
par(mfrow=c(4,5), mar=c(0,0,0,0))
for(i in 1:20){
  plot(eigen_faces[[i]], axes = FALSE)
}


#we are going to try and represent a single test picture as linear combination of an average picture and the eigen_faces
coef <- t(eigenvectors) %*% (mat_test[,5] - mat_mean)
dim(coef)
#now coef is the representation of image 5 using the eigenvectors

photo_test5 <- list()

photo_test5[[1]] <- as.cimg(mat_test[,5], x=100, y=100)

#we are going to represent the same image using different amounts of eigenvectors
for(i in 2:20){
  
  approx_vector <- mat_mean + eigenvectors[,1:i] %*% coef[1:i, 1]
  photo_test5[[i]] <- as.cimg(as.numeric(approx_vector), x=100, y=100)
}

opar<- par()
par(mfrow=c(4,5), mar=c(0,0,0,0))
for(i in 1:20) plot(photo_test5[[i]], axes=FALSE)
par(opar)


#we are now going to approximate all the faces in our  test dataset
coef<-t(eigenvectors) %*% (mat_test - mat_mean)
dim(coef)

#now we are going to do the same thing using the training dataset:
coef_training <- t(eigenvectors)%*%(mat_pixels - mat_mean)

#just visualizing how the train pictures behave  in a 2D space:
par(mfrow=c(1,1))
colface <- rainbow(15)[rep(1:15, rep(10,15))]
plot(coef_training[1,], coef_training[2,], pch=21, bg=colface)

avgcoef <-matrix(0, ncol = 15, nrow = 20)
for(k in 1:15){
 
  avgcoef[,k] = apply(coef_training[ ,(1+(k-1)*10) : (k*10)], 1, mean)
}

par(mfrow=c(1,1))
colface <- rainbow(15)
plot(avgcoef[1,], avgcoef[2,], pch=21, bg=rainbow(15))


indexClosest <- numeric()
for(j in 1:15){
  indexClosest[j] = which.min(apply((avgcoef - coef[,j])^2, 2, mean))
}

#here we see the predictions for our test faces
indexClosest

