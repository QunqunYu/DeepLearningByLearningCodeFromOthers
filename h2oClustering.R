##################################################################################
# This file is going to compare the clustering method in H2O
# Main object is using autoencoder to do clustering
# http://learn.h2o.ai/content/hands-on_training/dimensionality_reduction.html
#
##################################################################################


# Dimensionality reduction of MNIST

library(h2o)
library(darch)
data <- readMNIST("H2OPL/data/")

# Load the MNIST digit recognition dataset into R
# http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call:  show_digit(train$x[5,])   to see a digit.
# brendan o'connor - gist.github.com/39760 - anyall.org
train <- list()
train$x <- matrix(0, nrow = 60000, ncol = 784)
train$y <- rep(0, 60000)
train$n <- 60000
test <- list()
test$n <- 10000
test$x <- matrix(0, nrow = 10000, ncol = 784)
test$y <- rep(0, 10000)

load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('H2OPL/data/train-images-idx3-ubyte')
  test <<- load_image_file('H2OPL/data/t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('H2OPL/data/train-labels-idx1-ubyte')
  test$y <<- load_label_file('H2OPL/data/t10k-labels-idx1-ubyte')  
}


show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

load_mnist()
show_digit(train$x[9,])

write(tr)
