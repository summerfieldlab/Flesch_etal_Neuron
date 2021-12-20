cat("\014")
library(smacof)
library(R.matlab)
rm(list=ls())
data <- readMat('rdmSet_indscal_controlrdms_raw.mat');


nROI=4
nsubj=31 # number of subjects
ncond=50 # this is the number of experimental conditions or objects
weight <- array(rep(NaN, 2*2*nsubj*nROI),dim=c(2,2,nsubj,nROI))
gspace <- array(rep(NaN, ncond*2*nROI),dim=c(ncond,2,nROI))
stressall <- array(rep(NaN, 1*nROI), dim=c(1,nROI))
for (k in 1:nROI) {
    y <- list()
    for (i in 1:nsubj) { # prepare data structure
    y[[i]] <- data[[1]][,,i,k]
    }
    #fit metric indscal
    out <- smacofIndDiff(y, ndim = 2, type = c("ratio"),
                constraint = c("indscal"), itmax = 100000, eps = 1e-10, verbose = TRUE)
    stressall[k] <- out$stress

    for (i in 1:nsubj) {
    weight[,,i,k] <- out$cweights[[i]] #subject-specific weights
    }
    gspace[,,k] <- out$gspace #common space
}
plot(out)

filename <- paste("out_indscal_2d_control_raw", ".mat", sep = "")
writeMat(filename, weight=weight, gspace=gspace, verbose = 0)
