cat("\014")
library(smacof)
library(R.matlab)
rm(list=ls())
data <- readMat('project/results_paper/GLM_3_RSA/GROUP/rdmSet_indscal_all7mods.mat');


nROI=5
nsubj=31 # number of subjects
ncond=50 # this is the number of experimental conditions or objects
weight <- array(rep(NaN, 3*3*nsubj*nROI),dim=c(3,3,nsubj,nROI))
gspace <- array(rep(NaN, ncond*3*nROI),dim=c(ncond,3,nROI))
stressall <- array(rep(NaN, 1*nROI), dim=c(1,nROI))
for (k in 1:nROI) {
    y <- list()
    for (i in 1:nsubj) { # prepare data structure
    y[[i]] <- data[[1]][,,i,k]
    }
    #fit metric indscal
    out <- smacofIndDiff(y, ndim = 3, type = c("ratio"),
                constraint = c("indscal"), itmax = 100000, eps = 1e-10, verbose = TRUE)
    stressall[k] <- out$stress

    for (i in 1:nsubj) {
    weight[,,i,k] <- out$cweights[[i]] #subject-specific weights
    }
    gspace[,,k] <- out$gspace #common space
}
plot(out)

filename <- paste("out_indscal_3d_paper", ".mat", sep = "")
writeMat(filename, weight=weight, gspace=gspace, verbose = 0)
