#!/bin/bash
#
# wrapper function to run all mlp simulations with rdm loss on hidden layer
#
# note: this code is quite ancient and requires Tensorflow 1.x to work 
#
# Timo Flesch, 2020

# single hidden layer, baseline
python rdmloss/blobs_constrained_exp1.py;

# single hidden layer, rdm loss
python rdmloss/blobs_constrained_exp2.py;

# two hidden layers, baseline 
python rdmloss/blobs_constrained_exp3.py;

# two hidden layers, rdm loss 2nd
python rdmloss/blobs_constrained_exp4.py;

# two hidden layers, rdm loss on both 
python rdmloss/blobs_constrained_exp5.py;
