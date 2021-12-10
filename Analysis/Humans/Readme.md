# Analysis Pipelines

Note: All pipelines require MATLAB with RSAtoolbox and SPM on the filepath.
These functions recreate the key analyses reported in the manuscript. 
For prettier figures (i.e. the ones shown in the ms), first run the pipelines below

## Pre-Screening (arena task)

Run 'pipeline_analyse_arena.m' to recreate analyses of the pre-screening data.  
In particular:

- correlations with various model RDMs
- comparison of estimated grid prior with our inclusion criterion
- MDS projections of ratings given by participants with low, medium and high priors

## Training and Test Phase: Behavioural Data

Run 'pipeline_analyse_behaviour.m' to recreate analyses of the behavioural data from the 
main phase of the experiment.  
 **Note: These pipelines run multiple analyses and store the resutls in 
mat files. Results can then be visualised with the plot_figures.ipynb iPython notebook**  
Conducted analyses:

- learning curves 
- accuracy baseline vs test 
- choice pattern analysis
- psychophysical model estimates baseline vs test 
- switch vs stay analysis

## Test Phase: fMRI Data

Run 'pipeline_analyse_fmri.m' to recreate all reported analyses of the fMRI data.
Note that this pipeline is quite compute heavy, as it includes preprocessing, GLM estimation and 
searchlight RSA. 
**Most results can be visualised with the plot_figures iPython notebook. heatmaps of fMRI activity can be 
inspected with your favourite fMRI analysis package. We used bspmview and MRIcron for figures shown in the paper.**