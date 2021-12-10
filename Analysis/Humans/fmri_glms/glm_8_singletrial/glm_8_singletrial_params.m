function params = glm_8_singletrial_params()
    %% glm_8_singletrial_params()
    %
    % set parameters for glm estimation
    %
    % Timo Flesch, 2018,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford

    params = struct();

    
    params.glmName = 'glm_8_singletrial';
    % directories
    params.projectdir = '/media/timo/data/DPHIL_01_TREES_FMRI/Paper/code/';
    params.dir.imDir = [params.projectdir 'Data/Humans/part3_fmri/fmri_link/'];
    params.dir.conditionDir = [params.projectdir 'Results/Humans/regressors/'];
    params.dir.glmSubDir = [params.glmName '/'];
    params.dir.glmDir = [params.projectdir 'Results/Humans/GLMs/' params.dir.glmSubDir];
    params.dir.runSubDir = 'ep2d_64mx_35mm_TE30ms_';
    params.dir.dmatSubDir = 'dmats/';
    params.dir.estSubDir = 'estimates/';
    params.dir.tSubDir = 'dmats/'; %'tContrasts/';
    params.dir.groupSubDir = 'groupLevel/';
    params.dir.losoSubDir = 'losoROI/';

    % file handling
    params.files.overwriteContrasts = 1; % delete already estimated contrasts;

    % design matrix
    params.dmat.units = 'secs';
    params.dmat.microtime_res = 32;
    params.dmat.microtime_onset = 16;
    params.dmat.TR = 2;

    % various other parameters
    params.misc.highpass = 128; % get rid of very low frequency oscillations
    params.misc.basisfuncts = [0 0];
    params.misc.volterra = 1; % order of convolution for model interaction (volterra). [1,2]
    params.misc.global_norm = 'none'; % no global normalisation of values
    params.misc.mask = ''; % explicit mask (for ROI based analyis, I think...)
    params.misc.serialcorr = 'AR(1)'; % autoregressive model to take account for temporal correlation of signal
    params.misc.mthresh = 0.8;

    % regular expressions for filenames
    params.regex.functional = '^swauf.*\.nii$'; % functional images
    params.regex.motionregs = '^rp.*\.txt$'; % txt file with motion regressors (from realignment preproc step)
    params.regex.contrasts = '^con.*\.nii$'; % contrast images

    % numbers
    params.num.subjects = 32; % number of subjects (o rly)
    params.num.runs = 6; % number of runs
    params.num.conditions = 2; % number of conditions
    params.num.motionregs = 6; % number of motion regressors

    params.num.badsSubjects = [28];
    params.num.goodSubjects = 1:params.num.subjects;
    params.num.goodSubjects(params.num.badsSubjects) = [];

    params.num.runIDs = [8, 12, 16, 20, 24, 28];
    params.num.s22runIDs = [10, 14, 18, 22, 26, 30];

    % monitoring params
    params.monitor.reviewDMAT = 0;
    params.monitor.reviewContrasts = 0;

end
