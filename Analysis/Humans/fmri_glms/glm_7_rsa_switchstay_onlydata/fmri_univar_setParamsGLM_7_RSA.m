function params = fmri_univar_setParamsGLM_7_RSA()
%% FMRI_SET_GLMPARAMS()
%
% set parameters for glm estimation
%
% Timo Flesch, 2020,
% Human Information Processing Lab,
% Experimental Psychology Department
% University of Oxford

  params = struct();

  params.glmName =  'GLM_7_RSA_SWITCHSTAY';
  % directories
  params.dir.imDir        = 'project/data/data_fmri/renamed/';
  params.dir.conditionDir = 'project/data/data_behav/final/fmri_identifiers/conditions/';
  params.dir.glmSubDir   = [params.glmName '/'];
  params.dir.glmDir      = ['project/results_paper/' params.dir.glmSubDir];
  params.dir.runSubDir  = 'ep2d_64mx_35mm_TE30ms_';
  params.dir.dmatSubDir = 'dmats/';
  params.dir.estSubDir  = 'estimates/';
  params.dir.tSubDir    = 'dmats/';%'tContrasts/';
  params.dir.groupSubDir = 'groupLevel/';
  params.dir.losoSubDir = 'losoROI/';

  % file handling
  params.files.overwriteContrasts = 1; % delete already estimated contrasts;

  % design matrix
  params.dmat.units           = 'secs'; % TODO description
  params.dmat.microtime_res   =     32; % TODO description
  params.dmat.microtime_onset =     16; % TODO description
  params.dmat.TR              =      2; % inter image interval

  % various other parameters
  params.misc.highpass        =     128; % get rid of very low frequency oscillations
  params.misc.basisfuncts     =   [0 0]; % dunno
  params.misc.volterra        =       1; % order of convolution for model interaction (volterra). [1,2]
  params.misc.global_norm     =  'none'; % no global normalisation of values
  params.misc.mask            =      ''; % explicit mask (for ROI based analyis, I think...)
  params.misc.serialcorr      = 'AR(1)'; % autoregressive model to take account for temporal correlation of signal
  params.misc.mthresh         =     0.8; % dunno


  % regular expressions for filenames
  params.regex.functional   = '^wauf.*\.nii$'; % functional images
  params.regex.motionregs   = '^rp.*\.txt$';    % txt file with motion regressors (from realignment preproc step)
  params.regex.contrasts    = '^con.*\.nii$';   % contrast images


  % numbers
  params.num.subjects   = 32; % number of subjects (o rly)
  params.num.runs       =  6; % number of runs  
  params.num.motionregs =  6; % number of motion regressors

  params.num.badsSubjects = [28];
  params.num.goodSubjects = 1:params.num.subjects;
  params.num.goodSubjects(params.num.badsSubjects) = [];

  params.num.runIDs    =    [8,12,16,20,24,28];
  params.num.s22runIDs =    [10,14,18,22,26,30];

  % monitoring params
  params.monitor.reviewDMAT      = 0;
  params.monitor.reviewContrasts = 0;


end
