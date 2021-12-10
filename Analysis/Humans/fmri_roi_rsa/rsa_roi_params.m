function params = rsa_roi_params()
  %% rsa_roi_params()
  %
  % sets parameters for rdm computation
  %
  % Timo Flesch, 2020,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford


  params = struct();

  %% directories
  params.dir.projectDir = '/media/timo/data/DPHIL_01_TREES_FMRI/Paper/code/';

  params.dir.behavDir = [params.dir.projectDir 'Data/Humans/part3_fmri/behav/scan/'];
  params.dir.dissimDir = [params.dir.projectDir 'Data/Humans/part1_arena/'];

  params.dir.inDir = [params.dir.projectDir 'Results/Humans/GLMs/glm_4_rsa/'];
  params.dir.outDir = [params.dir.projectDir 'Results/Humans/RSA/ROIs/'];
  params.dir.maskDir = [params.dir.projectDir 'Results/Humans/Masks/'];

  params.dir.subDir.SPM     = 'dmats/';
  params.dir.subDir.RDM     = 'rsa/';
  params.dir.subDir.GRP     = 'grouplevel/';
  params.dir.subDir.rsabetas = 'betas/';

  %% file names
  params.names.subjectDir   =       @(subID)sprintf('TIMO%03d',subID);
  params.names.groupMask = 'groupMask_cval_searchlight';
    params.names.rdmSetIn = 'brainRDMs_cval_searchlight';
    params.names.rdmSetOut = 'brainRDMs_cval_searchlight';
    params.names.betasOut = 'betas_cval_searchlight';
    params.names.models = 'modelRDMs_cval_searchlight';
    params.names.modelset = 'all7mods'; % all 7


  %% model correlations
  params.corrs.modelset  = str2num(params.names.modelset);  
  params.corrs.doOrth    =  0; % apply Gram-Schmidt orthogonalisation (y/n)
  params.corrs.method    = 'spearman'; % kendall, regression (instead of correlations), spearmann
  params.corrs.whichruns =      'cval'; % 'avg', 'cval'.cval is full runxrun matrix, avg is the average of this over runs

  
  %% numbers
  params.num.subjects   = 32; % number of subjects (o rly)
  params.num.runs       =  6; % number of runs
  params.num.conditions = 50; % number of conditions
  params.num.motionregs =  6; % number of motion regressors 

  params.num.badsSubjects = [28];
  params.num.goodSubjects = 1:params.num.subjects;
  params.num.goodSubjects(params.num.badsSubjects) = [];

  params.num.runIDs    =     [8,12,16,20,24,28];
  params.num.s22runIDs =    [10,14,18,22,26,30];

  %% rsa
  params.rsa.method    = 'searchlight'; % 'roi', 'searchlight'
  params.rsa.whichruns =        'cval'; % 'avg', 'cval'. avg is mean RDM across runs. If crossval is selected, creates nRunsxnRuns RDM (brain and models), where within run dissims are NaN
  params.rsa.metric    = 'correlation'; % distance metric
  params.rsa.whiten    =             0; % whiten betas (irrespective of dist measure)
  params.rsa.radius    =             3; % radius of searchlight sphere

  %% hpc
  params.hpc.parallelise = 1;
  params.hpc.numWorkers  = 6;




  %% statistical inference
  params.statinf.doFisher  = 0; % fisher transformation (if method==spearman and test == t-test)
  params.statinf.threshVal = .05;
  params.statinf.threshStr = '005';
  params.statinf.method    = 'ttest'; % signrank or t-test
  params.statinf.tail      = 'right';   % right or both makes sense for modelcorrelations
