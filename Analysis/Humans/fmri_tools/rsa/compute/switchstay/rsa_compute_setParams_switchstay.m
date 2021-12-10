function params = rsa_compute_setParams_switchstay(phaseID)
  %% RSA_COMPUTE_SETPARAMS()
  %
  % sets parameters for rdm computation
  %
  % Timo Flesch, 2020,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford


  params = struct();

  %% directories
  % params.dir.projectDir = '/mnt/big/timo/EXPERIMENTS/EXP_TREES/Granada2/';
  params.dir.projectDir = 'project';

  params.dir.behavDir  = [params.dir.projectDir 'exp_3_granada_fmri/results/behav/'];
  params.dir.dissimDir = [params.dir.projectDir 'exp_1_granada_arenatask/data/final/renamed/fmri_identifiers/'];

  params.dir.inDir    = [params.dir.projectDir 'exp_3_granada_fmri/results_paper/GLM_7_RSA_SWITCHSTAY/'];
  params.dir.outDir   = [params.dir.projectDir 'exp_3_granada_fmri/results_paper/rsa_switchstay/' phaseID '/'];
  params.dir.maskDir  = [params.dir.projectDir 'exp_3_granada_fmri/results_paper/masks/'];

  params.dir.subDir.SPM     = 'dmats/';
  params.dir.subDir.RDM     = 'rsa/';
  params.dir.subDir.GRP     = 'GROUP/';
  params.dir.subDir.rsabetas = 'betas/';

  %% file names
  params.names.subjectDir   =       @(subID)sprintf('TIMO%03d',subID);
  params.names.groupMask    =        'groupMask_rsaSearchlight_paper';
  % params.names.rdmSetIn     =              ['rdmSet_' phaseID '_']; %_cval_slRSA_paper_
  % params.names.rdmSetOut    =              ['rdmSet_' phaseID '_']; %_cval_slRSA_paper_
  % params.names.betasOut     =          ['modelBetas_' phaseID '_']; %_cval_slRSA_paper_
  params.names.rdmSetIn     =              ['rdmSet_glm7_' phaseID '_']; %_cval_slRSA_paper_
  params.names.rdmSetOut    =              ['rdmSet_glm7_' phaseID '_']; %_cval_slRSA_paper_
  params.names.betasOut     =          ['modelBetas_glm7_' phaseID '_']; %_cval_slRSA_paper_
  params.names.models       =               'fmri_rsa_modelRDMs_cval';
  params.names.modelset     =                            'rotAndcomp'; % was "rotAndcomp"

  %% numbers
  params.num.subjects   = 32; % number of subjects (o rly)
  params.num.runs       =  6; % number of runs
  params.num.conditions = 50; % number of conditions
  params.num.motionregs =  6; % number of motion regressors (ACTUALLY 6 BUT INCLUDE the STAY REGRESSOR)

  params.num.badsSubjects = [28];
  params.num.goodSubjects = 1:params.num.subjects;
  params.num.goodSubjects(params.num.badsSubjects) = [];

  params.num.runIDs    =     [8,12,16,20,24,28];
  params.num.s22runIDs =    [10,14,18,22,26,30];

  %% rsa
  params.rsa.method    =         'roi'; % 'roi', 'searchlight'
  params.rsa.whichruns =        'cval'; % 'avg', 'crossval'. avg is mean RDM across runs. If crossval is selected, creates nRunsxnRuns RDM (brain and models), where within run dissims are NaN
  params.rsa.metric    = 'correlation'; % distance metric
  params.rsa.whiten    =             0; % whiten betas (irrespective of dist measure)
  params.rsa.radius    =             3; % radius of searchlight sphere

  %% hpc
  params.hpc.parallelise = 1;
  params.hpc.numWorkers  = 4;




    %% statistical inference
    params.statinf.doFisher  = 0; % fisher transformation (if method==spearman and test == t-test)
    params.statinf.threshVal = .05;
    params.statinf.threshStr = '005';%char(replace(num2str(params.statinf.threshVal),'.','')); % for file names
    params.statinf.method    = 'ttest'; % signrank or t-test
    params.statinf.tail       = 'right';   % right or both makes sense for modelcorrelations
