function params = rsa_corrs_setParams_switchstay(phaseID)
  %% RSA_CORRS_SETPARAMS()
  %
  % sets parameters for correlation between estimated brain and model RDMs
  %
  % Timo Flesch, 2019

  params = struct();
% params.dir.projectDir = 'project';
  %% directories
  % directory for behavioural data
  params.dir.behavDir = 'project/results_paper/behav/';
  % directory for dissimilarity ratings
  params.dir.dissimDir = 'projectexp_1_granada_arenatask/results/matfiles/';
  % directory for tree images
  params.dir.imageDir  = 'project/task/treestask_fmri/files/stimuli/';
  params.dir.bgDir     = 'projectFigures/task/';
  % imaging directories
  params.dir.inDir    = 'project/results_paper/GLM_7_RSA_SWITCHSTAY/';
  params.dir.inDir2    = 'project/results_paper/GLM_7_RSA_SWITCHSTAY/';
  params.dir.outDir   =  'project/results_paper/rsa_switchstay/';
  params.dir.maskDir  = 'project/results_paper/masks/';
  % params.dir.maskDir  = 'project/results_paper/rsa/cval_correlation/regress_singleRuns/masks/set4';
  params.dir.subDir.SPM     = 'dmats/';
  params.dir.subDir.RDM     = 'rsa/';
  params.dir.subDir.out   = 'corrs/';
  params.dir.subDir.GRP     = 'GROUP/';



  %% file names
  params.names.subjectDir  = @(subID)sprintf('TIMO%03d',subID);
  params.names.groupMask   = 'groupMask_rsaSearchlight';
  % params.names.rdmSetIn     =              ['rdmSet_' phaseID '_']; %_cval_slRSA_paper_
  % params.names.rdmSetOut    =              ['rdmSet_' phaseID '_']; %_cval_slRSA_paper_
  % params.names.corrsOut     =          ['modelBetas_' phaseID '_']; %_cval_slRSA_paper_
  params.names.rdmSetIn     =              ['rdmSet_glm7_' phaseID '_']; %_cval_slRSA_paper_
  params.names.rdmSetOut    =              ['rdmSet_glm7_' phaseID '_']; %_cval_slRSA_paper_
  params.names.corrsOut     =          ['modelBetas_glm7_' phaseID '_']; %_cval_slRSA_paper_
  params.names.models      =          'fmri_rsa_modelRDMs_cval';
  params.names.modelset    = 'forPAPER_';


  %% model correlations
  params.corrs.modelset  = str2num(params.names.modelset);
  % params.corrs.modellist = [1,2,12,10,11,3,4,9,8]; % list of models to include
  %params.corrs.modellist = [21,22,23,26]; % list of models to include
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

  params.num.runIDs    =    [8,12,16,20,24,28];
  params.num.s22runIDs =    [10,14,18,22,26,30];


  %% statistical inference
  params.statinf.doFisher  = 0; % fisher transformation (if method==spearman and test == t-test)
  params.statinf.threshVal = .05;
  params.statinf.threshStr = '005';%char(replace(num2str(params.statinf.threshVal),'.','')); % for file names
  params.statinf.method    = 'ttest'; % signrank or t-test
  params.statinf.tail       = 'right';   % right or both makes sense for modelcorrelations
