function rdmCollection = rsa_compute_performRSA_ROI_switchstay(roiName,phaseName)
  %% rsa_compute_performRSA_ROI()
  %
  % computes rdms for each subject, saves results
  % at subject and group-average level
  %
  % Timo Flesch, 2020
  % Human Information Processing Lab
  % University of Oxford


  params = rsa_compute_setParams_switchstay(phaseName);
  n_workers = params.hpc.numWorkers;
  if exist('roiName','var') % if user has specified roi, overwrite params
    params.names.roiMask = roiName;
  end

  if ~exist('phaseName','var')
    phaseName = 'stay';
  end

  if strcmp(phaseName,'stay')
    phaseIDs = [ones(1,50),zeros(1,50)];
    badtrialIDs = 1:300;
  elseif strcmp(phaseName,'switch')
    phaseIDs = [zeros(1,50),ones(1,50)];
    badtrialIDs = 301:600;
  end

  % load group level mask (wholebrain OR structural ROI)
  
  % gmaskMat  = fmri_io_nifti2mat([params.names.roiMask '.nii'],params.dir.maskDir,1);
  
  % gmaskVect = gmaskMat(:);
  % gmaskVect(gmaskVect==0) = NaN;
  
  % gmaskIDsBrain = find(~isnan(gmaskVect));
  % grpDir = [params.dir.inDir params.dir.subDir.GRP];
  % bad_trials = load([grpDir 'bad_trials_switchstay']);
  % bad_trials = bad_trials.bad_trials(:,badtrialIDs);
  
  
  grpDir = [params.dir.inDir params.dir.subDir.GRP];
  rdmCollection = nan(length(params.num.goodSubjects),params.num.conditions*params.num.runs,params.num.conditions*params.num.runs);
  parfor (ii = 1:length(params.num.goodSubjects),n_workers)
    subID = params.num.goodSubjects(ii);
    % load single subject mask
    gmaskMat  = fmri_io_nifti2mat([params.names.roiMask 'sub' num2str(subID) '.nii'],params.dir.maskDir,1);  
    gmaskVect = gmaskMat(:);
    gmaskVect(gmaskVect==0) = NaN;    
    gmaskIDsBrain = find(~isnan(gmaskVect));
    % laoad bad trial indices
    bad_trials = load([grpDir 'bad_trials_switchstay']);
    bad_trials = bad_trials.bad_trials(:,badtrialIDs);

    bad_trials_sub = bad_trials(subID,:);
    % navigate to subject folder
    subStr = params.names.subjectDir(subID);
    subDir = [params.dir.inDir subStr '/'];

    disp(['processing subject ' subStr]);
    spmDir = [params.dir.inDir subStr '/' params.dir.subDir.SPM];
    rsaDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];

    % load SPM.mat
    cd(spmDir);
    tmp = load(fullfile(pwd,['../' params.dir.subDir.SPM 'SPM.mat']));
    SPM = tmp.SPM;

    % import betas, mask them appropriately
    disp('....importing betas');
    bStruct = struct();
    
    [bStruct.b,bStruct.events] = load_betas(SPM,params.num.runs,phaseIDs,params.num.motionregs,gmaskIDsBrain);
    
    
    bStruct.b(isnan(bStruct.b)) = 0;
    % bStruct.b = bStruct.b(:,~isnan(gmaskVect));
    bStruct.b = reshape(bStruct.b,[size(bStruct.b,1)/params.num.runs,params.num.runs,size(bStruct.b,2)]);
    bStruct.events = reshape(bStruct.events,[params.num.conditions,params.num.runs]);
    bStruct.idces = gmaskIDsBrain;

    % if mahalanobis, import residuals, whiten betas
    if strcmp(params.rsa.metric,'mahalanobis') || strcmp(params.rsa.metric,'crossnobis') || params.rsa.whiten==1
      disp('....importing residuals')
      cd(spmDir);
      r = rsa_helper_getResiduals(SPM,gmaskIDsBrain,0);
      r = reshape(r,[size(r,1)/params.num.runs,params.num.runs,size(r,2)]);
      r(isnan(r)) = 0;
      rStruct = struct();
      rStruct.r = r;
      rStruct.idces = gmaskIDsBrain;
      disp(' .....  whitening the parameter estimates');
      bStruct.b = rsa_helper_whiten(bStruct.b,rStruct.r);
      cd(rsaDir);
    end
    % compute rdms
    switch params.rsa.whichruns
    case 'avg'
      rdm = rsa_compute_rdmSet_avg(bStruct.b,params.rsa.metric);
    
    case 'cval'
      rdm = rsa_compute_rdmSet_cval(bStruct.b,params.rsa.metric);
    end
    rdm(bad_trials_sub==1,:) = NaN;
    rdm(:,bad_trials_sub==1) = NaN;
    rdmCollection(ii,:,:) = rdm;
    % set entries with missing trials to NaN 

    % navigate to output subfolder
    if ~exist(rsaDir,'dir')
      mkdir(rsaDir);
    end
    cd(rsaDir);

    % save results (with condition labels)
    subRDM = struct();
    subRDM.rdm = squeeze(rdmCollection(ii,:,:));
    subRDM.roiName = params.names.roiMask;
    subRDM.roiIDCES = gmaskIDsBrain;
    subRDM.events   = bStruct.events(:,1);
    subRDM.subID    = subID;
    parsave([params.names.rdmSetOut params.names.roiMask '.mat'],subRDM);
  end
  % navige to group level folder
  cd(grpDir);

  % ..and store group average (for visualisation)
  groupRDM          = struct();
  groupRDM.rdm      = squeeze(nanmean(rdmCollection,1));
  % groupRDM.roiName  = params.names.roiMask;
  % groupRDM.roiIDCES = gmaskIDsBrain;
  % groupRDM.events   = bStruct.events(:,1);
  % groupRDM.subID    = subID;
  save(['groupAvg_' params.names.rdmSetOut params.names.roiMask '.mat'],'groupRDM');
  % delete(gcp);

end

function parsave(fileName,subRDM)
    save(fileName,'subRDM');
end


function [b,events] = load_betas(SPM,nRuns,phaseIDs,nMotRegs,maskIndices)    
    %
    % discard motion regressors and constant terms (for run),
    % assume that each run was modeled with separate set of regressors:
    nTotal      = length(SPM.Vbeta);
    betaCounts  =          1:nTotal;
    betaMask    = [repmat([phaseIDs,zeros(1,nMotRegs)],1,nRuns),zeros(1,nRuns)];
    betaIDs     =  find(betaMask==1);
    betasToLoad = SPM.Vbeta(betaIDs);  
    if exist('maskIndices','var')
      b = spm_data_read(betasToLoad, maskIndices);
    else
      b = spm_data_read(betasToLoad);
    end
    events =                    {betasToLoad.descrip};
  
  end
  