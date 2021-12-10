function rdmCollection = rsa_compute_performRSA_ROI_switchstay_glm7(roiName,phaseName)
  %% rsa_compute_performRSA_ROI()
  %
  % computes rdms for each subject, saves results
  % at subject and group-average level
  %
  % Timo Flesch, 2020
  % Human Information Processing Lab
  % University of Oxford


  params = rsa_compute_setParams_switchstay_glm7(phaseName);
  n_workers = params.hpc.numWorkers;
  if exist('roiName','var') % if user has specified roi, overwrite params
    params.names.roiMask = roiName;
  end

  if ~exist('phaseName','var')
    phaseName = 'stay';
  end

  % load trial indices 
  trial_ids = load('trial_ids.mat'); trial_ids = trial_ids.trial_ids;
  
  
  grpDir = [params.dir.inDir params.dir.subDir.GRP];
  rdmCollection = nan(length(params.num.goodSubjects),params.num.conditions*params.num.runs,params.num.conditions*params.num.runs);
  parfor (ii = 1:length(params.num.goodSubjects),n_workers)
%  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    % load single subject mask
    % gmaskMat  = fmri_io_nifti2mat([params.names.roiMask 'sub' num2str(subID) '.nii'],params.dir.maskDir,1);  
    gmaskMat  = fmri_io_nifti2mat([params.names.roiMask '.nii'],params.dir.maskDir,1);  
    gmaskVect = gmaskMat(:);
    gmaskVect(gmaskVect==0) = NaN;    
    gmaskIDsBrain = find(~isnan(gmaskVect));
    % laoad trial indices
    trials = trial_ids{subID};

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
%     keyboard
    % import betas, mask them appropriately
    disp('....importing betas');
    bStruct = struct();
    
    [bStruct.b,bStruct.events] = load_betas(SPM,params.num.runs,phaseName,trials,params.num.motionregs,gmaskIDsBrain);
    
    
    % bStruct.b(isnan(bStruct.b)) = 0;
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
  
  save(['groupAvg_' params.names.rdmSetOut params.names.roiMask '.mat'],'groupRDM');
  

end

function parsave(fileName,subRDM)
    save(fileName,'subRDM');
end


function [b,events] = load_betas(SPM,nRuns,phaseName,trials,nMotRegs,maskIndices)    
    %
    % discard motion regressors and constant terms (for run),
    % prepopulate b and events matrices with nruns*nconds entries (all NaN)
    % insert actual data at appropriate places 
    
    if strcmp(phaseName,'stay')
        phaseID = 1;
    elseif strcmp(phaseName,'switch')
        phaseID = 2;
    end
    
    % add motion regressors and constants to trial indices     
    trials_new = [];
    for runID = 1:nRuns
        tmp = trials(trials(:,2)==runID,:);
        trials_new = [trials_new;tmp;zeros(6,5);];
    end
    trials_new = [trials_new;zeros(6,5)];
    
    % create beta mask 
    betaMask = zeros(1,length(SPM.Vbeta));
    for runID = 1:nRuns
        betaMask(1,trials_new(:,1)==phaseID & trials_new(:,2)==runID) = 1;
    end   
    % load betas 
    betaIDs     =  find(betaMask==1);
    betasToLoad = SPM.Vbeta(betaIDs);  
    if exist('maskIndices','var')
      b = spm_data_read(betasToLoad, maskIndices);
    else
      b = spm_data_read(betasToLoad);
    end
    events =                    {betasToLoad.descrip};
    % remember that above doesn't contain conditions with missing data
    % hence, init new beta mat and add conditions at appropriate places
    idces = trials_new(betaMask==1,:);
    betaMat = nan(50*6,size(b,2));
    eventMat = cell(50*6,1);
    ii = 1;
    for iRun = 1:nRuns
        for iTask = 1:2
            for iBranch = 1:5
                for iLeaf = 1:5
                    trial = find(idces(:,2)==iRun & idces(:,3)==iTask & idces(:,4)==iBranch & idces(:,5)==iLeaf);
                    if ~isempty(trial)
                        betaMat(ii,:) = b(trial,:);
                        eventMat{ii} = events{trial};
                    end
                    ii = ii+1;
                end
            end
        end
    end
    b = betaMat;
    events = eventMat;
            
  end
  