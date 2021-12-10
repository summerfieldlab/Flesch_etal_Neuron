function rsa_compute_regressRDMs_searchlight_switchstay(phaseName)
    %% rsa_compute_performRSA_searchlight()
    %
    % computes rdms using a spherical searchlight approach and performs regression
    % on candidate model RDMs
    %
    % Timo Flesch, 2020
    % Human Information Processing Lab
    % University of Oxford
    if ~exist('phaseName')
      phaseName = 'stay';
    end

    if strcmp(phaseName,'stay')
      phaseIDs = [ones(1,50),zeros(1,50)];
      badtrialIDs = 1:300;
    elseif strcmp(phaseName,'switch')
      phaseIDs = [zeros(1,50),ones(1,50)];
      badtrialIDs = 301:600;
    end

    params = rsa_compute_setParams_switchstay(phaseName);
    % group-level whole brain mask
    gmaskMat  = fmri_io_nifti2mat([params.names.groupMask '.nii'],params.dir.maskDir);
    gmaskVect = gmaskMat(:);
    gmaskIDsBrain = find(~isnan(gmaskVect));

    grpDir = [params.dir.inDir params.dir.subDir.GRP];

    % generate candidate models
    modelRDMs = rsa_corrs_genSLModelRDMs_cval()
    % modelRDMs = rsa_corrs_genINDSCALModelRDMs_cval()

    %% recruit an army of minions!
    if params.hpc.parallelise
        parpool(params.hpc.numWorkers);
    end

    parfor (ii = 1:length(params.num.goodSubjects), params.hpc.numWorkers)
    % for (ii = 1:length(params.num.goodSubjects))

      subID = params.num.goodSubjects(ii);
      % navigate to subject folder
      subStr = params.names.subjectDir(subID);
      subDir = [params.dir.inDir subStr '/'];

      disp(['Searchlight RSA Model Regressions - processing subject ' subStr]);
      spmDir = [params.dir.inDir subStr '/' params.dir.subDir.SPM];
      rsaDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];
      corrs = [];

      % load SPM.mat
      cd(spmDir);
      SPM = load(fullfile(pwd,['../' params.dir.subDir.SPM 'SPM.mat']));
      SPM = SPM.SPM;

      % import all betas
      bStruct = struct();
      [bStruct.b,bStruct.events] = load_betas(SPM,params.num.runs,phaseIDs,params.num.motionregs);
      
      bStruct.b = reshape(bStruct.b,[prod(size(gmaskMat)),size(bStruct.b,4)]);
      bStruct.b = reshape(bStruct.b,[size(bStruct.b,1),params.num.conditions,params.num.runs]);
      bStruct.events = reshape(bStruct.events,[params.num.conditions,params.num.runs]);

      X = helper_construct_designMatrix(modelRDMs,ii);
      rdmSet = [];

      for sphereID = 1:length(gmaskIDsBrain)

        % obtain coordinates of centroid
        [x,y,z] = ind2sub(size(gmaskMat),gmaskIDsBrain(sphereID));
        % create spherical mask
        [sphereIDs,~] = fmri_mask_genSphericalMask([x,y,z],params.rsa.radius,gmaskMat);

        % mask betas with sphere, only use masked values
        betas = squeeze(bStruct.b(sphereIDs,:,:));
        betas = permute(betas,[2,3,1]);

        rdm = rsa_compute_rdmSet_cval(betas,params.rsa.metric);

        % centre the dependent variable
        Y = nanzscore(vectorizeRDM(rdm))';

        % perform regression
        b = regress(Y,X);
        corrs(:,sphereID) = b;
        [~,rdmSet(sphereID,:,:)] = rsa_compute_averageCvalRDMs(rdm,params.num.runs,params.num.conditions);

      end
      results = struct();
      results.betas = corrs;
      results.params = params;

      if ~exist('rsaDir','dir')
        mkdir(rsaDir);
      end
      cd(rsaDir);

      % save results
      subRDM = struct();
      subRDM.rdms = rdmSet;
      subRDM.events   = bStruct.events(:,1);
      subRDM.subID    = subID;
      subRDM.indices  = gmaskIDsBrain;
      helper_parsave([params.names.rdmSetOut],subRDM);

      if ~exist('params.dir.subDir.rsabetas','dir')
          mkdir(params.dir.subDir.rsabetas);
      end

      cd(params.dir.subDir.rsabetas);
      helper_parsave([params.names.betasOut '_set_' params.names.modelset '_sub_' num2str(subID)],results);

    end

    if params.hpc.parallelise
        delete(gcp);
    end

end

function helper_parsave(fName,results)
  save(fName,'results');
end

function dmat = helper_construct_designMatrix(modelRDMs,ii)
  dmat = [];
  for modID = 1:length(modelRDMs)
      dmat(:,modID) = nanzscore(vectorizeRDM(squeeze(modelRDMs(modID).rdms(ii,:,:))));
  end
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
