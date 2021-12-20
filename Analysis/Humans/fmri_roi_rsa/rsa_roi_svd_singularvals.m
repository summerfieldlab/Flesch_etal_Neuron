function rsa_roi_svd_singularvals(roiName)
  %% rsa_roi_svd_singularvals
  %
  % computes all sorted singular values for a given ROI
  % at single subject level
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford


  params = rsa_roi_params();

  if exist('roiName','var') % if user has specified roi, overwrite params
    params.names.roiMask = roiName;
  end

  % % load group level mask (wholebrain OR structural ROI)
  % % switch params.rsa.method
  % % case 'searchlight'
  % %   gmaskMat  = fmri_io_nifti2mat([params.names.groupMask '.nii'],params.dir.maskDir);
  % % case 'roi'
  % gmaskMat  = fmri_io_nifti2mat([params.names.roiMask '.nii'],params.dir.maskDir,1);
  % % end
  % gmaskVect = gmaskMat(:);
  % gmaskVect(gmaskVect==0) = NaN;
  % % gmaskVect(isnan(gmaskVect)) = 0;
  % % gmaskIDsBrain = 1:length(gmaskVect);
  % gmaskIDsBrain = find(~isnan(gmaskVect));
  grpDir = [params.dir.inDir params.dir.subDir.GRP];

  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    % load single subject mask
    gmaskMat  = fmri_io_nifti2mat([params.names.roiMask 'sub' num2str(subID) '.nii'],params.dir.maskDir,1);  
    gmaskVect = gmaskMat(:);
    gmaskVect(gmaskVect==0) = NaN;    
    gmaskIDsBrain = find(~isnan(gmaskVect));
    % navigate to subject folder
    subStr = params.names.subjectDir(subID);
    subDir = [params.dir.inDir subStr '/'];

    disp(['processing subject ' subStr]);
    spmDir = [params.dir.inDir subStr '/' params.dir.subDir.SPM];
    rsaDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];

    % load SPM.mat
    cd(spmDir);
    load(fullfile(pwd,['../' params.dir.subDir.SPM 'SPM.mat']));

    % import betas, mask them appropriately
    disp('....importing betas');
    bStruct = struct();
    [bStruct.b,bStruct.events] = rsa_helper_getBetas(SPM,params.num.runs,params.num.conditions,params.num.motionregs,gmaskIDsBrain);
    % [bStruct.b,bStruct.events] = rsa_helper_getBetas(SPM,params.num.runs,params.num.conditions,params.num.motionregs);

    bStruct.b(isnan(bStruct.b)) = 0;
    
    % bStruct.b = bStruct.b(:,~isnan(gmaskVect));
    bStruct.b = reshape(bStruct.b,[size(bStruct.b,1)/params.num.runs,params.num.runs,size(bStruct.b,2)]);
    for runID = 1:params.num.runs

        dat = squeeze(bStruct.b(:,runID,:))';      
        dat = dat - mean(dat,1);
        [U,S,V] = svd(dat);
      singular_values(runID,:) = diag(S);
    end
    % average across runs
    singular_values = squeeze(mean(singular_values,1));

    % navigate to output subfolder
    if ~exist(rsaDir,'dir')
      mkdir(rsaDir);
    end
    cd(rsaDir);

    % save results (with condition labels)
    results = struct();
    results.singular_values = singular_values
    results.roiName = params.names.roiMask;
    results.roiIDCES = gmaskIDsBrain;
    results.events   = bStruct.events(:,1);
    results.subID    = subID;
    save(['singular_values_' params.names.roiMask],'results');
    all_svs(ii,:) = singular_values;
  end
  % navige to group level folder
  cd(params.dir.outDir);

  % ..and store group average (for visualisation)  
  save(['groupAvg_singular_values_' params.names.roiMask],'all_svs');

end
