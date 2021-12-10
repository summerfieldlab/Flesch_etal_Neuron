function rsa_corrs_regressBrainRDMs_ROI_switchstay(maskName,phaseName)
  %% rsa_corrs_corrBrainRDMs()
  %
  % regresses model rdms against brain rdms
  %
  % Timo Flesch, 2019,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford

  params = rsa_corrs_setParams_switchstay(phaseName);

 % load pre-computed model RDMs
  % if ~exist('modelRDMs','var')
  %   modelRDMs = load(params.names.models);
  %   fns = fieldnames(modelRDMs);
  %   modelRDMs = modelRDMs.(fns{1});
  % end
  grpDir = [params.dir.inDir params.dir.subDir.GRP];

  % only include selection of models
  modelRDMs = rsa_corrs_genSLModelRDMs_cval();


  %% do it
  numMods = length(modelRDMs);
  corrs = zeros(length(params.num.goodSubjects),numMods);
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    subStr = params.names.subjectDir(subID);
    subDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];
    cd(subDir);

    brainRDMs = load([subDir params.names.rdmSetIn maskName '.mat']);
    brainRDM = brainRDMs.subRDM.rdm;
    % keyboard
    % construct design matrix
    X = helper_construct_designMatrix(modelRDMs,ii);
    % keyboard
    if strcmp(params.corrs.whichruns,'avg')
      [~,brainRDM] = rsa_compute_averageCvalRDMs(brainRDM,params.num.runs,params.num.conditions);
    end
    % Y = nanzscore(vectorizeRDM(brainRDM))';
    Y = nanzscore(brainRDM(:));
    % compute CPD for all model RDMs
    
    b = regress(Y,X);
    corrs(ii,:) = b;
  end

  results = struct();
  results.corrs = corrs;
  results.params = params.corrs;

  cd(grpDir);
  save(['groupAvg_' params.names.corrsOut '_' maskName '.mat'],'results');

end


function dmat = helper_construct_designMatrix(modelRDMs,ii)
  dmat = [];
  for modID = 1:length(modelRDMs)
      % dmat(:,modID) = nanzscore(vectorizeRDM(squeeze(modelRDMs(modID).rdms(ii,:,:))));
      x = squeeze(modelRDMs(modID).rdms(ii,:,:));
      x = x(:);
      dmat(:,modID) = nanzscore(x);
  end
end
