function rsa_indscal_regressBrainRDMs_LOSO_ROI(maskName)
  %% rsa_indscal_regressBrainRDMs_LOSO_ROI()
  %
  % regresses model rdms against brain rdms
  % in a leave-one-subject-out fashion
  %
  % Timo Flesch, 2019,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford

  params = rsa_compute_setParams();

  % load pre-computed model RDMs
  % if ~exist('modelRDMs','var')
  %   modelRDMs = load(params.names.models);
  %   fns = fieldnames(modelRDMs);
  %   modelRDMs = modelRDMs.(fns{1});
  % end
  grpDir = [params.dir.inDir params.dir.subDir.GRP];

  % only include selection of models
  modelRDMs = rsa_corrs_genINDSCALModelRDMs_cval();

  n_workers = 6;

  %% do it
  numMods = length(modelRDMs);
  corrs = zeros(length(params.num.goodSubjects),numMods);
  rdms  = nan(length(params.num.goodSubjects),params.num.conditions,params.num.conditions);
  parfor (ii = 1:length(params.num.goodSubjects),n_workers)
      disp(['iteration ' num2str(ii) '/' num2str(length(params.num.goodSubjects))]);
      trainsubs = params.num.goodSubjects;
      trainsubs(ii) = [];
      x = [];
      y = [];
      %% concatenate training subs
      for jj = 1:length(trainsubs)
          subID = trainsubs(jj);
          subStr = params.names.subjectDir(subID);
          subDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];
          cd(subDir);
          brainRDMs = load([subDir params.names.rdmSetIn maskName]);
          brainRDM = brainRDMs.subRDM.rdm;
          % construct design matrix
          x_sub = helper_construct_designMatrix(modelRDMs,jj);
          x = cat(1,x,x_sub);
          y_sub = nanzscore(vectorizeRDM(brainRDM))';
          y = cat(1,y,y_sub);
      end
      %% get param estimates
      b_iter = regress(y,x);

      %% predict
      % dmat held-out sub
      testSubID = params.num.goodSubjects(ii);
      x_test = helper_construct_designMatrix(modelRDMs,ii);

      % predicted rdm
      y_pred = x_test*b_iter;
      rdm_pred = squareform(y_pred);
      % keyboard
      % average across runs
      [~,rdm_pred_f] = rsa_compute_averageCvalRDMs(rdm_pred,params.num.runs,params.num.conditions);
      % store results
      corrs(ii,:) = b_iter;
      rdms(ii,:,:) = rdm_pred_f;
  end

  results = struct();
  results.corrs = corrs;
  results.params = params;
  results.rdms = rdms;

  cd(grpDir);
  parsave(['results_indscal_rdms_' maskName],results);

end

function parsave(str,results)
    save(str,'results');
end

function dmat = helper_construct_designMatrix(modelRDMs,ii)
  dmat = [];
  for modID = 1:length(modelRDMs)
      dmat(:,modID) = nanzscore(vectorizeRDM(squeeze(modelRDMs(modID).rdms(ii,:,:))));
  end
end
