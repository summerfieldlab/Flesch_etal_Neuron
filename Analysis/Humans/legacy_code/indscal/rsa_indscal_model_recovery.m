function results = rsa_indscal_model_recovery(modelRDMs, rdmToRecover)
    %% rsa_indscal_model_recovery
    %
    % tests whether the choice of model RDMs affects the ability to
    % recover certain representational structures
    % In other words, allows me to test that my indscal approach is bias free.
    %
    % approach: for each rdm in rdmsToRecover: try to predict with modelRDMs,
    % then generate predicted rdm with betas*modelRDMs,
    % then perform mds on rdmToRecover as well as predictedRDM, compare the two.
    % do also compute rank correlation between the two.
    % Do this for various choices of modelRDMs
    %
    % participants are simulated by applying a small amount of uniformly random
    % noise to the rdmsToRecover
    %
    % PARAMETERS
    % modelRDMs: rdms used as regressors for LOSO approach
    % rdmsToRecover: "ground truth" rdms I try to predict with above regression model
    %
    % Timo Flesch, 2019,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford

    n_subs = 20    % number of hypothetical subjects
    std_noise = 0.0001; % very low. 0.01 low, 20 high % std of random noise

    n_workers = 6 % number of workers for parallel processing

    params = rsa_corrs_setParams()
    grpDir = [params.dir.inDir params.dir.subDir.GRP];
    % load modelRDMs
    % modelRDMs = load(params.names.models);
    % fns = fieldnames(modelRDMs);
    % modelRDMs = modelRDMs.(fns{1});



    % initialise arrays
    numMods = length(modelRDMs);
    corrs = zeros(n_subs,numMods);
    rdms  = nan(n_subs,params.num.conditions,params.num.conditions);

    % generate subject-specific "ground truth" RDMs
    rdm_gt = squeeze(modelRDMs(rdmToRecover).rdms(1,:,:));
    brainRDMs = [];
    nConds = params.num.conditions;
    nRuns = params.num.runs;
    for ii = 1:n_subs
        brainRDMs(ii,:,:) = rdm_gt + std_noise.*randn(nRuns*nConds,nRuns*nConds);
        for jjRun = 1:nConds:(nConds*nRuns)
          brainRDM(jjRun:jjRun+nConds-1,jjRun:jjRun+nConds-1) = NaN;
        end
    end

    % now only include models to test
    % modelRDMs = modelRDMs(modelRDMs);
    gtcorrs = [];
    gtrdms = [];
    % start the fun
    parfor (ii = 1:n_subs,n_workers)
        disp(['iteration ' num2str(ii) '/' num2str(n_subs)]);
        trainsubs = 1:n_subs
        trainsubs(ii) = [];
        x = [];
        y = [];
        %% concatenate training subs
        for jj = 1:length(trainsubs)
            subID = trainsubs(jj);
            brainRDM = squeeze(brainRDMs(subID,:,:));
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
        subs = 1:n_subs;
        testSubID = subs(ii);
        x_test = helper_construct_designMatrix(modelRDMs,ii);

        % predicted rdm
        y_pred = x_test*b_iter;
        rdm_pred = squareform(y_pred);
        % keyboard
        % average across runs
        [~,rdm_pred_f] = rsa_compute_averageCvalRDMs(rdm_pred,nRuns,nConds);
        % store results
        corrs(ii,:) = b_iter;
        rdms(ii,:,:) = rdm_pred_f;
        [~,gtrdms(ii,:,:)] = rsa_compute_averageCvalRDMs(squeeze(brainRDMs(ii,:,:)),nRuns,nConds);

        % compute correlation between gt and predicted RDM
        gtcorrs(ii,:) = rankCorr_Kendall_taua(vectorizeRDM(rdm_pred),vectorizeRDM(squeeze(brainRDMs(ii,:,:))));
    end
    results = struct();
    results.corrs = corrs;
    results.params = params.corrs;
    results.rdms = rdms;
    results.gtrdms = gtrdms;
    results.gtcorrs = gtcorrs;

    cd(grpDir);
    % parsave(['simulation_indscal_onlyslmats_toModel_' num2str(rdmToRecover)],results);
    parsave(['simulation_indscal_verylownoisegt_SLrdms_toModel_' num2str(rdmToRecover)],results);
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
