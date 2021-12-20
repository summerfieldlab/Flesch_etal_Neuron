function results = rsa_roi_glmfit(maskName, whichrdms,modelname)
    %% rsa_roi_glmfit()
    %
    % regresses brain rdms against selection of model rdms
    % and computes log-likelihood, BIC and log model evidence
    %
    % Timo Flesch, 2021
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford

    if ~exist('modelname')
      modelname='fullmodel';
    end
    if ~exist('whichrdms')
      whichrdms = 1:7;
    end

    params = rsa_roi_params();

    grpDir = [params.dir.inDir params.dir.subDir.GRP];

    modelRDMs = rsa_searchlight_genmodels();
    % only include selection of models
    modelRDMs = modelRDMs(whichrdms);

    %% do it
    numMods = length(modelRDMs);
    corrs = zeros(length(params.num.goodSubjects), numMods);
    lls = zeros(length(params.num.goodSubjects), 1);
    bics = zeros(length(params.num.goodSubjects), 1);
    lmes = zeros(length(params.num.goodSubjects), 1);

    for (ii = 1:length(params.num.goodSubjects))
        subID = params.num.goodSubjects(ii);
        subStr = params.names.subjectDir(subID);
        subDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];
        cd(subDir);

        brainRDMs = load([subDir params.names.rdmSetIn maskName '.mat']);
        brainRDM = brainRDMs.subRDM.rdm;
%         [~,brainRDM] = rsa_compute_averageCvalRDMs(brainRDM,6,50);
        % construct design matrix
        X = helper_construct_designMatrix(modelRDMs, ii);

        if strcmp(params.corrs.whichruns, 'avg')
            [~, brainRDM] = rsa_compute_averageCvalRDMs(brainRDM, params.num.runs, params.num.conditions);
        end

        y = nanzscore(vectorizeRDM(brainRDM))';
        % perform linear regression
        [b, ~, stats] = glmfit(X, y, 'normal', 'link', 'identity');
        y_ = glmval(b, X, 'identity');
        % standard deviation of gaussian:
        sigma = sqrt(nanmean(stats.resid .^ 2));
%         keyboard
        lls(ii,:) = nansum(log(normpdf(y, y_, sigma)));
        bics(ii,:) = compute_BIC(lls(ii,:),length(modelRDMs),length(y));
        lmes(ii,:) = compute_LogEvidence(bics(ii,:));

        corrs(ii, :) = b(2:end);
    end

    results = struct();
    results.corrs = corrs;
    results.lls = lls;
    results.bics = bics;
    results.lmes = lmes;

    cd(params.dir.outDir);
    save(['glmfit_grp_model' modelname '_' params.names.betasOut '_' maskName '.mat'], 'results');

end

function dmat = helper_construct_designMatrix(modelRDMs, ii)
    dmat = [];

    for modID = 1:length(modelRDMs)
        modelRDM = squeeze(modelRDMs(modID).rdms(ii, :, :));
%         [~,modelRDM] = rsa_compute_averageCvalRDMs(modelRDM,6,50);
        dmat(:, modID) = nanzscore(vectorizeRDM(modelRDM));
    end

end


