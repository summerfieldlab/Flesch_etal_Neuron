function rsa_roi_svd_regressmodels(maskName, nDims)
    %% rsa_roi_svd_regressmodels()
    %
    % regresses model rdms against brain rdms
    %
    % Timo Flesch, 2019,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford

    params = rsa_roi_params();

    grpDir = [params.dir.inDir params.dir.subDir.GRP];

    % only include selection of models
    modelRDMs = rsa_searchlight_genmodels();

    %% do it
    numMods = length(modelRDMs);
    corrs = zeros(length(params.num.goodSubjects), numMods);

    for (ii = 1:length(params.num.goodSubjects))
        subID = params.num.goodSubjects(ii);
        subStr = params.names.subjectDir(subID);
        subDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];
        cd(subDir);
        brainRDMs = load([subDir params.names.rdmSetIn '_' num2str(nDims) 'D_' maskName '.mat']);
        brainRDM = brainRDMs.subRDM.rdm;
        
        % construct design matrix
        X = helper_construct_designMatrix(modelRDMs, ii);

        if strcmp(params.rsa.whichruns, 'avg')
            [~, brainRDM] = rsa_compute_averageCvalRDMs(brainRDM, params.num.runs, params.num.conditions);
        end

        Y = nanzscore(vectorizeRDM(brainRDM))';
        % compute CPD for all model RDMs
        b = regress(Y, X);
        corrs(ii, :) = b;
    end

    results = struct();
    results.corrs = corrs;
    results.params = params;

    cd(params.dir.outDir);
    save(['groupAvg_' params.names.betasOut '_regress' '_' num2str(nDims) 'D_' maskName '.mat'], 'results');

end

function dmat = helper_construct_designMatrix(modelRDMs, ii)
    dmat = [];

    for modID = 1:length(modelRDMs)
        dmat(:, modID) = nanzscore(vectorizeRDM(squeeze(modelRDMs(modID).rdms(ii, :, :))));
    end

end
