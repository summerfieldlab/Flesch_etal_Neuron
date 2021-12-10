function rsa_searchlight_regressmodels()
    %% rsa_searchlight_regressmodels()
    %
    % computes rdms using a spherical searchlight approach and performs regression
    % on candidate model RDMs
    %
    % Timo Flesch, 2020
    % Human Information Processing Lab
    % University of Oxford
    
    params = rsa_searchlight_params();
    % group-level whole brain mask (generated with gen group mask script)
    gmaskMat = fmri_io_nifti2mat([params.names.groupMask '.nii'], params.dir.maskDir);
    gmaskVect = gmaskMat(:);
    gmaskIDsBrain = find(~isnan(gmaskVect));
    grpDir = [params.dir.inDir params.dir.subDir.GRP];

    % generate candidate models
    modelRDMs = rsa_searchlight_genmodels()

    %% recruit an army of minions!
    if params.hpc.parallelise
        parpool(params.hpc.numWorkers);
    end

    parfor (ii = 1:length(params.num.goodSubjects), params.hpc.numWorkers)
        warning('off','all');
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
        SPM = load(fullfile(pwd, ['../' params.dir.subDir.SPM 'SPM.mat']));
        SPM = SPM.SPM;

        % import all betas
        bStruct = struct();
        [bStruct.b, bStruct.events] = rsa_helper_getBetas(SPM, params.num.runs, params.num.conditions, params.num.motionregs);
        bStruct.b = reshape(bStruct.b, [prod(size(gmaskMat)), size(bStruct.b, 4)]);
        bStruct.b = reshape(bStruct.b, [size(bStruct.b, 1), params.num.conditions, params.num.runs]);
        bStruct.events = reshape(bStruct.events, [params.num.conditions, params.num.runs]);

        X = helper_construct_designMatrix(modelRDMs, ii);
        rdmSet = [];

        for sphereID = 1:length(gmaskIDsBrain)

            % obtain coordinates of centroid
            [x, y, z] = ind2sub(size(gmaskMat), gmaskIDsBrain(sphereID));
            % create spherical mask
            [sphereIDs, ~] = fmri_mask_genSphericalMask([x, y, z], params.rsa.radius, gmaskMat);

            % mask betas with sphere, only use masked values
            betas = squeeze(bStruct.b(sphereIDs, :, :));
            betas = permute(betas, [2, 3, 1]);

            rdm = rsa_compute_rdmSet_cval(betas, params.rsa.metric);

            % centre the dependent variable
            Y = nanzscore(vectorizeRDM(rdm))';

            % perform regression
            b = regress(Y, X);
            corrs(:, sphereID) = b;
            [~, rdmSet(sphereID, :, :)] = rsa_compute_averageCvalRDMs(rdm, params.num.runs, params.num.conditions);

        end

        results = struct();
        results.betas = corrs;
        results.params = params;

        if ~exist('rsaDir', 'dir')
            mkdir(rsaDir);
        end

        cd(rsaDir);

        % save results
        subRDM = struct();
        subRDM.rdms = rdmSet;
        subRDM.events = bStruct.events(:, 1);
        subRDM.subID = subID;
        subRDM.indices = gmaskIDsBrain;
        helper_parsave([params.names.rdmSetOut], subRDM);

        if ~exist('params.dir.subDir.rsabetas', 'dir')
            mkdir(params.dir.subDir.rsabetas);
        end

        cd(params.dir.subDir.rsabetas);
        helper_parsave([params.names.betasOut '_set_' params.names.modelset '_sub_' num2str(subID)], results);

    end

    if params.hpc.parallelise
        delete(gcp);
    end


end

function helper_parsave(fName, results)
    save(fName, 'results');
end

function dmat = helper_construct_designMatrix(modelRDMs, ii)
    dmat = [];

    for modID = 1:length(modelRDMs)
        dmat(:, modID) = nanzscore(vectorizeRDM(squeeze(modelRDMs(modID).rdms(ii, :, :))));
    end

end
