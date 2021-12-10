function rdmCollection = rsa_roi_compute(roiName)
    %% rsa_roi_compute()
    %
    % computes rdms for each subject, saves results
    % at subject and group-average level
    % (now with crossvalidated ROIs)
    % Timo Flesch, 2020
    % Human Information Processing Lab
    % University of Oxford

    params = rsa_roi_params();
    n_workers = params.hpc.numWorkers;

    if exist('roiName', 'var') % if user has specified roi, overwrite params
        params.names.roiMask = roiName;
    end

    grpDir = [params.dir.inDir params.dir.subDir.GRP];

    switch params.rsa.whichruns
        case 'avg'
            rdmCollection = nan(length(params.num.goodSubjects), params.num.conditions, params.num.conditions);
        case 'cval'
            rdmCollection = nan(length(params.num.goodSubjects), params.num.conditions * params.num.runs, params.num.conditions * params.num.runs);
    end

    parfor (ii = 1:length(params.num.goodSubjects), n_workers)

        subID = params.num.goodSubjects(ii);
        % load single subject mask
        gmaskMat = fmri_io_nifti2mat([params.names.roiMask 'sub' num2str(subID) '.nii'], params.dir.maskDir, 1);
        gmaskVect = gmaskMat(:);
        gmaskVect(gmaskVect == 0) = NaN;
        gmaskIDsBrain = find(~isnan(gmaskVect));

        % navigate to subject folder
        subStr = params.names.subjectDir(subID);
        subDir = [params.dir.inDir subStr '/'];

        disp(['processing subject ' subStr]);
        spmDir = [params.dir.inDir subStr '/' params.dir.subDir.SPM];
        rsaDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];

        % load SPM.mat
        cd(spmDir);
        tmp = load(fullfile(pwd, ['../' params.dir.subDir.SPM 'SPM.mat']));
        SPM = tmp.SPM;

        % import betas, mask them appropriately
        disp('....importing betas');
        bStruct = struct();
        [bStruct.b, bStruct.events] = rsa_helper_getBetas(SPM, params.num.runs, params.num.conditions, params.num.motionregs, gmaskIDsBrain);

        bStruct.b(isnan(bStruct.b)) = 0;
        bStruct.b = reshape(bStruct.b, [size(bStruct.b, 1) / params.num.runs, params.num.runs, size(bStruct.b, 2)]);
        bStruct.events = reshape(bStruct.events, [params.num.conditions, params.num.runs]);
        bStruct.idces = gmaskIDsBrain;

        % if mahalanobis, import residuals, whiten betas
        if strcmp(params.rsa.metric, 'mahalanobis') || strcmp(params.rsa.metric, 'crossnobis') || params.rsa.whiten == 1
            disp('....importing residuals')
            cd(spmDir);
            r = rsa_helper_getResiduals(SPM, gmaskIDsBrain, 0);
            r = reshape(r, [size(r, 1) / params.num.runs, params.num.runs, size(r, 2)]);
            r(isnan(r)) = 0;
            rStruct = struct();
            rStruct.r = r;
            rStruct.idces = gmaskIDsBrain;
            disp(' .....  whitening the parameter estimates');
            bStruct.b = rsa_helper_whiten(bStruct.b, rStruct.r);
            cd(rsaDir);
        end

        % compute rdms
        switch params.rsa.whichruns
            case 'avg'
                rdmCollection(ii, :, :) = rsa_compute_rdmSet_avg(bStruct.b, params.rsa.metric);
            case 'cval'
                rdmCollection(ii, :, :) = rsa_compute_rdmSet_cval(bStruct.b, params.rsa.metric);
        end

        % navigate to output subfolder
        if ~exist(rsaDir, 'dir')
            mkdir(rsaDir);
        end

        cd(rsaDir);

        % save results (with condition labels)
        subRDM = struct();
        subRDM.rdm = squeeze(rdmCollection(ii, :, :));
        subRDM.roiName = params.names.roiMask;
        subRDM.roiIDCES = gmaskIDsBrain;
        subRDM.events = bStruct.events(:, 1);
        subRDM.subID = subID;
        parsave([params.names.rdmSetOut params.names.roiMask '.mat'], subRDM);
    end

    % navige to group level folder
    cd(params.dir.outDir);

    % ..and store group average (for visualisation)
    groupRDM = struct();
    groupRDM.rdm = squeeze(nanmean(rdmCollection, 1));
    groupRDM.roiName = params.names.roiMask;

    save(['groupAvg_' params.names.rdmSetOut params.names.roiMask '.mat'], 'groupRDM');

end

function parsave(fileName, subRDM)
    save(fileName, 'subRDM');
end
