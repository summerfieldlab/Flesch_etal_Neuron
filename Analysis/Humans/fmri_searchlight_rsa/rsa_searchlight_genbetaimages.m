function rsa_searchlight_genbetaimages()
    %
    % generates grouplevel-masked volumes of regression coefficients
    % (to include only voxels that have values for all participants)
    %
    % Timo Flesch, 2019
    % Human Information Processing Lab
    % University of Oxford

    params = rsa_searchlight_params();
    % load grouplevel mask
    gmaskMat = fmri_io_nifti2mat([params.names.groupMask '.nii'], params.dir.maskDir);
    gmaskVect = gmaskMat(:);

    allBetas = [];
    % generate images
    params = rsa_searchlight_params();

    for (ii = 1:length(params.num.goodSubjects))
        subID = params.num.goodSubjects(ii);
        disp(['processing subject ' num2str(subID)])
        % load mask indices
        subStr = sprintf('TIMO%03d', subID);

        maskIDs = load([params.dir.inDir subStr '/' params.dir.subDir.RDM params.names.rdmSetIn]);

        maskIDs = maskIDs.results.indices;

        % load correlation coefficients

        corrDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM params.dir.subDir.rsabetas];
        load([corrDir params.names.betasOut '_set_' params.names.modelset '_sub_' num2str(subID)]);

        betas = results.betas;
        % loop through models
        for modID = 1:size(betas, 1)
            % get mask indices of single subject betas
            [x, y, z] = ind2sub(size(gmaskMat), maskIDs);
            XYZ = [x y z]';
            % generate volume from indices and tau values
            volMat = fmri_volume_genVolume(size(gmaskMat), XYZ, betas(modID, :));
            % apply group-level mask
            volMat(isnan(gmaskVect)) = NaN;
            % add to big matrix
            allBetas(ii, modID, :, :, :) = volMat;

            % write volume
            fName = fullfile(params.dir.outDir, ['betas_searchlight_' params.names.modelset '_mod' num2str(modID) '_sub' num2str(subID) '.nii']);
            fmri_io_mat2nifti(volMat, fName, 'rdm model betas (regression)', 16);
        end

    end

    % average big matrix and save group-level tau volumes
    for modID = 1:size(betas, 1)
        grpTaus = squeeze(mean(allBetas(:, modID, :, :, :), 1));
        fName = fullfile(params.dir.outDir, ['betas_searchlight_' params.names.modelset '_mod' num2str(modID) 'groupAvg.nii']);
        fmri_io_mat2nifti(grpTaus, fName, 'rdm model betas (regression)', 16);
    end

    % bonus: store computed tau matrix
    betas = allBetas;
    fName = fullfile(params.dir.outDir, ['betas_searchlight_' params.names.modelset '_allsubs_masked']);
    save(fName, 'betas');
end
