function results = fmri_taskswitching_extract_betas()
    %% fmri_taskswitching_extract_betas()
    % 
    % extracts single subject betas from ROIs 
    % 
    % Timo Flesch, 2020
    % University of Oxford 

    %% parameters 
    roi_masks = {'clustermask_switchstay_BA7.nii', ...
                'clustermask_switchstay_leftIFG.nii', ...
                'clustermask_switchstay_superiormedial.nii'};
    % roi_masks = {'ROI_switchstay_171voxels_Sphere12_BA7.nii', ...
    %             'ROI_switchstay_171voxels_Sphere12_leftIFG.nii', ...
    %             'ROI_switchstay_171voxels_Sphere12_superiormedial.nii'};
    roiDir = 'project/results_paper/masks/';
    results.s_ids = 1:32;
    results.badsub = [19,28];
    results.s_ids(results.badsub) = [];

    conDir = 'project/results_paper/glm_1_switchStay_cueLock/';
    sDir   =       @(subID)sprintf('TIMO%03d/',subID);

    results.betas = zeros(length(roi_masks),length(results.s_ids));

    for ii = 1:length(roi_masks)
        roiMat = fmri_io_nifti2mat(roi_masks{ii}, roiDir);
        roiMat = roiMat(:);
        for jj = 1:length(results.s_ids)
            vxlMat = fmri_io_nifti2mat('con_0002.nii', [conDir sDir(results.s_ids(jj)) 'dmats/' ]);
            vxlMat = vxlMat(:);
            results.betas(ii,jj) =mean(vxlMat(roiMat==1 & ~isnan(vxlMat)));
        end
    end
    
end