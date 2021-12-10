function wrapper_rsa_compressangle()
    %% wrapper_rsa_compressangle()
    %
    % computes rdms within ROIs
    %
    % Timo Flesch, 2020,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford


    % roilist = {'ROI_LEFTIFG_x=-4.450000e+01_y=7_z=3.150000e+01_153voxels_Sphere12', ...
    %     'ROI_LEFTLINGUAL_x=-1.650000e+01_y=-9.450000e+01_z=3.500000e+00_91voxels_Sphere12', ...
    %     'ROI_RIGHTANGULARGYRUS_x=36_y=-49_z=42_170voxels_Sphere12', ...
    %     'ROI_RIGHTIFG_x=43_y=14_z=1.750000e+01_161voxels_Sphere12', ...
    %     'ROI_RIGHTLINGUAL_x=8_y=-8.750000e+01_z=0_50voxels_Sphere12'};
    % roilist = {'leftEVC_mod1_','rightEVC_mod1_', 'rightIFG_mod3_','rightPAR_mod3_'}
    roilist = {'leftEVC_all7mods_mod5_','rightEVC_all7mods_mod5_','rightDLPFC_all7mods_mod3_','rightMCC_all7mods_mod3_','rightPAR_all7mods_mod3_'};

    gcp = parpool(6)
    for maskID = 1:length(roilist)
       mask = roilist{maskID};
       disp(['PARAMMOD REGRESSION with mask ' mask]);
       rsa_corrs_regressBrainRDMs_ROI_CompressAngle([mask '.mat']);
       disp(['...done']);
     end
     delete(gcp)
