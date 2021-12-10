function wrapper_rsa_roi_disp_svd()
  %%
  %
  % computes ROI-based RDMs
  % using my own pipeline, and tdt for comparison
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  %
  % roiMasks = {'r_mask_wfu_BA46_left','r_mask_wfu_BA46_right'};

% roiMasks = {'ROI_LEFTIFG_x=-4.450000e+01_y=7_z=3.150000e+01_153voxels_Sphere12', ...
%     'ROI_LEFTLINGUAL_x=-1.650000e+01_y=-9.450000e+01_z=3.500000e+00_91voxels_Sphere12', ...
%     'ROI_RIGHTANGULARGYRUS_x=36_y=-49_z=42_170voxels_Sphere12', ...
%     'ROI_RIGHTIFG_x=43_y=14_z=1.750000e+01_161voxels_Sphere12', ...
%     'ROI_RIGHTLINGUAL_x=8_y=-8.750000e+01_z=0_50voxels_Sphere12'};
  
roiMasks = {'leftEVC_all7mods_mod5_','rightEVC_all7mods_mod5_','rightDLPFC_all7mods_mod3_','rightMCC_all7mods_mod3_','rightPAR_all7mods_mod3_'};
roiLabels = {'leftEVC','rightEVC','rightDLPFC','MCC','rightParietal'};

  nDims = 50;

  for maskID = 1:length(roiMasks)
    mask = roiMasks{maskID};
    disp(['ROI RSA with mask ' mask]);
    rsa_disp_showCorrs_ROI_SVD(nDims, mask,{'(i) grid','(ii) grid-rotated','(iii) orthogonal','(iv) parallel', '(v) branchiness','(vi) leafiness','(vii) diagonal'},roiLabels{maskID})

    disp(['...done']);
  end
