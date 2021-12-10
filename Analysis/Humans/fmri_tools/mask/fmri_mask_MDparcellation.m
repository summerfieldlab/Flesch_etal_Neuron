function fmri_mask_MDparcellation(MDROI_name,maskDir,ref_name)
  %%
  %
  % takes image of labelled MD regions and creates
  % parcellations with one image per region (based on unique identifiers)
  %
  % original image obtained from http://imaging.mrc-cbu.cam.ac.uk/imaging/MDsystem
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  if ~exist('ref_name','var')
    doReslice = 0;
  else
    doReslice =1;
  end
  % load image
  vol = spm_data_read([maskDir MDROI_name]);
  % convert
  volu = cast(vol,'uint8');
  % loop through regions and store unique mask files
  regionIDs = unique(volu(volu>0));
  for iiReg = 1:length(regionIDs)
    volReg = volu;
    volReg(volReg~=regionIDs(iiReg)) = 0;
    fname = ['md_roi_' num2str(regionIDs(iiReg)) '.nii'];
    % keyboard
    fmri_io_mat2nifti(volReg,fname,'MD region of interest',16,spm_vol(MDROI_name));
    if doReslice
      fmri_io_reslice(spm_vol(fname),spm_vol([ref_name]),1);
    end
  end
end
