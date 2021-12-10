function roi = fmri_mask_mni2roi(coords,radius,img)
  %% fmri_mask_mni2roi(vols,coords,radius)
  %
  % creates ROI based on mni coordinates and returns
  % linear indices of voxels within the ROI
  %
  % Timo Flesch, 2018,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford

  xY = struct();
  xY.def = 'sphere';
  xY.rej = {'cluster'};
  xY.xyz = coords';
  xY.spec = radius;
  xY.str = sprintf('%1.1fmm sphere',radius);
  % Extract ROI
  [~,~,roi] = spm_ROI(xY,img);



end
