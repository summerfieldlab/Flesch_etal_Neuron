function fmri_helper_changeSPMpaths(oldPath,newPath)
  %% fmri_helper_changeSPMpaths()
  %
  % If moved to different directory, absolute paths in
  % SPM files need to be updated
  %
  % This function takes care of that, for each individual subject
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  nSubs = 14;

  for subID = 1:nSubs
    subStr = sprintf('TIMO%03d',subID);
    spmDir = [newPath 'exp_3_granada_fmri/results/glm_dec_3/' subStr '/dmats/' ];
    load([spmDir 'SPM.mat']);
    SPM = spm_changepath(SPM,oldPath,newPath);
    save([spmDir 'SPM.mat'],'SPM');
  end
end
