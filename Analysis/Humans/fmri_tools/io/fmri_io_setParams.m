function params = fmri_io_setParams()
  %% fmri_io_setParams()
  %
  % set params for spm/matlab file IO
  %
  % Timo Flesch, 2019

  params = struct();


  % see spm_vol for details
  params.vol = struct();
  params.vol.fname   = '~/todo.nii';
  params.vol.dim     =   [45 55 45];
  params.vol.dt      =        [2 0]; % uint8
  params.vol.pinfo   =  [1; 0; 352];
  params.vol.mat     =  [-3.5000, 0, 0, 81.5000; 0, 3.5000,  0, -115.5000; 0, 0, 3.5000, -73.5000; 0, 0, 0, 1.0000];
  params.vol.n       =        [1,1];
  params.vol.descrip =  'todo todo';
  params.vol.private =           [];

end
