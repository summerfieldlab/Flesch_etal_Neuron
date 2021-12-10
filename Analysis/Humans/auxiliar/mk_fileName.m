function fName = mk_fileName(id,tPhase)
  %% MK_FILENAME(ID,TPHASE)
  % generates file name that matches
  % folder structure of fMRI data
  %
  % INPUTS
  % id:      numerical subject indentifier
  % tPhase:  'arena', 'training', 'refresher' or 'scan'
  %
  % Timo Flesch, 2018
  % Human Information Processing Lab
  % University of Oxford

  if (~exist('tPhase','var'))
    tPhase = 'arena';
  end

  if id <10
    fName = ['TIMO00' num2str(id) '_' tPhase];
  else
    fName = ['TIMO0' num2str(id) '_' tPhase];
  end
end
