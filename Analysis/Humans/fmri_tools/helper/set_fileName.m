function fName = set_fileName(id)
  %% MK_FILENAME(ID)
  % generates file name that matches
  % folder structure of fMRI data
  %
  % INPUTS
  % id:      numerical subject indentifier
  %
  % Timo Flesch, 2018,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford
  
  fName = sprintf('TIMO%03d',id);
end
