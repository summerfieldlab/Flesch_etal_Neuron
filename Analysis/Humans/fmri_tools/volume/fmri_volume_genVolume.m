function volMat = fmri_volume_genVolume(dim,coords,intensities)
  %% fmri_volume_genVolume()
  %
  % given a set of coordinates and intensities,
  % this function generates a volume
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  volMat = nan(dim);
  for ii = 1:size(coords,2)
    volMat(coords(1,ii),coords(2,ii),coords(3,ii)) = intensities(ii);
  end

end
