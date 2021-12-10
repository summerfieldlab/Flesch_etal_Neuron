function xData_reduced = rsa_helper_reduceDimensionality(xData,nDims)
  %% rsa_helper_reduceDimensionality(xData,nDims)
  %
  % performs SVD on data and keeps only the first n singular values
  % for reconstruction of the full data matrix
  % XData: [n_obs,n_conds]
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford
  xData = xData-mean(xData,1);
  [U,S,V] = svd(xData);
  S_reduced = 0*S;
  for ii = 1:nDims
    S_reduced(ii,ii) = S(ii,ii);
  end
  
  xData_reduced = U*S_reduced*V';

end
