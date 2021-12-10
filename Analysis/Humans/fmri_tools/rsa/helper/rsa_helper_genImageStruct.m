function imgStruct = rsa_helper_genImageStruct(imgs)
  %% rsa_helper_genImageStruct(imgs)
  %
  % generates struct of images
  % used to label axes of RDMs in visualisation step
  %
  % Timo Flesch, 2018,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford

  imgs = cast(cat(1,imgs(50:-1:26,:,:,:),imgs(25:-1:1,:,:,:)),'double')./255;
  
  imgStruct = struct();
  imgStruct.images = struct();
  for ii = 1:size(imgs,1)
    imgStruct.images(ii).image = squeeze(imgs(ii,:,:,:));
  end

end
