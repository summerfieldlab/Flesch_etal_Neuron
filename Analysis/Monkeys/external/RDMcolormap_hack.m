function cols=RDMcolormap_hack(nCols)
% this function provides a convenient colormap for visualizing
% dissimilarity matrices. 
%__________________________________________________________________________
% Copyright (C) 2012 Medical Research Council
%
% slightly hacked by Timo, 2017
% should now produce beautiful gradations of leafiness

if ~exist('nCols')
    nCols = 6;
end

anchorCols=[40,20,0
            68,255 10]./255;

anchorCols_hsv=rgb2hsv(anchorCols);
incVweight=1;
anchorCols_hsv(:,3)=(1-incVweight)*anchorCols_hsv(:,3)+incVweight*linspace(0.5,1,size(anchorCols,1))';

brightness(anchorCols);
anchorCols=hsv2rgb(anchorCols_hsv);

cols=colorScale(anchorCols,nCols);
cols1=cols;

