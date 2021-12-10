function dataOut = smoothMean(dataIn,windowSize,stepSize)
%% DATAOUT = SMOOTHMEAN(DATAIN,WINDOWSIZE,STEPSIZE)
% running mean (Sort of)
% 
% (c) Timo Flesch, 2016

%% MAIN

if ~exist('stepSize')
	stepSize = 1;
end

dataOut = [];
dIDX = 1;
for ii = 1:stepSize:size(dataIn,2)-windowSize
	dataOut(:,dIDX) = nanmean(dataIn(:,ii:ii+windowSize),2);
	dIDX = dIDX+1;
end  

end 