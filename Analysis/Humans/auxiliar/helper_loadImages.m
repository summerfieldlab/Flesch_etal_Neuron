function stimMat = helper_loadImages(imageFolder,stimuli,bgCol)
	if ~exist('bgCol') bgCol = 150; end

	stimMat = [];
	for ii = 1:length(stimuli)
		img = imread([imageFolder stimuli{ii}]);
		img(img==0) = bgCol; %bgcol
		stimMat = cat(4,stimMat,img);
	end
	stimMat = permute(stimMat,[4,1,2,3]);
	stimMat = flip(stimMat,1);
end
