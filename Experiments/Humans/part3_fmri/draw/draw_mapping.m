function draw_mapping(w,mappingPos,mappingSize,mappingWidth,mappingCol,mappingTxt,centre,fontSize)
	%% DRAW_MAPPING()
	%
	% draws key mapping left and right of stimulus
	%
	% Timo Flesch, 2018
	
	% define coordinates
	coordsLeftTxt  = [centre(1)+mappingPos(1,1)-mappingSize(1)+10,centre(2)+mappingPos(1,2)-8];
	% coordsLeftBox  = [centre(1)+mappingPos(1,1)-mappingSize(1),centre(2)+mappingPos(1,2)-mappingSize(2),centre(1)+mappingPos(1,1)+mappingSize(1),centre(2)+mappingPos(1,2)+mappingSize(2)];
	coordsRightTxt = [centre(1)+mappingPos(2,1)-mappingSize(1)+10,centre(2)+mappingPos(2,2)-8];
	% coordsRightBox = [centre(1)+mappingPos(2,1)-mappingSize(1),centre(2)+mappingPos(2,2)-mappingSize(2),centre(1)+mappingPos(2,1)+mappingSize(1),centre(2)+mappingPos(2,2)+mappingSize(2)];

	% left box:
	Screen('TextSize',w,fontSize);
	Screen('DrawText',w,mappingTxt{1},coordsLeftTxt(1),coordsLeftTxt(2),mappingCol);
	
	% Screen('FrameRect',w,mappingCol,coordsLeftBox,mappingWidth);


	% right box:
	Screen('TextSize',w,fontSize);
	Screen('DrawText',w,mappingTxt{2},coordsRightTxt(1),coordsRightTxt(2),mappingCol);
	% Screen('FrameRect',w,mappingCol,coordsRightBox,mappingWidth);

end