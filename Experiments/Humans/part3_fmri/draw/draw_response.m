function draw_response(w,mappingPos,mappingSize,mappingWidth,mappingCol,mappingTxt,centre,fontSize, chosenOption)
	%% DRAW_RESPONSE()
	%
	% draws key mapping only for option selected by participant
	%
	% Timo Flesch, 2018
	
	% define coordinates
	coordsLeftTxt  = [centre(1)+mappingPos(1,1)-mappingSize(1)+10,centre(2)+mappingPos(1,2)-8];
	coordsLeftBox  = [centre(1)+mappingPos(1,1)-mappingSize(1),centre(2)+mappingPos(1,2)-mappingSize(2),centre(1)+mappingPos(1,1)+mappingSize(1),centre(2)+mappingPos(1,2)+mappingSize(2)];
	coordsRightTxt = [centre(1)+mappingPos(2,1)-mappingSize(1)+10,centre(2)+mappingPos(2,2)-8];
	coordsRightBox = [centre(1)+mappingPos(2,1)-mappingSize(1),centre(2)+mappingPos(2,2)-mappingSize(2),centre(1)+mappingPos(2,1)+mappingSize(1),centre(2)+mappingPos(2,2)+mappingSize(2)];

	if(chosenOption == 1)
		% left box:
		Screen('DrawText',w,mappingTxt{1},coordsLeftTxt(1),coordsLeftTxt(2),mappingCol);
		Screen('FrameRect',w,mappingCol,coordsLeftBox,mappingWidth);
	elseif(chosenOption == 2)
		% right box:
		Screen('DrawText',w,mappingTxt{2},coordsRightTxt(1),coordsRightTxt(2),mappingCol);
		Screen('FrameRect',w,mappingCol,coordsRightBox,mappingWidth);
	end

end