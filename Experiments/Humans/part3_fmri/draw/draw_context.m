function draw_context(w,ctxtSize,ctxtWidth,ctxtColour,centre)
	%% DRAW_CONTEXT(W,CTXTSIZE,CTXTWIDTH,CTXTCOLOURS,CENTRE)
	%
	% draws contextual cue as coloured rectangle 
	%
	% Timo Flesch, 2018

	coords = [centre(1)-ctxtSize(1),centre(2)-ctxtSize(2),centre(1)+ctxtSize(1),centre(2)+ctxtSize(2)];
	Screen('FrameRect',w,ctxtColour,coords,ctxtWidth);

end