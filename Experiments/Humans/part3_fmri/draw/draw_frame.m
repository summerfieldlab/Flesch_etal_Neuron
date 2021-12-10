function draw_frame(w,frameSize,frameWidth,frameColour,centre)
	%% DRAW_FRAME(W,CTXTSIZE,CTXTWIDTH,CTXTCOLOURS,CENTRE)
	%
	% draws frame
	%
	% Timo Flesch, 2018

	coords = [centre(1)-frameSize(1),centre(2)-frameSize(2),centre(1)+frameSize(1),centre(2)+frameSize(2)];
	Screen('FrameRect',w,frameColour,coords,frameWidth);

end