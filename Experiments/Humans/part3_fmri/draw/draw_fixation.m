function draw_fixation(w,fixSize,fixWidth,fixColour,centre)
	%% DRAW_FIXATION(W,FIXSIZE,FIXWIDTH,FIXCOLOUR,FIXDURATION,CENTRE)
	%
	% draws fixation cross
	%
	% Timo Flesch, 2018

	coords = [-fixSize(1),fixSize(1),0,0;0,0,-fixSize(2),fixSize(2)];
	Screen('DrawLines', w, coords, fixWidth, fixColour, centre);
end