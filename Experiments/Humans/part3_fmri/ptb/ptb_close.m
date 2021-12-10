function ptb_close()
	%% PTB_CLOSE()
	% closes PTB session and tidies up
	%
	% Timo Flesch, 2018
	
	% get cursor back
	ShowCursor;
	% get keyboard back
	ListenChar(0);
	% reset priority
	Priority(0);
	% and close full screen mode
	Screen('CloseAll');
	clear mex

end
