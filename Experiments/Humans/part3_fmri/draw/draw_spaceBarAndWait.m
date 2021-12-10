function draw_spaceBarAndWait(w,centre,yCoords,params)
	%% DRAW_SPACEBARANDWAIT(W,YCOORDS,PARAMS)
	%
	% indicates which button to press to continue
	% ... and waits for response
	%
	% Timo Flesch, 2018

	Screen('Textfont',w,params.txt.fontType1);
	Screen('TextColor',w,params.col.txt_headline);
	Screen('TextSize',w,params.txt.fontSize_headline2);
	DrawFormattedText(w,'Pulsa la banda espaciadora para continuar.','center',centre(2)+yCoords);
	Screen('Flip',w);
	KbStrokeWait();
end
