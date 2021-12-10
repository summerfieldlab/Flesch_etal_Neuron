function draw_introPage(w,centre,imDir,params)
	%% DRAW_INTROPAGE(W,CENTRE,IMDIR,PARAMS)
	%
	% draws overview page and waits for response
	%
	% Timo Flesch, 2018
	
	Screen('Textfont',w,params.txt.fontType1);
	Screen('TextColor',w,params.col.txt_headline);
	Screen('TextSize',w,params.txt.fontSize_headline2);
	DrawFormattedText(w,'This Experiment consists of four phases:','center',centre(2)-120);
	draw_phases(w,centre,imDir,0);
	draw_spaceBarAndWait(w,centre,200,params);
		
end