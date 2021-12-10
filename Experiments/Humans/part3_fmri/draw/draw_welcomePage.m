function draw_welcomePage(w,centre,params)
	%% DRAW_WELCOMEPAGE(W,CENTRE,PARAMS)
	%
	% draws welcome page and waits for response
	%
	% Timo Flesch, 2018

	Screen('TextSize',w,params.txt.fontSize_headline1);
	Screen('Textfont',w,params.txt.fontType1);
	Screen('TextColor',w,params.col.txt_headline);
	DrawFormattedText(w,'Welcome!','center','center');
	Screen('TextSize',w,params.txt.fontSize_headline2);
	draw_spaceBarAndWait(w,centre,80,params);
end
