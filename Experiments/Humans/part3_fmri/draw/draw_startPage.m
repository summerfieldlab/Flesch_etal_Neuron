function draw_startPage(w,params)
	%% DRAW_STARTPAGE()
	%
	% draws start page, which indicates that
	% experiment is about to begin
	%
	% Timo Flesch, 2018

	Screen('TextSize',w,params.txt.fontSize_headline2);
	Screen('Textfont',w,params.txt.fontType1);
	Screen('TextColor',w,params.col.txt_headline);
	DrawFormattedText(w,'El experimento empezará en un momento! \n Pro favor informa los investigadores cuando estés listo!','center','center');	
	Screen('Flip',w);
	KbStrokeWait();

end
