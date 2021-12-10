function draw_progressPage(sessID,sessName,w,centre,imDir,params,sessDuration)
	%% DRAW_PROGRESSPAGE(SESSID,SESSNAME,W,CENTRE,IMDIR,PARAMS)
	%
	% draws progress page and waits for response
	%
	% Timo Flesch, 2018

	if ~exist('sessDuration')
		sessDuration = 0;
	end

	Screen('Textfont',w,params.txt.fontType1);
	Screen('TextColor',w,params.col.txt_headline);
	Screen('TextSize',w,params.txt.fontSize_headline2);
	DrawFormattedText(w,['Vamos a empezar con' num2str(sessName) '  \n ' ' La duración será de unos ' num2str(sessDuration) ' minutos, sin descansos'],'center',centre(2)-180);
	draw_phases(w,centre,imDir,sessID);
	draw_spaceBarAndWait(w,centre,200,params);
end
