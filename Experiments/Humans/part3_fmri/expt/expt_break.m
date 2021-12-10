function expt_break(w,breakDuration,breakText,taskChangeText,breakPos,blockID,numBlocks,isTaskChange,spaceKey)
	%% EXPT_BREAK()
	%
	% shows text and timer during breaks within runs/blocks
	%
	% Timo Flesch, 2018


	breakOnset  =     GetSecs();
	continueExp =             0;
	t_remain    = breakDuration;

	if isTaskChange
		txt = ['Congratulations, you have finished block ' num2str(blockID) ' out of ' num2str(numBlocks) '! \n'...
	'\n' breakText '\n \n' taskChangeText];
	else
		txt = ['Congratulations, you have finished block ' num2str(blockID) ' out of ' num2str(numBlocks) '! \n'...
	'\n' breakText];
	end

	while (t_remain>0 && ~continueExp)
		[~,~,keyCode] = KbCheck();

		if keyCode(spaceKey)
        	continueExp = 1;
	    end

	    Screen('TextSize',w,33);
	    Screen('TextColor',w,[1 1 1]);
	    DrawFormattedText(w,'BREAK','center',150,[1 1 1]);
	    if t_remain>5
	        Screen('TextSize',w,20);
	        Screen('TextColor',w,[1 1 1]);

	        DrawFormattedText(w,txt,'center','center');

	        Screen('TextSize',w,20);
	        Screen('TextColor',w,[0 1 0]);
	    else
	        Screen('TextSize',w,25);
	        Screen('TextColor',w,[1 0 0]);
	        breakText = ['It''s time to get ready!\n\n'];
	        DrawFormattedText(w,breakText,'center','center');
	    end

	    minutesRemain = floor(t_remain/60);
	    secondsRemain = mod(t_remain,60);

	    if minutesRemain == 1
	        timerText = ['The break ends in ' num2str(minutesRemain) ' minute and ' num2str(secondsRemain) ' seconds'];
	    else
	        timerText = ['The break ends in ' num2str(minutesRemain) ' minutes and ' num2str(secondsRemain) ' seconds'];
	    end

	    DrawFormattedText(w,timerText,'center',650);
	    Screen('Flip',w);

	    t_remain = breakDuration-floor(GetSecs-breakOnset);

	end
end
