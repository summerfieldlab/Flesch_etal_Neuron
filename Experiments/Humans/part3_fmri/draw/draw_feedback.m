function draw_feedback(w,responseSide,conditionReward,isCorrect, params,centre)
	%% DRAW_FEEDBACK()
	%
	% draws feedback for correct, incorrect or missed trials
	%
	% Timo Flesch,2018

	% set coordinates
	coordsTxtTimeout = [centre(1)-100,centre(2)-5];
	coordsLeftTxt  = [centre(1)+params.fdb.pos(1,1)-params.fdb.size(1)+15,centre(2)+params.fdb.pos(1,2)-20];
	coordsLeftBox  = [centre(1)+params.fdb.pos(1,1)-params.fdb.size(1),centre(2)+params.fdb.pos(1,2)-params.fdb.size(2),centre(1)+params.fdb.pos(1,1)+params.fdb.size(1),centre(2)+params.fdb.pos(1,2)+params.fdb.size(2)];
	coordsRightTxt = [centre(1)+params.fdb.pos(2,1)-params.fdb.size(1)+15,centre(2)+params.fdb.pos(2,2)-20];
	coordsRightBox = [centre(1)+params.fdb.pos(2,1)-params.fdb.size(1),centre(2)+params.fdb.pos(2,2)-params.fdb.size(2),centre(1)+params.fdb.pos(2,1)+params.fdb.size(1),centre(2)+params.fdb.pos(2,2)+params.fdb.size(2)];

	% set box colour and contents depending on key mapping
	% if (isCorrect)
	% 	chosenOptionColour = params.col.resp_correct;
	% else
	% 	chosenOptionColour = params.col.resp_incorrect;
	% end

	rewardStr = lib_numToSignedString(conditionReward);

	if conditionReward > 0
		rewCol = params.col.resp_correct;
	elseif conditionReward < 0
		rewCol = params.col.resp_incorrect;
	elseif conditionReward == 0
		rewCol = [0,0,0];
	end


	if (params.keys.mapping == 0) % left accept, right reject
		boxItems = {rewardStr,'0'};
		itemCols = {rewCol, params.col.resp_default}

	elseif(params.keys.mapping == 1)  % left reject, right accept
		boxItems = {'0',rewardStr};
		itemCols = {params.col.resp_default, rewCol};
	end


	% handle response
	if (isnan(responseSide))
		% time out
		Screen('TextSize',w,params.txt.fontSize_timeout);
		Screen('DrawText',w,params.to.txt{1},coordsTxtTimeout(1),coordsTxtTimeout(2),params.col.resp_timeOut);
	else
		draw_fixation(w,params.fix.size,params.fix.width,params.col.fix,centre);
		Screen('TextSize',w,params.txt.fontSize_feedback);
		Screen('DrawText',w,boxItems{1},coordsLeftTxt(1),coordsLeftTxt(2),itemCols{1});
		Screen('DrawText',w,boxItems{2},coordsRightTxt(1),coordsRightTxt(2),itemCols{2});
		if (responseSide == 1) % left
			% left box:
			% Screen('DrawText',w,boxItems{1},coordsLeftTxt(1),coordsLeftTxt(2),chosenOptionColour);
			% Screen('FrameRect',w,chosenOptionColour,coordsLeftBox,params.mapping.width);
			Screen('FrameRect',w,params.col.resp_default,coordsLeftBox,params.mapping.width);
		elseif (responseSide == 2) % right
			% right box:

			% Screen('DrawText',w,boxItems{2},coordsRightTxt(1),coordsRightTxt(2),chosenOptionColour);
			% Screen('FrameRect',w,chosenOptionColour,coordsRightBox,params.mapping.width);

			Screen('FrameRect',w,params.col.resp_default,coordsRightBox,params.mapping.width);

		end
	end
end
