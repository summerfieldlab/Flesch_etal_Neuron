function results = task_refresher(params,expt)
	%% TASK_PRE_BEHAV(PARAMS,EXPT)
	%
	% defines first training phase of experiment and iterates through trials
	%
	% Timo Flesch, 2018

	results = struct();
	%% 1. load tree images
	[treeImages,hasFailed] = lib_loadRGBAImages(expt.treeIDs,params.io.stims);
	if hasFailed
		return;
	end

	%% 2. run experiment
	try
		% create session
		ptb = ptb_open(params);

		% log start time
		results.time.startTime = datestr(now);

		% show instructions
		% expt_instr('general',ptb,params);
    	expt_instr('refresher',ptb,params);

		% Display blank (grey) screen
    	Screen(ptb.w,'FillRect',params.col.bg); % blank screen
    	results.time.exp_start = Screen(ptb.w,'Flip');  % write to screen
    	WaitSecs(1);

    	% Let's go!
    	for ii=1:params.num.trials_total
			% change key mapping if randomisation requested
			if params.keys.randomised
				if (rand(1)>0.5)
					params.keys.mapping =      params.keys.mapping == 0;
					params.mapping.txt  =    fliplr(params.mapping.txt);
					params.keys.response = fliplr(params.keys.response);
				end
				results.keys.mapping(ii) = params.keys.mapping;
				results.keys.txt{ii}     = params.mapping.txt;
				results.keys.resp(ii,:)  = params.keys.response;
			end
	        % wait for response while stim is on screen
	        legalKeyStroke = 0; % don't allow repeated responses  with legal key
	        abort          = 0; % abort expt

	        % draw fixation and contextual cue
	        draw_fixation(ptb.w,params.fix.size,params.fix.width,params.col.fix,ptb.scr.centre);
	        draw_context(ptb.w,params.ctxt.size,params.ctxt.width,params.col.ctxt_both(expt.task(ii),:),ptb.scr.centre);
					% draw_frame(ptb.w,params.frame.size,params.frame.width,params.col.frame,ptb.scr.centre);
	        Screen(ptb.w,'Flip');
	        %img = Screen('GetImage',ptb.w);
	        %imwrite(img,['fixation' num2str(ii) '.jpg']);
	        [results.time.onset_fix(ii)] = GetSecs();
	        WaitSecs(params.timing.context/1000);


	        % % draw first ISI
	        % draw_fixation(ptb.w,params.fix.size,params.fix.width,params.col.fix,ptb.scr.centre);
	        % draw_frame(ptb.w,params.frame.size,params.frame.width,params.col.frame,ptb.scr.centre);
	        % Screen(ptb.w,'Flip');
	        % %img = Screen('GetImage',ptb.w);
	        % %imwrite(img,['ISI' num2str(ii) '.jpg']);
	        % results.time.onset_ISI_1(ii) = GetSecs();
	        % WaitSecs(params.timing.ISI/1000);


    		% draw stimulus & response contingencies
    		stimTexture = draw_stimulus(ptb.w,ptb.rect,squeeze(treeImages(ii,:,:,:)),ptb.stimRect);
    		draw_mapping(ptb.w,params.mapping.pos,params.mapping.size,params.mapping.width,params.col.mapping,params.mapping.txt,ptb.scr.centre,params.txt.fontSize_mapping);
			draw_context(ptb.w,params.ctxt.size,params.ctxt.width,params.col.ctxt_both(expt.task(ii),:),ptb.scr.centre);
        	% draw_frame(ptb.w,params.frame.size,params.frame.width,params.col.frame,ptb.scr.centre);
    		Screen(ptb.w,'Flip');
			%img = Screen('GetImage',ptb.w);
        	%imwrite(img,['stim' num2str(ii) '.jpg']);
     		[results.time.onset_stim(ii)] = GetSecs();

    		% listen for responses until stim presentation interval is over,
    		% but only register legal keystrokes (escape, resp left/right) once.
    		% allow multiple pauses.
    		while ~abort && (GetSecs-results.time.onset_stim(ii)) <= (params.timing.stimulus/1000) && ~legalKeyStroke
				if ~legalKeyStroke
        			[~,~,keyCode] = KbCheck();

					if (keyCode(params.keys.ID.key_abort))
						legalKeyStroke = 1;
						abort = 1;

					elseif (keyCode(params.keys.ID.key_pause))
						expt_pause(ptb.w,params.col.txt_paused,params.keys.ID.key_pause)

					elseif (keyCode(params.keys.ID.key_resp1))
						legalKeyStroke = 1;
						results.responses.category(ii) = params.keys.response(1);
						results.responses.screen(ii)   = 1; % left
						results.responses.missed(ii)   = 0;
						results.rt(ii) = GetSecs() - results.time.onset_stim(ii);
						respTexture = draw_stimulus(ptb.w,ptb.rect,squeeze(treeImages(ii,:,:,:)),ptb.stimRect);
						draw_response(ptb.w,params.mapping.pos,params.mapping.size,params.mapping.width,params.col.mapping,params.mapping.txt,ptb.scr.centre,params.txt.fontSize_feedback,1);
						draw_context(ptb.w,params.ctxt.size,params.ctxt.width,params.col.ctxt_both(expt.task(ii),:),ptb.scr.centre);
						% draw_frame(ptb.w,params.frame.size,params.frame.width,params.col.frame,ptb.scr.centre);
						Screen(ptb.w,'Flip');
						%img = Screen('GetImage',ptb.w);
						%imwrite(img,['accept' num2str(ii) '.jpg']);

					elseif (keyCode(params.keys.ID.key_resp2))
						legalKeyStroke = 1;
						results.responses.category(ii) = params.keys.response(2);
						results.responses.screen(ii)   = 2; % right
						results.responses.missed(ii)   = 0;
						results.rt(ii) = GetSecs() - results.time.onset_stim(ii);
						respTexture = draw_stimulus(ptb.w,ptb.rect,squeeze(treeImages(ii,:,:,:)),ptb.stimRect);
						draw_response(ptb.w,params.mapping.pos,params.mapping.size,params.mapping.width,params.col.mapping,params.mapping.txt,ptb.scr.centre,params.txt.fontSize_feedback,2);
						draw_context(ptb.w,params.ctxt.size,params.ctxt.width,params.col.ctxt_both(expt.task(ii),:),ptb.scr.centre);
						% draw_frame(ptb.w,params.frame.size,params.frame.width,params.col.frame,ptb.scr.centre);
						Screen(ptb.w,'Flip');
						%img = Screen('GetImage',ptb.w);
						%imwrite(img,['reject' num2str(ii) '.jpg']);

					end
				end
    		end

    		% handle flags:
    		if abort
    			% if experimenter pressed Escape, abort trial loop
    			break;
    		elseif ~legalKeyStroke
    			% if not aborted but still no legak key registered, log as missed trial (redundancy doesn't hurt)
    			results.responses.missed(ii)   =   1;
    			results.responses.category(ii) = NaN;
				results.responses.screen(ii)   = NaN;
				results.responses.reward(ii)   = NaN;
				results.responses.correct(ii)  =   0; % logicals can't have NaNs
			else % if not aborted and legal key pressed, log received reward and whether or not resp was correct
				if (results.responses.category(ii) == 1) % if tree accepted
					results.responses.reward(ii) = expt.rewards(ii);
				elseif (results.responses.category(ii) == -1) % if tree rejected
					results.responses.reward(ii) = 0;
				end
        	end
        	results.responses.correct(ii) = (expt.rewards(ii) >=0 && results.responses.category(ii) == 1) || (expt.rewards(ii) < 0 && results.responses.category(ii) == -1); % correct if good tree accepted or bad tree rejected

			% remove stimulus
			Screen('Close',stimTexture);
			% remove response image (if response was provided)
			if (results.responses.missed(ii)==0)
				Screen('Close',respTexture);
			end

    		% % draw second ISI
    		% draw_fixation(ptb.w,params.fix.size,params.fix.width,params.col.fix,ptb.scr.centre);
    		% Screen(ptb.w,'Flip');
    		% [results.time.onset_ISI_2(ii)] = GetSecs();
    		% WaitSecs(params.timing.ISI/1000);

			% draw feedback
			draw_feedback(ptb.w,results.responses.screen(ii),expt.rewards(ii),results.responses.correct(ii),params,ptb.scr.centre);
			draw_context(ptb.w,params.ctxt.size,params.ctxt.width,params.col.ctxt_both(expt.task(ii),:),ptb.scr.centre);
			% draw_frame(ptb.w,params.frame.size,params.frame.width,params.col.frame,ptb.scr.centre);
			Screen(ptb.w,'Flip');
			%img = Screen('GetImage',ptb.w);
			%imwrite(img,['feedback' num2str(ii) '.jpg']);
			results.time.onset_fb(ii) = GetSecs();
			WaitSecs(params.timing.feedback/1000);

			% ITI
			Screen(ptb.w,'Flip');
			%img = Screen('GetImage',ptb.w);
			%imwrite(img,['ITI' num2str(ii) '.jpg']);
			results.time.ITI(ii) = GetSecs();
			WaitSecs(expt.ITIs(ii)/1000);


    	end

		% log end time
		results.time.stopTime = datestr(now);


		% close ptb and tidy up
		ptb_close()

	catch ME
		ptb_close()
		rethrow(ME);
	end

end
