function results = task_scan(params,expt)
	%% TASK_PRE_SCAN(PARAMS,EXPT)
	%
	% defines first fmri test phase of experiment and iterates through trials
	%
	% Timo Flesch, 2018

	results = struct();
	results.time.leadInStart  = [];
	results.time.leadInEnd    = [];
	results.time.leadOutStart = [];
	results.time.leadOutEnd   = [];
	results.time.startRun     = [];
	results.time.endRun       = [];
	results.time.trigRun      = [];

	%% 1. load tree images
	[treeImages,hasFailed] = lib_loadRGBAImages(expt.treeIDs,params.io.stims);
	if hasFailed
		return;
	end

	%% 2. run experiment
	try
		% create session
		ptb = ptb_open(params);

		% log start time:
		results.time.startTime = datestr(now);


		% Display blank (grey) screen
		% Screen('TextFont', ptb.w, 'Calibri');
		Screen(ptb.w,'FillRect',params.col.bg); % blank screen
		results.time.exp_start = Screen(ptb.w,'Flip');  % write to screen
		WaitSecs(1);

		% Let's go!
		for ii=1:params.num.trials_total

			% check if new run
			if (expt.counter_run(ii) == 1)
				Screen('TextSize',ptb.w,33);
				Screen('TextColor',ptb.w,[1 1 1]);
				continueExp = 0;
				% 1. wait for experimenter to start
				while (~continueExp)
					[~,~,keyCode] = KbCheck();
					DrawFormattedText(ptb.w,'PAUSA','center','center',[1 1 1]);
					Screen(ptb.w,'Flip');
					if keyCode(params.keys.ID.key_cont)
						results.time.startRun(end+1) = GetSecs();
						continueExp = 1;
					end
				end
				% 2. wait for trigger to arrive
				isTrig = 0;
				while (~isTrig)
					[~,~,keyCode] = KbCheck();
					DrawFormattedText(ptb.w,'ESPERANDO TRIGGER','center','center',[1 1 1]);
					Screen(ptb.w,'Flip');
					if keyCode(params.keys.ID.key_trigger)
						results.time.trigRun(end+1) = GetSecs();
						isTrig = 1;
					end
				end
				% 3. lead-in
				Screen(ptb.w,'Flip');
				results.time.leadInStart(end+1) = GetSecs();
				WaitSecs(params.timing.leadOut/1000);
				results.time.leadInEnd(end+1) = GetSecs();
			end

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


			% draw fixation and contextual cue
			draw_fixation(ptb.w,params.fix.size,params.fix.width,params.col.fix,ptb.scr.centre);
			draw_context(ptb.w,params.ctxt.size,params.ctxt.width,params.col.ctxt_both(expt.task(ii),:),ptb.scr.centre);

			Screen(ptb.w,'Flip');
			[results.time.onset_fix(ii)] = GetSecs();
			WaitSecs(params.timing.context/1000);


			% draw stimulus & response contingencies
			stimTexture = draw_stimulus(ptb.w,ptb.rect,squeeze(treeImages(ii,:,:,:)),ptb.stimRect);
			draw_mapping(ptb.w,params.mapping.pos,params.mapping.size,params.mapping.width,params.col.mapping,params.mapping.txt,ptb.scr.centre,params.txt.fontSize_mapping);
			draw_context(ptb.w,params.ctxt.size,params.ctxt.width,params.col.ctxt_both(expt.task(ii),:),ptb.scr.centre);
			Screen(ptb.w,'Flip');
			[results.time.onset_stim(ii)] = GetSecs();

			% listen for responses until stim presentation interval is over,
			% but only register legal keystrokes (escape, resp left/right) once.
			% allow multiple pauses.
			% wait for response while stim is on screen
			legalKeyStroke = 0; % don't allow repeated responses  with legal key
			abort          = 0; % abort expt

			while ~abort && (GetSecs-results.time.onset_stim(ii)) <= (params.timing.stimulus/1000)
				if ~legalKeyStroke
					[~,~,keyCode] = KbCheck();

					if (keyCode(params.keys.ID.key_abort))
						legalKeyStroke = 1;
						abort = 1;

					elseif (keyCode(params.keys.ID.key_pause))
						expt_pause(ptb.w,params.col.txt_paused,params.keys.ID.key_pause);

					elseif (keyCode(params.keys.ID.key_resp1))
						legalKeyStroke = 1;
						results.responses.category(ii) = params.keys.response(1);
						results.responses.screen(ii)   = 1; % left
						results.responses.missed(ii)   = 0;
						results.rt(ii) = GetSecs() - results.time.onset_stim(ii);
						respTexture = draw_stimulus(ptb.w,ptb.rect,squeeze(treeImages(ii,:,:,:)),ptb.stimRect);
						draw_response(ptb.w,params.mapping.pos,params.mapping.size,params.mapping.width,params.col.mapping,params.mapping.txt,ptb.scr.centre,params.txt.fontSize_feedback,1);
						draw_context(ptb.w,params.ctxt.size,params.ctxt.width,params.col.ctxt_both(expt.task(ii),:),ptb.scr.centre);

						Screen(ptb.w,'Flip');

					elseif (keyCode(params.keys.ID.key_resp2))
						legalKeyStroke = 1;
						results.responses.category(ii) = params.keys.response(2);
						results.responses.screen(ii)   = 2; % right
						results.responses.missed(ii)   = 0;
						results.rt(ii) = GetSecs() - results.time.onset_stim(ii);
						respTexture = draw_stimulus(ptb.w,ptb.rect,squeeze(treeImages(ii,:,:,:)),ptb.stimRect);
						draw_response(ptb.w,params.mapping.pos,params.mapping.size,params.mapping.width,params.col.mapping,params.mapping.txt,ptb.scr.centre,params.txt.fontSize_feedback,2);
						draw_context(ptb.w,params.ctxt.size,params.ctxt.width,params.col.ctxt_both(expt.task(ii),:),ptb.scr.centre);

						Screen(ptb.w,'Flip');
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

			% ITI
			if (expt.counter_run(ii) ~= params.num.trials_run)
				Screen(ptb.w,'Flip');
				results.time.ITI(ii) = GetSecs();
				WaitSecs(expt.ITIs(ii)/1000);
			end

			% if end of run, lead-out and save data
			if (expt.counter_run(ii) == params.num.trials_run) % && ii~=params.num.trials_total)
				Screen(ptb.w,'Flip');
				results.time.leadOutStart(end+1) = GetSecs();
				WaitSecs(params.timing.leadOut/1000);
				results.time.leadOutEnd(end+1)   = GetSecs();
				data = struct();
				data.params     =  params;
				data.conditions =    expt;
				data.results    = results;

				results.time.endRun(end+1) = GetSecs();
				save([params.io.output 's' num2str(params.task.subjID) '_scan_tmpData_run_' num2str(expt.runs(ii))],'data');
			end
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
