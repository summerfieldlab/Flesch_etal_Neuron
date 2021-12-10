function expt_instr(instrID,ptb,params)
	%% EXPT_INSTR(INSTRID,PTB)
	%
	% displays multi-page instructions
	%
	% Timo Flesch, 2018


	switch instrID
		case 'general'
			draw_welcomePage(ptb.w,ptb.scr.centre,params);
			draw_introPage(ptb.w,ptb.scr.centre,params.io.imgs,params);
		case 'refresher'
			%sessDuration = round((params.timing.context+params.timing.stimulus+params.timing.feedback+params.timing.ISI+params.timing.ITI)*params.num.trials_total/1000/60);
			sessDuration = round((params.timing.context+1500+params.timing.feedback+params.timing.ISI+params.timing.ITI)*params.num.trials_total/1000/60);
			draw_progressPage(1,4,ptb.w,ptb.scr.centre,params.io.imgs,params,sessDuration);
			instructions = setInstructions_refresher();
			draw_instructions(ptb.w,ptb.scr.centre,params,instructions);
			draw_startPage(ptb.w,params);
		case 'scan'
			%sessDuration = round((params.timing.context+params.timing.stimulus+params.timing.ISI+params.timing.ITI)*params.num.trials_total/1000/60);
			sessDuration = round((params.timing.context+1500+params.timing.ISI+params.timing.ITI)*params.num.trials_total/1000/60);
			draw_progressPage(2,'test',ptb.w,ptb.scr.centre,params.io.imgs,params,sessDuration);
			instructions = setInstructions_post_scan();
			draw_instructions(ptb.w,ptb.scr.centre,params,instructions);
			draw_startPage(ptb.w,params);
	end
end
