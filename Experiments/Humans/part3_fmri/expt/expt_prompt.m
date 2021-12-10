function subject = expt_prompt(taskPhase)
	%% RESPONSES = EXPT_PROMPT()
	% creates dialogues (type of exp, subject details etc)
	%
	% Timo Flesch, 2018

	%%NOTE important: set total number of runs
	numRunsTotal  = 6;

	%% get input
	switch taskPhase
	case 'train'
		prompt      = {'Subject ID','Age:', 'Gender(M/F):', 'Handedness(L/R):', 'Task Order','Rules'};
		default     = {'0','0','M','R','1','1'};
		answer      = inputdlg(prompt,'Subject Data',1,default);
		%% save subject data
		subject = {};
		subject.id             = str2num(answer{1});
		subject.age            = str2num(answer{2});
		subject.gender         =          answer{3};
		subject.handedness     =          answer{4};
		subject.taskorder      = str2num(answer{5});
		subject.ruleAssignment = str2num(answer{6});
	case 'scan'
		prompt      = {'Subject ID','Age:', 'Gender(M/F):', 'Handedness(L/R):', 'Task Order','Rules', 'NumRuns'};
		default     = {'0','0','M','R','1','1','6'};
		answer      = inputdlg(prompt,'Subject Data',1,default);
		%% save subject data
		subject = {};
		subject.id             = str2num(answer{1});
		subject.age            = str2num(answer{2});
		subject.gender         =          answer{3};
		subject.handedness     =          answer{4};
		subject.taskorder      = str2num(answer{5});
		subject.ruleAssignment = str2num(answer{6});
		subject.numRuns        = str2num(answer{7});
		subject.runStart       = numRunsTotal-subject.numRuns+1;
		subject.numRunsTotal   = numRunsTotal;
	end
end
