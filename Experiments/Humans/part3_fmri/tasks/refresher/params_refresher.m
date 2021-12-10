function params = params_refresher(subj,exemplars)
	%% PARAMS_PRE_BEHAV(SUBJ,EXEMPLARS)
	%
	% sets all paramaters for first training phase
	%
	% Timo Flesch, 2018

	params = struct();

	%% TASK ---------------------------------------------------------------------------------------
	params.task.exemplars =           exemplars;
	params.task.order     =      subj.taskorder;  % 1 = north-south; 2 = south,north
	params.task.curric    =     'i'; % b = blocked;     i = interleaved
	params.task.ruleID    = subj.ruleAssignment;
	params.task.sessID    =                   4;

    %% NUM  ---------------------------------------------------------------------------------------
	params.num.blocks      =                1;
	params.num.branchiness =       [1,2,3,4,5];
	params.num.leafiness   =       [1,2,3,4,5];
	params.num.catbound    = floor(mean(params.num.leafiness));
	if (subj.ruleAssignment == 1)
		params.num.rewards_branchiness     =   [-50,-25,0,25,50];
		params.num.rewards_leafiness       =   [-50,-25,0,25,50];
	elseif (subj.ruleAssignment == 2)
		params.num.rewards_branchiness     =   fliplr([-50,-25,0,25,50]);
		params.num.rewards_leafiness       =   fliplr([-50,-25,0,25,50]);
	elseif (subj.ruleAssignment == 3)
		params.num.rewards_branchiness     =           [-50,-25,0,25,50];
		params.num.rewards_leafiness       =   fliplr([-50,-25,0,25,50]);
	elseif (subj.ruleAssignment == 4)
		params.num.rewards_branchiness     =   fliplr([-50,-25,0,25,50]);
		params.num.rewards_leafiness       =           [-50,-25,0,25,50];
	end

	params.num.exemplars   = length(exemplars);
	params.num.tasks       =                 2; % north,south

	params.num.trials_total = params.num.blocks*length(params.num.branchiness)*...
	length(params.num.leafiness)*params.num.exemplars*params.num.tasks;
	params.num.trials_block = params.num.trials_total/params.num.blocks;
	params.num.trials_task  = params.num.trials_total/params.num.tasks;


	%% FILE I/O -----------------------------------------------------------------------------------
	params.io.ptb    = ''; % psychtoolbox
	params.io.imgs   =         [pwd '/files/images/']; % stimuli
	params.io.stims  =        [pwd '/files/stimuli/']; % stimuli
	params.io.output =           [pwd '/files/data/']; % data
	params.io.txt    =              [pwd '/lib/txt/']; % default text files for instructions, break, thank you
	params.io.instr  =     [pwd '/tasks/pre1/instr/']; % text files for task-specific instructions


	%% COLOURS ------------------------------------------------------------------------------------
	params.col.bg                = .5.*[1,1,1]; % background

	params.col.txt_headline      =     [1,1,1]; % headline text
	params.col.txt_body          =     [1,1,1]; % main text (instructions/break)
	params.col.txt_timer         =     [0,1,0]; % timer text (clock in break)
	params.col.txt_warning       =     [1,0,0]; % warning text (e.g. clock timeout)
	params.col.txt_paused        =     [.8,0,0]; % pause text (experimenter)

	params.col.ctxt_north        =     [0, 178, 255]./255; % north context
	params.col.ctxt_south        =     [255, 178, 0]./255; % south context
	params.col.ctxt_both = [params.col.ctxt_north;params.col.ctxt_south];

	params.col.mapping           =     [0,0,0]; % response mapping
	params.col.resp_correct      =     [0,1,0]; % correct response
	params.col.resp_incorrect    =     [1,0,0]; % incorrect response
	params.col.resp_timeOut      =     [1,1,0]; % timeOut response
	params.col.resp_default      =     [0,0,0]; % default

	params.col.fix               =     [0,0,0]; % fixation
	params.col.borders           =     [0,0,0]; % any borders
	params.col.frame             =     [0,0,0]; % frame
	params.col.stim              =     [0,0,0]; % stimulus (if graph. primitive)


	%% KEYS ---------------------------------------------------------------------------------------
	KbName('UnifyKeyNames');
	params.keys.ID.key_pause =              KbName('p');
	params.keys.ID.key_abort =         KbName('Escape');
	params.keys.ID.key_resp1 =              KbName('f'); % LeftArrow
	params.keys.ID.key_resp2 =              KbName('j'); % RightArrow
	params.keys.ID.key_navLeft =    KbName('LeftArrow');
	params.keys.ID.key_navRight =  KbName('RightArrow');
	params.keys.ID.key_instr =          KbName('Space');
	params.keys.validKeys    = helper_getValidKeys(params.keys.ID);
	params.keys.mapping      = rand(1) > 0.5;
	params.keys.randomised   = 1;
	if (params.keys.mapping == 0)
		params.keys.response   = [1,-1]; % key mapping (y,n)
	elseif (params.keys.mapping == 1)
		params.keys.response   = [-1,1]; % key mapping (n,y)
	end


	%% PTB ----------------------------------------------------------------------------------------
	params.ptb.dbg             = 0; % Debug mode
	params.ptb.doublebuffer    = 1;
	params.ptb.listenChar      = 0; % show pressed buttons (y/n = 1/0)
	params.ptb.hideCursor      = 1; % hide mouse cursor (y/n=1/0)

	params.ptb.blendfunc_sfNew =           'GL_SRC_ALPHA'; % for Screen('Blendfunction'), to have proper anti aliasing
	params.ptb.blendfunc_dfNew = 'GL_ONE_MINUS_SRC_ALPHA'; % dito


	%% INSTRUCTIONS -------------------------------------------------------------------------------
	params.instr.txt_pos   =   [0,-200];
	params.instr.img_pos  =    [0,100];
	params.instr.img_size  =  [536,400];

	%% FIXATION -----------------------------------------------------------------------------------
	params.fix.size  =   [20,20]; % width, height
	params.fix.width =         4; %
	params.fix.pos   =     [0,0]; % x,y


	%% CONTEXT ------------------------------------------------------------------------------------
	params.ctxt.size  = [300,300]; % width, height
	params.ctxt.width =        40; %
	params.ctxt.pos   =     [0,0]; % x,y


    %% STIM ---------------------------------------------------------------------------------------
	params.stim.size = [300,300]; % width, height
	params.stim.width =       4; %
	params.stim.pos  =     [0,0]; % x,y


	%% FRAME --------------------------------------------------------------------------------------
	params.frame.size = [300,300]; % width, height
	params.frame.width =       2; %
	params.frame.pos  =     [0,0]; % x,y


	%% MAPPING ------------------------------------------------------------------------------------
	params.mapping.size  =           [50,50]; % width, height
	params.mapping.width =                4; % rect width
	params.mapping.pos   = [0-200,0;0+200,0]; % x,y
    if (params.keys.mapping == 0)
		params.mapping.txt  = {'aceptar','rechazar'};
	elseif (params.keys.mapping == 1)
		params.mapping.txt  = {'rechazar','aceptar'};
	end


	%% FEEDBACK -----------------------------------------------------------------------------------
	params.fdb.size =               [50,50]; % width, height
	params.fdb.pos  =     [0-200,0;0+200,0]; % x,y
	params.fdb.width =                    4; % rect width
  params.fdb.txt   =           {'¡demasiado lento!'};


  %% TIMEOUT ------------------------------------------------------------------------------------
	params.to.pos  =              [0-100,0]; % x,y
	params.to.width =                    4; % rect width
  params.to.txt   =           {'¡demasiado lento!'};


	%% BREAKS  ---------------------------------------------------------------------------------------
	% params.breaks.txt = 'It''s time for a break!\n The experiment continues automatically once the timer has reached zero. \n If you want to continue now, press SPACE';
	% % params.breaks.txtB = 'Pay attention to the colour of the contextual cue at the beginning of each trial. \n You will now learn which trees grow best in the second garden!';
	% params.breaks.duration =    5*60; % five minutes
	% params.breaks.ypos     = -400;


	%% TIMING  ---------------------------------------------------------------------------------------
  params.timing.doJitter   = false;
	params.timing.context  = 1000;
	params.timing.stimulus = 3000;
	params.timing.feedback =  500;
	params.timing.ISI      =  0;%500;
	params.timing.ITI      = 1000;
	params.timing.minITI   =  800;
	params.timing.maxITI   = 1200;


	%% TEXT  --------------------------------------------------------------------------------------
	params.txt.fontSize_headline1 = 42;
	params.txt.fontSize_headline2 = 22;
	params.txt.fontSize_body1     = 20;
	params.txt.fontSize_body2     = 14;
	params.txt.fontSize_body3     = 12;
	params.txt.fontSize_mapping   = 22;
	params.txt.fontSize_timeout   = 32;
	params.txt.fontSize_feedback  = 32;

	params.txt.fontStyle_headline =   'bold'; % headlines
	params.txt.fontStyle_resp     =   'bold'; % responses
	params.txt.fontStyle_warning  =   'bold'; % warnings
	params.txt.fontStyle_body     = 'normal'; % body text

	params.txt.fontType1          = 'Calibri'; % main font type (keep OS compatibility in mind!)

end


function validKeys = helper_getValidKeys(params)
	validKeys = [];
	fn = fieldnames(params);
	for ii = 1:length(fn)
		validKeys = [validKeys,params.(fn{ii})];
	end
end
