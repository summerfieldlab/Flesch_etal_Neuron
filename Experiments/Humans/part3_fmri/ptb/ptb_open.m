function ptb = ptb_open(params)
	%% PTB_OPEN(PARAMS)
	%
	% opens a ptb session with
	% settings defined in params_ptb()
	%
	% Timo Flesch, 2018

	% prepare ptb:
	% AssertOpenGL; % check if PTB is based on OpenGL and actually works
	% KbName('UnifyKeyNames'); % use MacOSX naming scheme

	% call the two functions above and
	% rescale color vals from [0,255] to [0,1]
	% on each call of PsychImaging('OpenWindow'):
	PsychDefaultSetup(2);

	ptb = struct();

	%% Screen Settings
	% debug mode
	if params.ptb.dbg
		Screen('Preference','SkipSyncTests',1);
		Screen('Preference', 'ConserveVRAM', 64);
		Screen('Preference', 'VisualDebugLevel', 3);
		waitfor(msgbox('Yo, keep in mind that you''re in debug mode and synch tests are set to 1!'));
	end

	ptb.scr.ID = max(Screen('Screens'));

    [ptb.w, ptb.rect] = PsychImaging('OpenWindow', ptb.scr.ID, params.col.bg);

    [ptb.scr.width,ptb.scr.height] = Screen('WindowSize', ptb.w);
    ptb.scr.centre = [ptb.scr.width/2 ptb.scr.height/2];

    % Query FPS (frames per second) and IFI (inter frame interval)
    ptb.fps=Screen('FrameRate',ptb.w);      % frames per second
    ptb.ifi=Screen('GetFlipInterval', ptb.w);
    if ptb.fps==0
        ptb.fps=1/ptb.ifi;
    end
	 % switch on alpha blending
    Screen('BlendFunction', ptb.w, params.ptb.blendfunc_sfNew, params.ptb.blendfunc_dfNew);

    % Set Font
	Screen('Textfont',ptb.w,params.txt.fontType1)

    %% Compute Settings
    % reduce chance of interference by other programmes running on PC:
    Priority(MaxPriority(ptb.w));

    % Hide cursor
    if params.ptb.hideCursor
    	HideCursor;
    end

    % Oppress display of button presses in command window
    if ~params.ptb.listenChar
    	ListenChar(2);
	end

    %% Keyboard/Input settings
    % restrict keys
    RestrictKeysForKbCheck(params.keys.validKeys);


    %% Stimulus Settings
	ptb.stimRectSize = [0 0 params.stim.size(1),params.stim.size(2)];
	[ptb.stimRect, ~, ~] = CenterRect(ptb.stimRectSize, ptb.rect);
end
