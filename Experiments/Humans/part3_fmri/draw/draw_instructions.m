function draw_instructions(w,centre,params,instructions)
	%% DRAW_INSTRUCTIONS()
	%
	% draws multi-page instructions as composition of 
	% text and image, centered on the screen
	% 
	% Timo Flesch, 2018

	pageCounter = 1;
	pageMax     = length(instructions);
	pageMin     = 1;
	startExpt   = false;

	while (~startExpt)
		% show text	

		draw_instr_text(w,centre,params,~isempty(instructions(pageCounter).img),instructions(pageCounter).txt);	
		
		% show image
		draw_instr_image(w,centre,params,instructions(pageCounter).img);
		% show navigation buttons 
		% draw_instr_nav(w,centre,params);
		Screen('Flip',w);		
		% wait for response	
		[pageCounter,startExpt] = handle_nav(params.keys.ID.key_navLeft,params.keys.ID.key_navRight,params.keys.ID.key_instr,pageCounter,pageMax,pageMin);
		
	end

end

function [pageCounter,startExpt] = handle_nav(keyLeft,keyRight,keySpace,pageCounter,pageMax,pageMin)
	%% [PAGECOUNTER,STARTEXPT] = HANDLE_NAV(KEYLEFT,KEYRIGHT,KEYSPACE,PAGECOUNTER,PAGEMAX,PAGEMIN)
	% 
	% handles responses for instructions 
	%
	% Timo Flesch, 2018

	startExpt = false;

	% [~,~,keyCode] = KbCheck();   
	[~,keyCode,~] = KbStrokeWait();
    if (keyCode(keyLeft)) % if LEFT
		pageCounter = pageCounter-1;
	elseif (keyCode(keyRight)) % if RIGHT
		pageCounter = pageCounter+1;
	elseif (pageCounter==pageMax && keyCode(keySpace))
		startExpt = true;
	end
	if pageCounter < pageMin
		pageCounter = pageMin;
	elseif pageCounter > pageMax
		pageCounter = pageMax;
	end
end


function draw_instr_text(w,centre,params,hasImg,txt)
	Screen('TextSize',w,params.txt.fontSize_body1)
	Screen('Textfont',w,params.txt.fontType1);
	Screen('TextColor',w,params.col.txt_body);
	if hasImg
		DrawFormattedText(w,txt,'center',centre(2)+params.instr.txt_pos(2));
	else 
		DrawFormattedText(w,txt,'center','center');
	end
end

function draw_instr_image(w,centre,params,img)
	if ~isempty(img)
		img_expt = imread([params.io.imgs img],'jpg');
		Screen('PutImage',w,img_expt,[centre(1)+params.instr.img_pos(1)-params.instr.img_size(1)/2,centre(2)+params.instr.img_pos(2)-params.instr.img_size(2)/2,centre(1)+params.instr.img_pos(1)+params.instr.img_size(1)/2,centre(2)+params.instr.img_pos(2)+params.instr.img_size(2)/2]);
	end
end