function stimTexture = draw_stimulus(w,rect,treeIMG,stimRect)
	%% DRAW_STIMULUS(W,RECT,TREEIMG,IMRECT)
	%
	% draws stimulus (RGBA image)
	% 
	% Timo Flesch, 2018
    
	stimTexture = Screen('MakeTexture', w, treeIMG);
	Screen('DrawTexture', w, stimTexture, [], stimRect);
	
end