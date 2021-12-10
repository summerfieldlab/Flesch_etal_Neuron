function draw_phases(w,centre,imPath,phaseID)
	%% DRAW_PHASES()
	%
	% draws experimental phases (blocks)
	% and highlights the one the participant is about to start
	%
	% Timo Flesch, 2018

 	switch phaseID
	 	case 0
	 		img_expt = imread([imPath 'phases_all.jpg'],'jpg');
			Screen('PutImage',w,img_expt,[centre(1)-270,centre(2)-100,centre(1)+270,centre(2)+100]);
	 	case 1
	 		img_expt = imread([imPath 'phases_1.png'],'png');
			Screen('PutImage',w,img_expt,[centre(1)-270,centre(2)-100, centre(1)+270, centre(2)+100]);
	 	case 2
	 		img_expt = imread([imPath 'phases_2.jpg'],'jpg');
			Screen('PutImage',w,img_expt,[centre(1)-270,centre(2)-100,centre(1)+270,centre(2)+100]);
	 	case 3
	 		img_expt = imread([imPath 'phases_3.jpg'],'jpg');
			Screen('PutImage',w,img_expt,[centre(1)-270,centre(2)-100,centre(1)+270,centre(2)+100]);
	 	case 4
	 		img_expt = imread([imPath 'phases_4.png'],'png');
			Screen('PutImage',w,img_expt,[centre(1)-270,centre(2)-100,centre(1)+270,centre(2)+100]);
	end

end
