function inStruct = setInstructions_scan()
%% SETINSTRUCTIONS_PRE_SCAN()
	%
	% sets instructions as struct
	% with fields for text and image
	%
	% Timo Flesch, 2018

	inStruct = struct();

		inStruct(1).txt     = 'Now we''ll test if you''ve discovered the correct rules.  \n You''ll receive instructions on the test phase. \n Please use the LEFT and RIGHT arrow keys to navigate through this tutorial.';
		inStruct(1).img     = '' ;

		inStruct(2).txt     = 'Again, each trial begins with an image of the garden you''re currently in, \n for example the south garden';
		inStruct(2).img     = 'intro_south.jpg';

		% inStruct(3).txt     = 'Next, you''ll only see a fixation cross in the centre of the screen.';
		% inStruct(3).img     = 'intro_isi.jpg';

		inStruct(3).txt     = 'Shortly after, an image of a tree will appear, together with the key assignment. \n  This key assignment changes randomly from trial to trial, so please pay attention!!';
		inStruct(3).img     =  'intro_stim.jpg';

		inStruct(4).txt     = 'You decide whether you want to plant the tree or not. \n To communicate your decision, you press either the left or right arrow key. \n In this example, I''ve pressed the left key, which is associated with "accept" on this trial \n Your chosen option is highlighted, and the unchosen option disappears:';
		inStruct(4).img     = 'intro_accept.jpg';

		inStruct(5).txt     = 'So far, everything is exactly as for the training session. \n However, as we''re interested in how well you''ve learned the rules, \n there will be NO feedback anymore.';
		inStruct(5).img     = '' ;

		inStruct(6).txt    = 'This means that after you''ve communicated your decision (accept or reject), \n you''ll see a grey screen before the next trial begins. \n The intervals between each trial are going to be longer than during the training phase. \n There will be a break in the middle of this phase';
		inStruct(6).img     = '' ;


		inStruct(7).txt   = 'SUMMARY   \n 1. This is a TEST phase, there will be NO feedback.  \n \n 2. On each trial, a coloured square tells you which garden you''re currently in. \n \n 3. This may or may not stay constant for several trials. Pay attention! \n \n 4. Use arrow keys (left,right) to communicate your choice. \n \n 5. Pay attention to the key mapping displayed on the screen. \n \n  This tells you whether the left or right arrow corresponds to "accept". \n \n 6. Try to be as accurate as possible, this is very important \n \n ';
		inStruct(7).img   = '';

		inStruct(8).txt   = 'Now press SPACE to continue to the next screen,\n or use the LEFT and RIGHT arrow keys to navigate again through the instructions.'
		inStruct(8).img   = '';

end
