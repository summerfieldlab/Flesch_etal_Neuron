/* **************************************************************************************

Draws raphael objects
(c) Timo Flesch, 2016 [timo.flesch@gmail.com]
Summerfield Lab, Experimental Psychology Department, University of Oxford

************************************************************************************** */
function drawKeys(keyAssign) {
/*
	draws response keys during stim presentation
*/
	keyIMGs = [];
	switch (keyAssign) {
		case 0:
			// keyIMGs[0] = board.paper.object.image([parameters.keyURL + 'arrow_' + 'left' +	 '_alpha_reject.png'],board.paper.centre[0]-88,board.paper.centre[1]-175,parameters.visuals.size.keyIMG[0],parameters.visuals.size.keyIMG[1]);
			// keyIMGs[1] = board.paper.object.image([parameters.keyURL + 'arrow_' + 'right' + '_alpha_accept.png'],board.paper.centre[0]+14,board.paper.centre[1]-175,parameters.visuals.size.keyIMG[0],parameters.visuals.size.keyIMG[1]);
			keyIMGs[0] = board.paper.object.image([parameters.keyURL + 'arrow_' + 'left' +	 '_alpha_reject.png'],board.paper.centre[0]-parameters.visuals.size.stim[0]/2-parameters.visuals.size.keyIMG[0],board.paper.centre[1]-parameters.visuals.size.keyIMG[1]/2,parameters.visuals.size.keyIMG[0],parameters.visuals.size.keyIMG[1]);
			keyIMGs[1] = board.paper.object.image([parameters.keyURL + 'arrow_' + 'right' + '_alpha_accept.png'],board.paper.centre[0]+parameters.visuals.size.stim[0]/2,board.paper.centre[1]-parameters.visuals.size.keyIMG[1]/2,parameters.visuals.size.keyIMG[0],parameters.visuals.size.keyIMG[1]);
			break;
		case 1:
			keyIMGs[0] = board.paper.object.image([parameters.keyURL + 'arrow_' + 'left' + '_alpha_accept.png'],board.paper.centre[0]-parameters.visuals.size.stim[0]/2-parameters.visuals.size.keyIMG[0],board.paper.centre[1]-parameters.visuals.size.keyIMG[1]/2,parameters.visuals.size.keyIMG[0],parameters.visuals.size.keyIMG[1]);
			keyIMGs[1] = board.paper.object.image([parameters.keyURL + 'arrow_' + 'right' + '_alpha_reject.png'],board.paper.centre[0]+parameters.visuals.size.stim[0]/2,board.paper.centre[1]-parameters.visuals.size.keyIMG[1]/2,parameters.visuals.size.keyIMG[0],parameters.visuals.size.keyIMG[1]);
			break;
	}
	return keyIMGs;

}


function drawChoiceRect(scrSide) {
	/*
	 	draws rectangle around choosen option

	*/

	switch (scrSide) {
		case 'right':
			// rect = drawRect(board.paper.object,[board.paper.centre[0]+12,board.paper.centre[1]-180,80,80]);
			rect = drawRect(board.paper.object,[board.paper.centre[0]+parameters.visuals.size.stim[0]/2,board.paper.centre[1]-parameters.visuals.size.keyIMG[1]/2,80,80]);

			rect.attr({"stroke-width":4});
			break;
		case 'left':
			// rect = drawRect(board.paper.object,[board.paper.centre[0]-90,board.paper.centre[1]-180,80,80]);
			rect = drawRect(board.paper.object,[board.paper.centre[0]-parameters.visuals.size.stim[0]/2-parameters.visuals.size.keyIMG[0],board.paper.centre[1]-parameters.visuals.size.keyIMG[1]/2,80,80]);
			rect.attr({"stroke-width":4});
			break;
	}
	return rect;
}


function drawTree(treeName) {
/*
	draws a nice tree
*/

  return board.paper.object.image(parameters.treeURL.concat(treeName),board.paper.centre[0]-parameters.visuals.size.stim[0]/2,board.paper.centre[1]-parameters.visuals.size.stim[1]/2,parameters.visuals.size.stim[0],parameters.visuals.size.stim[1]).attr({"opacity": 0});

}


function drawGarden(gardenName) {
/*
	draws orchard, either blurred or not blurred
*/

	switch (gardenName) {
	case 'north':
		board.cue.context = drawRect(board.paper.object,[board.paper.centre[0]-parameters.visuals.size.garden[0]/2,board.paper.centre[1]-parameters.visuals.size.garden[1]/2,parameters.visuals.size.garden[0],parameters.visuals.size.garden[1]]);
		board.cue.context.attr({stroke:parameters.visuals.cols.ctx_north,"stroke-width":20});
		break;
	case 'south':
		board.cue.context = drawRect(board.paper.object,[board.paper.centre[0]-parameters.visuals.size.garden[0]/2,board.paper.centre[1]-parameters.visuals.size.garden[1]/2,parameters.visuals.size.garden[0],parameters.visuals.size.garden[1]]);
		board.cue.context.attr({stroke:parameters.visuals.cols.ctx_south,"stroke-width":20});
		break;
	 }

	}
