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
			keyIMGs[0] = board.paper.object.image([parameters.keyURL + 'arrow_' + 'left' + '_alpha_reject.png'],board.paper.centre[0]-88,board.paper.centre[1]-175,parameters.visuals.size.keyIMG[0],parameters.visuals.size.keyIMG[1]);
			keyIMGs[1] = board.paper.object.image([parameters.keyURL + 'arrow_' + 'right' + '_alpha_accept.png'],board.paper.centre[0]+14,board.paper.centre[1]-175,parameters.visuals.size.keyIMG[0],parameters.visuals.size.keyIMG[1]);
			break;
		case 1:
			keyIMGs[0] = board.paper.object.image([parameters.keyURL + 'arrow_' + 'left' + '_alpha_accept.png'],board.paper.centre[0]-88,board.paper.centre[1]-175,parameters.visuals.size.keyIMG[0],parameters.visuals.size.keyIMG[1]);
			keyIMGs[1] = board.paper.object.image([parameters.keyURL + 'arrow_' + 'right' + '_alpha_reject.png'],board.paper.centre[0]+14,board.paper.centre[1]-175,parameters.visuals.size.keyIMG[0],parameters.visuals.size.keyIMG[1]);
			break;
	}
	return keyIMGs;

}


function drawTree(treeName) {
/*
	draws a nice tree
*/

  return board.paper.object.image(parameters.treeURL.concat(treeName),board.paper.centre[0]-parameters.visuals.size.stim[0]/2,board.paper.centre[1]-parameters.visuals.size.stim[1]/2,parameters.visuals.size.stim[0],parameters.visuals.size.stim[1]).attr({"opacity": 0});

}


function drawAcceptTree () {
/*
	puts tree into garden
*/

   board.garden.tree = board.stimuli.tree;
   board.garden.tree.attr({'width':parameters.visuals.size.fbt[0]});
   board.garden.tree.attr({'height':parameters.visuals.size.fbt[1]});
   board.garden.tree.attr({'x':board.paper.centre[0]-75});
   board.garden.tree.attr({'y':board.paper.centre[1]+35});
}


function drawRejectTree () {
/*
	does nothing
*/
	// for now, do nothing!

}


function drawFeedbackTree() {
/*
	resizes tree relative to received reward/penalty
*/
		// let tree grow!

		board.garden.tree = board.stimuli.tree;
		board.garden.tree.attr({'width':parameters.visuals.size.fbt[0]+sdata.expt_rewardIDX[coding.index]});
   	board.garden.tree.attr({'height':parameters.visuals.size.fbt[1]+sdata.expt_rewardIDX[coding.index]});
   	board.garden.tree.attr({'x':board.paper.centre[0]-75-0.5*sdata.expt_rewardIDX[coding.index]});
   	board.garden.tree.attr({'y':board.paper.centre[1]+35-0.8*sdata.expt_rewardIDX[coding.index]});
}


function drawGarden(gardenName,blurOrNot) {
/*
	draws orchard, either blurred or not blurred
*/
  if (blurOrNot) {
	board.blurcue.object = board.paper.object.image(parameters.gardenURL.concat(gardenName),board.paper.centre[0]-parameters.visuals.size.garden[0]/2,board.paper.centre[1]-parameters.visuals.size.garden[1]/2,parameters.visuals.size.garden[0],parameters.visuals.size.garden[1]).attr({"opacity":0});
	board.blurcue.object.blur(parameters.visuals.blurlvl);
	board.blurcue.context = drawRect(board.paper.object,[board.paper.centre[0]-parameters.visuals.size.garden[0]/2,board.paper.centre[1]-parameters.visuals.size.garden[1]/2,parameters.visuals.size.garden[0],parameters.visuals.size.garden[1]]);
  	board.blurcue.context.attr({stroke:"black","stroke-width":2});
  }
  else {
	board.cue.object = board.paper.object.image(parameters.gardenURL.concat(gardenName),board.paper.centre[0]-parameters.visuals.size.garden[0]/2,board.paper.centre[1]-parameters.visuals.size.garden[1]/2,parameters.visuals.size.garden[0],parameters.visuals.size.garden[1]).attr({"opacity":0});
	board.cue.context = drawRect(board.paper.object,[board.paper.centre[0]-parameters.visuals.size.garden[0]/2,board.paper.centre[1]-parameters.visuals.size.garden[1]/2,parameters.visuals.size.garden[0],parameters.visuals.size.garden[1]]);
  	board.cue.context.attr({stroke:"black","stroke-width":2});

  }

}


function drawTrunk() {
/*
	draws nothing but a trunk of a poor little dead tree :'( RIP, Tree!
*/

	board.garden.tree = board.paper.object.image(parameters.treeURL.concat('trunk.png'),board.paper.centre[0]-75,board.paper.centre[1]+10,150,150).attr({'opacity':0});
}
