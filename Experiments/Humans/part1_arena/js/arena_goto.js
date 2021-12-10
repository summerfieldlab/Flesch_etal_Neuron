/* **************************************************************************************

Trial Transitions
(c) Timo Flesch, 2016/17
Summerfield Lab, Experimental Psychology Department, University of Oxford

************************************************************************************** */

function gotoNextTrial(){
/*
	switches to next trial.
*/
	// save final coordinates
	exp_saveStimCoords();
	// remove all stims
	stims_emptySet();
	// update trialCount
	numbers.trialCount++;
	// add the new set of stims
	stims_fillSet();
        // save temp file
	saveExperiment();
}

function gotoNextTask() {
/*
	saves results and progresses with next task (e.g. in pre-post design)
*/
	exp_saveStimCoords();
	exp_saveParticipantData();

	stims_emptySet();
	arena_removeUI();
	if (instr_id=='dissimrating_pre'){
		finishDissimRatingExperiment_pre();
	}
	else {
		finishDissimRatingExperiment_post();
	}
	//arena_drawThanks();

}



function arena_drawThanks() {
/*
helper function to draw a little thank you message
*/
board.circle.remove();
// stopClock();
$('h1').html(['']);
stimulus = board.paper.object.image('happytree.jpg',board.centre[0]-100,board.centre[1]-100,200,200);

}
