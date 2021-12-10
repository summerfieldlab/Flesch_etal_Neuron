/* **************************************************************************************

Draw and Remove Stimuli
(c) Timo Flesch, 2016/17
Summerfield Lab, Experimental Psychology Department, University of Oxford

************************************************************************************** */
var g;
// DRAW STIMULI ON CANVAS
function stims_fillSet() {
/*
	1. converts images into raphael objects,
	2. places them along the circle,
	3. adds them to raphael set
	4. makes set draggable
*/

	// rather stupid solution...time for a break, deffo. lol
	board.set 	= board.paper.object.set();
	phiSet = rnd_randomSampling(math_linspace(0,Math.floor(360/params_exp.numStimuli),360),params_exp.numStimuli);

	for(var i=0;i<params_exp.numStimuli;i++) {
		stimIDX = i+(params_exp.numStimuli*(numbers.trialCount-1));

		if(FLAG_DEBUG) {
			//console.log(stimIDX)
			//console.log(stimIDX==numbers.stimCount);
		}
		
		// set coordinates and generate object
		coords = [Math.floor((board.radius-params_vis.stimSize/1.4)*math_cos(phiSet[i])),Math.floor((board.radius-params_vis.stimSize/1.4)*math_sin(phiSet[i]))];
		stimulus = board.paper.object.image(params_exp.treeDir +stim.stimNames[stimIDX],board.centre[0]+coords[0]-params_vis.stimSize/2,board.centre[1]+coords[1]-params_vis.stimSize/2,params_vis.stimSize,params_vis.stimSize);
		if (params_vis.drawBB) {
			stimulus.g = stimulus.glow({'color':'#696969','stroke-width':1,'opacity':0.5});
		}
		// add object to set
		board.set.push(stimulus);
		stim.coordsOrig[numbers.stimCount] = [board.set[i].attr("x"), board.set[i].attr("y")];
		numbers.stimCount++;
	}
	// make all objects within the set draggable
	board.set.drag(move, start, up);


}


function stims_emptySet() {
/*
	removes all stimuli from set
*/
	board.set.remove();
}
