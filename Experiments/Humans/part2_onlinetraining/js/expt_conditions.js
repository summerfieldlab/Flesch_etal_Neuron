/* **************************************************************************************

Creates experimental conditions.

************************************************************************************** */
shuffSwitch = 0;

function createSdata(){
/*
	here happens all the magic
*/
	// index from very first to very last trial of entire exp
	sdata.expt_index 	= colon(1,parameters.nb_trialsTotal);

	// index from first to last trial within each phase
	sdata.expt_trial 	= gen_phaseTrialVect();

  // generate phase indices
	sdata.expt_phase 	       = gen_phaseVect();

	// generate block indices (determine when participant takes breaks)
	sdata.expt_block 	       = gen_blockVect();

	// generate context indices (to distinguish between branch and leaf tasks)
	sdata.expt_contextIDX   = gen_contextVect();

	// generate session indices (training 1 vs test 2)
	// sdata.expt_sessIDX      = gen_sessVect();

	// generate branchiness indices (rep([1,1,1,1,1,2,2,2......5,5,5,5]))
	sdata.expt_branchIDX   	= gen_branchVect();

	// generate leafiness indices (rep([1,2,3,4,5,1,2,3,4,5....]))
	sdata.expt_leafIDX 	    = gen_leafVect();

	// generate vector with exemplar indices [I have several exemplars per level]
	sdata.expt_exemplarIDX  = gen_exemplarVect();

	// generate category label indices (depending on task order)
	sdata.expt_catIDX	      = gen_catVect();

	// generate vector with reward values, depending on task and feature level
	sdata.expt_rewardIDX    = gen_rewardVect();

	// generate vector of key mappings (1: left-accept, right-reject, 0: left-reject, right-accept)
	sdata.expt_keyassignment = gen_keyAssignments();
	// generate vector of key mapping descriptors ('left','right' vs 'right','left')
	sdata.expt_keyStr       = gen_keyStrings();
	// generate vector with "optimal" return up to each trial n (for performance assessment)
	sdata.expt_returnOPT    = [];  //done on the fly
	sdata.expt_rewardOPT    = [];

	// whob whob shuffle
	shuffle_trials(shuffSwitch);
}


function gen_keyAssignments() {
	/*
	generate vector of key mappings
	0: left-reject, right-accept
	1: left-accept, right-reject
	*/
	keyMappings = randi(2,parameters.nb_trialsTotal);
	return keyMappings;
}


function gen_keyStrings() {
	/*
	 generate vector of key mapping descriptors
	 ('left','right' vs 'right','left')
 	*/
	keyStrings = [];
	keyMappings = [['right: accept',' left: reject'],['left: accept',' right: reject']];
	for (var ii = 0; ii < sdata.expt_keyassignment.length; ii++) {
		keyStrings.push(keyMappings[sdata.expt_keyassignment[ii]]);
	}
	return keyStrings;
}


function gen_phaseTrialVect() {
 	/*
 		generates vector of trial indices for each phase
	*/
	phaseTrialVect = [];
	for (var i = 0; i < parameters.nb_trialsPerPhase.length; i++) {
		phaseTrialVect = phaseTrialVect.concat(colon(1,parameters.nb_trialsPerPhase[i]));
	}
	return phaseTrialVect;
}

function gen_phaseVect() {
	/*
	generates vector with phase indices for each trial
	*/
	phaseVect = [];
	for (var ii=0; ii<parameters.nb_trialsPerPhase.length; ii++) {
		phaseVect = phaseVect.concat(repmat(ii+1,parameters.nb_trialsPerPhase[ii]));
	}
	return phaseVect;
}



function gen_blockVect(){
	/*
		generates array of block indices
	*/
	blockVect = [];

	for(i=0; i<parameters.nb_trialsPerPhase.length; i++) {
		for(j=1; j<=parameters.nb_blocksPerPhase[i];j++) {
			blockVect = blockVect.concat(repmat(j,parameters.nb_trialsPerPhase[i]/parameters.nb_blocksPerPhase[i]));
		}
	}
	return blockVect;
}



function gen_contextVect(){
/*
	generates vector with task indices
	idx 1 = leaf task
	idx 2 = branch task

*/
	contextVect = [];

	for (var i = 0; i < parameters.nb_trialsPerPhase.length; i++) {

		var blockTask1 = repmat(1,parameters.nb_trialsPerPhase[i]/2);
		var blockTask2 = repmat(2,parameters.nb_trialsPerPhase[i]/2);

		if (parameters.task_id.slice(1).join('-') == 'A-B') {
			contextVect = contextVect.concat(blockTask1).concat(blockTask2);
		}
		else if (parameters.task_id.slice(1).join('-') == 'B-A') {
			contextVect = contextVect.concat(blockTask2).concat(blockTask1);
		}
	}
	return contextVect;
}


function gen_leafVect(){
/*
	generates vector of leafiness levels
*/
	return repmat(colon(1,parameters.nb_leafiness),sdata.expt_index.length/parameters.nb_leafiness);
}


function gen_branchVect(){
/*
   generates vector of branchiness levels
*/
	var tmp       = [];
	// train
	var trainBlock = [];

	for(j=1;j<=parameters.nb_branchiness;j++){
		var thisBranch = repmat(j,parameters.nb_branchiness);
		tmp = tmp.concat(thisBranch);
	}
	return repmat(tmp,sdata.expt_index.length/(parameters.nb_branchiness*parameters.nb_leafiness));
}


function gen_catVect() {
/*
 * simplified catvect generation to avoid redundancy
 */
	catVect = [];
	// 1. obtain cat matrices
	condMatrices = loadTaskMatrix(parameters.val_rewAssignment);
	// 2. loop through trials and assign category  accordingly
	for(var ii = 0; ii< sdata.expt_contextIDX.length; ii++) {
    		if (sdata.expt_contextIDX[ii]==1) {
						catVect[ii] = condMatrices.catMat_leaf[sdata.expt_leafIDX[ii]-1][sdata.expt_branchIDX[ii]-1];
        }
	       else {
		        catVect[ii] = condMatrices.catMat_branch[sdata.expt_leafIDX[ii]-1][sdata.expt_branchIDX[ii]-1];
	    	}
	}
	return catVect;
}


function gen_rewardVect() {
/*
 * simplified rewvect generation to avoid redundancy
 */
	rewVect = [];
	// 1. obtain reward and rew matrices
	condMatrices = loadTaskMatrix(parameters.val_rewAssignment);
	// 2. loop through trials and assign rewegory  accordingly
	for(var ii = 0; ii< sdata.expt_contextIDX.length; ii++) {
    		if (sdata.expt_contextIDX[ii]==1) {
			rewVect[ii] = condMatrices.rewMat_leaf[sdata.expt_leafIDX[ii]-1][sdata.expt_branchIDX[ii]-1];
	        }
	        else {
		        rewVect[ii] = condMatrices.rewMat_branch[sdata.expt_leafIDX[ii]-1][sdata.expt_branchIDX[ii]-1];
	    	}
	}
	return rewVect;
}

function gen_exemplarVect() {
/*
	generates vector of exemplar indices	,
	both for training and test phase (note: I want different exemplars between phases, but same between tasks)
*/
	exemplarVect = [];

	// for each phase
	for (var ii = 0; ii < parameters.treeIDsPerPhase.length; ii++) {
		treeExemplars = parameters.treeIDsPerPhase[ii];
		// for each task
		for (var tt = 0; tt<2; tt++) {
			// for each exemplar (in phase vector)
			for (var pp = 0; pp<treeExemplars.length;pp++) {
				exemplar = treeExemplars[pp];
				// add exemplar*25 to exemplarVect
				exemplarVect = exemplarVect.concat(repmat(exemplar,parameters.nb_branchiness*parameters.nb_leafiness));
			}
		}
	}
	return exemplarVect;
}




function shuffle_trials(){
/*
	let's create some chaos...
*/
	 var shuffIDX = [];
	 shuffIDX = mk_shuffIDX();
	 sdata.expt_contextIDX	=   shuffle_vect(shuffIDX,sdata.expt_contextIDX);
	 sdata.expt_leafIDX 	  =      shuffle_vect(shuffIDX,sdata.expt_leafIDX);
	 sdata.expt_branchIDX 	=    shuffle_vect(shuffIDX,sdata.expt_branchIDX);
	 sdata.expt_exemplarIDX =  shuffle_vect(shuffIDX,sdata.expt_exemplarIDX);
	 sdata.expt_catIDX	    =    shuffle_vect(shuffIDX,sdata.expt_catIDX);
	 sdata.expt_rewardIDX   =    shuffle_vect(shuffIDX,sdata.expt_rewardIDX);
	 shuffSwitch = 1;
}


function mk_shuffIDX() {
/*
	shuffle idces
*/
	var shuffIDX = [];
	startIDX = 0;
	// for each phase
	for (var i = 0; i < parameters.nb_trialsPerPhase.length; i++) {
		//console.log('iter' + i);
		switch(bin2num(parameters.task_id.slice(0,1) == 'blocked') && (i==2 || i==4)){ // have blocked main and refresher training
			case true: // if blocked curric and main or refresher block
				// for each task: generate indices
				//console.log('shuffle task, phase ' + i);
				var blockTask1 = colon(startIDX,startIDX+parameters.nb_trialsPerPhase[i]/2-1);
				var blockTask2 = colon(startIDX+parameters.nb_trialsPerPhase[i]/2,startIDX+parameters.nb_trialsPerPhase[i]-1);
				shuffIDX = shuffIDX.concat(shuffle(blockTask1)).concat(shuffle(blockTask2));
				startIDX = startIDX+parameters.nb_trialsPerPhase[i];
				break;
			case false:
				//console.log('shuffle phase' + i);
				// for entire phase
				var blockTasks = colon(startIDX,startIDX+parameters.nb_trialsPerPhase[i]-1);
				shuffIDX = shuffIDX.concat(shuffle(blockTasks));
				startIDX = startIDX+parameters.nb_trialsPerPhase[i];
				break;
		}
	}
	return shuffIDX;
}




function genExemplarSets(setSizes,maxSize) {
	/*
	assigns unique sets of exemplars of setSizes size to each phase
	maxSize determines the maximum value (of exemplar indices).
	*/

	exemplarSets = [];
	exemplarVect = colon(1,maxSize);

	for (var ii=0;ii<setSizes.length;ii++) {
		exemplarVect = shuffle(exemplarVect);
		exemplarSets.push(exemplarVect.splice(0,setSizes[ii]));
	}

	return exemplarSets;
}

function shuffle_vect(shuffIDX,vectToShuffle){
/*
	shuffle vector with set of new indices
*/
	// append shuffled values
	for(var i=0;i<shuffIDX.length;i++){
		vectToShuffle.push(vectToShuffle[shuffIDX[i]]);
	}
	// delete original vector
	vectToShuffle.splice(0,shuffIDX.length);
	return vectToShuffle;
}


function loadTaskMatrix(matID) {
	taskMatrices = {};
	switch (matID){
		case 1:
			 // high high
			 taskMatrices.rewMat_leaf = [[-50,-50,-50,-50,-50],
			 			     [-25,-25,-25,-25,-25],
						     [0,0,0,0,0],
						     [25,25,25,25,25],
						     [50,50,50,50,50]];

			 taskMatrices.catMat_leaf = [[-1,-1,-1,-1,-1],
			 			     [-1,-1,-1,-1,-1],
						     [0,0,0,0,0],
						     [1,1,1,1,1],
						     [1,1,1,1,1]];

			 taskMatrices.rewMat_branch = [[-50,-25,0,25,50],
			 		  	       [-50,-25,0,25,50],
					  	       [-50,-25,0,25,50],
					  	       [-50,-25,0,25,50],
					  	       [-50,-25,0,25,50]];

			 taskMatrices.catMat_branch = [[-1,-1,0,1,1],
			 		  	       [-1,-1,0,1,1],
					  	       [-1,-1,0,1,1],
					  	       [-1,-1,0,1,1],
					  	       [-1,-1,0,1,1]];
			break;
		case 2:
			  // low low
			  taskMatrices.rewMat_leaf = [[50,50,50,50,50],
			  		 	      [25,25,25,25,25],
					 	      [0,0,0,0,0],
					 	      [-25,-25,-25,-25,-25],
					 	      [-50,-50,-50,-50,-50]];

			 taskMatrices.catMat_leaf = [[1,1,1,1,1],
			 			     [1,1,1,1,1],
						     [0,0,0,0,0],
						     [-1,-1,-1,-1,-1],
						     [-1,-1,-1,-1,-1]];

			 taskMatrices.rewMat_branch = [[50,25,0,-25,-50],
			 		  	       [50,25,0,-25,-50],
					  	       [50,25,0,-25,-50],
					  	       [50,25,0,-25,-50],
					  	       [50,25,0,-25,-50]];

			 taskMatrices.catMat_branch = [[1,1,0,-1,-1],
			 		  	       [1,1,0,-1,-1],
					  	       [1,1,0,-1,-1],
					  	       [1,1,0,-1,-1],
					  	       [1,1,0,-1,-1]];
			break;

		case 3:
			  // low high
			  taskMatrices.rewMat_leaf = [[50,50,50,50,50],
			  		 	      [25,25,25,25,25],
					 	      [0,0,0,0,0],
					 	      [-25,-25,-25,-25,-25],
					 	      [-50,-50,-50,-50,-50]];

			 taskMatrices.catMat_leaf = [[1,1,1,1,1],
			 			     [1,1,1,1,1],
						     [0,0,0,0,0],
						     [-1,-1,-1,-1,-1],
						     [-1,-1,-1,-1,-1]];

			 taskMatrices.rewMat_branch = [[-50,-25,0,25,50],
			 		  	       [-50,-25,0,25,50],
					  	       [-50,-25,0,25,50],
					  	       [-50,-25,0,25,50],
					  	       [-50,-25,0,25,50]];

			 taskMatrices.catMat_branch = [[-1,-1,0,1,1],
			 		  	       [-1,-1,0,1,1],
					  	       [-1,-1,0,1,1],
					  	       [-1,-1,0,1,1],
					  	       [-1,-1,0,1,1]];
			break;

		case 4:
			  // high low
			 taskMatrices.rewMat_leaf = [[-50,-50,-50,-50,-50],
			 			     [-25,-25,-25,-25,-25],
						     [0,0,0,0,0],
						     [25,25,25,25,25],
						     [50,50,50,50,50]];

			 taskMatrices.catMat_leaf = [[-1,-1,-1,-1,-1],
			 			     [-1,-1,-1,-1,-1],
						     [0,0,0,0,0],
						     [1,1,1,1,1],
						     [1,1,1,1,1]];

			 taskMatrices.rewMat_branch = [[50,25,0,-25,-50],
			 		  	       [50,25,0,-25,-50],
					  	       [50,25,0,-25,-50],
					  	       [50,25,0,-25,-50],
					  	       [50,25,0,-25,-50]];

			 taskMatrices.catMat_branch = [[1,1,0,-1,-1],
			 		  	       [1,1,0,-1,-1],
					  	       [1,1,0,-1,-1],
					  	       [1,1,0,-1,-1],
					  	       [1,1,0,-1,-1]];
			break;
	}
	return taskMatrices;
}
