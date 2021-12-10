/* **************************************************************************************

Creates experimental conditions.
(c) Timo Flesch, 2016 [timo.flesch@gmail.com]
Summerfield Lab, Experimental Psychology Department, University of Oxford

************************************************************************************** */
shuffSwitch = 0;

function createSdata(){
/*
	here happens all the magic
*/
	// index from very first to very last trial of WHOLE exp
	sdata.expt_index 	= colon(1,parameters.nb_total_train+parameters.nb_total_test);

	// index from first to last trial within each block
	sdata.expt_trial 	= repmat(colon(1,parameters.nb_trials_train),parameters.nb_blocks).concat(repmat(colon(1,parameters.nb_trials_test),parameters.nb_blocks_test));

	// generate block indices
	sdata.expt_block 	= gen_blockVect();

	// generate context indices (to distinguish between branch and leaf tasks)
	sdata.expt_contextIDX   = gen_contextVect();

	// generate session indices (training 1 vs test 2)
	sdata.expt_sessIDX      = gen_sessVect();

	// generate branchiness indices (rep([1,1,1,1,1,2,2,2......5,5,5,5]))
	sdata.expt_branchIDX 	= gen_branchVect();

	// generate leafiness indices (rep([1,2,3,4,5,1,2,3,4,5....]))
	sdata.expt_leafIDX 	= gen_leafVect();

	// generate vector with exemplar indices [I have several exemplars per level]
	sdata.expt_exemplarIDX  = gen_exemplarVect();

	// generate category label indices (depending on task order)
	sdata.expt_catIDX	= gen_catVect();

	// generate vector with reward values, depending on task and feature level
	sdata.expt_rewardIDX    = gen_rewardVect();

	// generate vector with "optimal" return up to each trial n (for performance assessment)
	sdata.expt_returnOPT    = [];  //done on the fly
	sdata.expt_rewardOPT    = [];


	// magic dust
	shuffle_trials(shuffSwitch);

	if (parameters.blockiness < 200 && shuffSwitch == 1) {
		shrink_blocks();
	}
}


function gen_blockVect(){
/*
	generates array of block indices
*/
	tmp = new Array();
	// training blocks
	for(i=1;i<=parameters.nb_blocks;i++){
		var thisBlock = repmat(i,parameters.nb_trials_train);
		tmp = tmp.concat(thisBlock);
	}
	// test block
	for(i=parameters.nb_blocks+1;i<=parameters.nb_blocks+parameters.nb_blocks_test;i++) {
		var thisBlock = repmat(i,parameters.nb_trials_test);
		tmp = tmp.concat(thisBlock);
	}
	return tmp;
}


function gen_sessVect() {
/*
	generates vector with session indices
	idx 1 = training session
	idx 2 = test session
*/
	return repmat(1,parameters.nb_total_train).concat(repmat(2,parameters.nb_total_test));
}


function gen_contextVect(){
/*
	generates vector with task indices
	idx 1 = leaf task
	idx 2 = branch task
	ix  3 = test task [need to get rid of this....]
*/

	var blockTask1 = repmat(1,parameters.nb_trials_train);
	var blockTask2 = repmat(2,parameters.nb_trials_train);
	var blockTask3 = repmat(1,parameters.nb_trials_test/parameters.nb_tasks_test).concat(repmat(2,parameters.nb_trials_test/parameters.nb_tasks_test)); // half one task, half other task
	if (parameters.task_id.slice(1).join('-') == 'A-B') {
		return blockTask1.concat(blockTask2).concat(blockTask3);
	}
	else if (parameters.task_id.slice(1).join('-') == 'B-A') {
		return blockTask2.concat(blockTask1).concat(blockTask3);
	}
}


function gen_leafVect(){
/*
	generates vector of leafiness levels
*/
	// training & test
	tmp = repmat(colon(1,parameters.nb_leafiness),sdata.expt_index.length/parameters.nb_leafiness);

	return tmp;
}


function gen_branchVect(){
/*
   generates vector of branchiness levels
*/
	var tmp       = new Array();
	// train
	var trainBlock = [];
	for(i=1;i<=parameters.nb_unique*parameters.nb_blocks;i++){
		for(j=1;j<=parameters.nb_branchiness;j++){
			var thisBranch = repmat(j,parameters.nb_leafiness);
			tmp = tmp.concat(thisBranch);
		}
	}
	trainBlock = repmat(tmp,parameters.nb_reps);

	//test
	var testBlock = [];
	var tmp       = [];
	for(i=1;i<=parameters.nb_unique*parameters.nb_blocks_test;i++){
		for(j=1;j<=parameters.nb_branchiness;j++){
			var thisBranch = repmat(j,parameters.nb_leafiness);
			tmp = tmp.concat(thisBranch);
		}
	}
	testBlock = repmat(tmp,parameters.nb_tasks_test);
	return trainBlock.concat(testBlock);
}


function gen_catVect() {
/* 
 * simplified catvect generation to avoid redundancy
 */
	catVect = [];
	// 1. obtain catard and cat matrices 
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
	both for training and test phase (note: I want different exemplars)
*/
	exemplarVectTrain = [];
	for(var blockID=0;blockID<parameters.nb_blocks;blockID++) {
		for(var ii=0;ii<parameters.exemplar_ids_train.length;ii++) {
			exemplarVectTrain = exemplarVectTrain.concat(repmat(parameters.exemplar_ids_train[ii],parameters.nb_branchiness*parameters.nb_leafiness));

		}
	}

	exemplarVectTest = [];
	for(var blockID=0;blockID<parameters.nb_blocks_test;blockID++) {
		for(var ii=0;ii<parameters.exemplar_ids_test.length;ii++) {
			exemplarVectTest = exemplarVectTest.concat(repmat(parameters.exemplar_ids_test[ii],parameters.nb_branchiness*parameters.nb_leafiness));

		}
	}


	return repmat(exemplarVectTrain,parameters.nb_reps).concat(repmat(exemplarVectTest,parameters.nb_reps_test*parameters.nb_tasks_test));
}



function shrink_blocks() {
/*
	if parameters.blockiness <200, this funct reduces blocksize accordingly	
	important: run after shuffling!!!
 */

	 sdata.expt_leafIDX 	=      shrink_vect(sdata.expt_leafIDX);
	 sdata.expt_branchIDX 	=    shrink_vect(sdata.expt_branchIDX);
	 sdata.expt_catIDX	=       shrink_vect(sdata.expt_catIDX);
	 sdata.expt_exemplarIDX =  shrink_vect(sdata.expt_exemplarIDX);
	 sdata.expt_contextIDX	=   shrink_vect(sdata.expt_contextIDX);
	 sdata.expt_rewardIDX   =    shrink_vect(sdata.expt_rewardIDX);
}

function shrink_vect(thisVect) {
/*
	takes a vector of two training blocks and splits into smaller blocks
 */
	// init empty vect
	newVect = [];
	// extract first and second task:
	trialsTrain1 = thisVect.slice(0,parameters.nb_trials_train);
	trialsTrain2 = thisVect.slice(parameters.nb_trials_train,parameters.nb_total_train);
	trialsTest   = thisVect.slice(parameters.nb_total_train,thisVect.length);
	// loop from 0 over blockiness to vectlength-blockiness
	//for each iteration: add #blockiness items from first and second task successively to new rect, return this
	for(var ii=0;ii<=trialsTrain1.length-parameters.blockiness;ii=ii+parameters.blockiness) {
		newVect = newVect.concat(trialsTrain1.slice(ii,ii+parameters.blockiness),trialsTrain2.slice(ii,ii+parameters.blockiness));
	}
	newVect = newVect.concat(trialsTest);
	return newVect;
}


function shuffle_trials(){
/*
	let's create some chaos...
*/
	 var shuffIDX = [];
	 shuffIDX = mk_shuffIDX();
	 sdata.expt_leafIDX 	=      shuffle_vect(shuffIDX,sdata.expt_leafIDX);
	 sdata.expt_branchIDX 	=    shuffle_vect(shuffIDX,sdata.expt_branchIDX);
	 sdata.expt_catIDX	=       shuffle_vect(shuffIDX,sdata.expt_catIDX);
	 sdata.expt_exemplarIDX =  shuffle_vect(shuffIDX,sdata.expt_exemplarIDX);
	 sdata.expt_contextIDX	=   shuffle_vect(shuffIDX,sdata.expt_contextIDX);
	 sdata.expt_rewardIDX   =    shuffle_vect(shuffIDX,sdata.expt_rewardIDX);
	 shuffSwitch = 1;
}


function mk_shuffIDX() {
/*
	shuffle idces
*/
	var shuffIDX = [];
	var startIDX =  0;

	// training
	switch(bin2num(parameters.task_id.slice(0,1) == 'blocked')){
		case 1:
			for(var i=1;i<=parameters.nb_blocks;i++){
				var tmp = [];
				tmp = shuffle(colon(startIDX,startIDX+parameters.nb_trials_train-1));
				shuffIDX = shuffIDX.concat(tmp);
				startIDX = startIDX+parameters.nb_trials_train;
			}

			break;

		case 0:
			shuffIDX = shuffle(colon(0,parameters.nb_total_train-1));

			break;
    }

    // test
    shuffIDX = shuffIDX.concat(shuffle(colon(parameters.nb_total_train,parameters.nb_total_train+parameters.nb_total_test-1)));
    return shuffIDX;
}


function shuffle_vect(shuffIDX,vectToShuffle){
/*
	shuffle vector with set of new indices
*/
	for(var i=0;i<shuffIDX.length;i++){
		vectToShuffle.push(vectToShuffle[shuffIDX[i]]);
	}
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
	
		case 5:
			 // high high
			 taskMatrices.rewMat_leaf = [[-50,-50,-50,-25,0],
			 			     [-50,-50,-25,0,25],
						     [-50,-25,0,25,50],
						     [-25,0,25,50,50],
						     [0,25,50,50,50]];
	
			 taskMatrices.catMat_leaf = [[-1,-1,-1,-1,0],
			 			     [-1,-1,-1,0,1],
						     [-1,-1,0,1,1],
						     [-1,0,1,1,1],
						     [0,1,1,1,1]];
			
			 taskMatrices.rewMat_branch = [[0,25,50,50,50],
			 		  	       [-25,0,25,50,50],
					  	       [-50,-25,0,25,50],
					  	       [-50,-50,-25,0,25],
					  	       [-50,-50,-50,-25,0]];
			 
			 taskMatrices.catMat_branch = [[0,1,1,1,1],
			 		  	       [-1,0,1,1,1],
					  	       [-1,-1,0,1,1],
					  	       [-1,-1,-1,0,1],
					  	       [-1,-1,-1,-1,0]];
			break;
		case 6:	  
			  // low low
			  taskMatrices.rewMat_leaf = [[50,50,50,25,0],
			 			      [50,50,25,0,-25],
						      [50,25,0,-25,-50],
						      [25,0,-25,-50,-50],
						      [0,-25,-50,-50,-50]];
			
			 taskMatrices.catMat_leaf = [[1,1,1,1,0],
			 			     [1,1,1,0,-1],
						     [1,1,0,-1,-1],
						     [1,0,-1,-1,-1],
						     [0,-1,-1,-1,-1]];
			
			 taskMatrices.rewMat_branch = [[0,-25,-50,-50,-50],
			 		  	       [25,0,-25,-50,-50],
					  	       [50,25,0,-25,-50],
					  	       [50,50,25,0,-25],
					  	       [50,50,50,25,0]];

			 taskMatrices.catMat_branch = [[0,-1,-1,-1,-1],
			 		  	       [1,0,-1,-1,-1],
					  	       [1,1,0,-1,-1],
					  	       [1,1,1,0,-1],
					  	       [1,1,1,1,0]];
			break;
		
		case 7:
			  // low high		  
			  taskMatrices.rewMat_leaf = [[50,50,50,25,0],
			 			      [50,50,25,0,-25],
						      [50,25,0,-25,-50],
						      [25,0,-25,-50,-50],
						      [0,-25,-50,-50,-50]];
			
			 taskMatrices.catMat_leaf = [[1,1,1,1,0],
			 			     [1,1,1,0,-1],
						     [1,1,0,-1,-1],
						     [1,0,-1,-1,-1],
						     [0,-1,-1,-1,-1]];
			
			 taskMatrices.rewMat_branch = [[0,25,50,50,50],
			 		  	       [-25,0,25,50,50],
					  	       [-50,-25,0,25,50],
					  	       [-50,-50,-25,0,25],
					  	       [-50,-50,-50,-25,0]];
			 
			 taskMatrices.catMat_branch = [[0,1,1,1,1],
			 		  	       [-1,0,1,1,1],
					  	       [-1,-1,0,1,1],
					  	       [-1,-1,-1,0,1],
					  	       [-1,-1,-1,-1,0]];
			break;
		
		case 8:
			  // high low 		  
			 taskMatrices.rewMat_leaf = [[-50,-50,-50,-25,0],
			 			     [-50,-50,-25,0,25],
						     [-50,-25,0,25,50],
						     [-25,0,25,50,50],
						     [0,25,50,50,50]];
	
			 taskMatrices.catMat_leaf = [[-1,-1,-1,-1,0],
			 			     [-1,-1,-1,0,1],
						     [-1,-1,0,1,1],
						     [-1,0,1,1,1],
						     [0,1,1,1,1]];
			
			 taskMatrices.rewMat_branch = [[0,-25,-50,-50,-50],
			 		  	       [25,0,-25,-50,-50],
					  	       [50,25,0,-25,-50],
					  	       [50,50,25,0,-25],
					  	       [50,50,50,25,0]];

			 taskMatrices.catMat_branch = [[0,-1,-1,-1,-1],
			 		  	       [1,0,-1,-1,-1],
					  	       [1,1,0,-1,-1],
					  	       [1,1,1,0,-1],
					  	       [1,1,1,1,0]];
			break;
	}
	return taskMatrices;
}		  
