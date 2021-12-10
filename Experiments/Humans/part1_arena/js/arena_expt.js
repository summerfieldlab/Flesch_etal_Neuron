/* **************************************************************************************

File IO, etc
(c) Timo Flesch, 2016/17
Summerfield Lab, Experimental Psychology Department, University of Oxford

************************************************************************************** */

function exp_saveStimCoords() {
/*
	saves coordinates of final arrangement of stims
*/
	for (var ii=0;ii<board.set.length;ii++) {
		stimIDX = ii+(params_exp.numStimuli*(numbers.trialCount-1));
		stim.coordsFinal[stimIDX] = [board.set[ii].attr("x"), board.set[ii].attr("y")];
	}
}


function exp_saveParticipantData() {
    if (instr_id=='dissimrating_pre') {

        data_pre.pre_trialID           =  stim.stimVect.trialID;
        data_pre.pre_stimExemplar      = stim.stimVect.exemplar;
        data_pre.pre_stimBranchLevel   =   stim.stimVect.branch;
        data_pre.pre_stimLeafLevel     =     stim.stimVect.leaf;
        data_pre.pre_stimCoords_Final  =       stim.coordsFinal;
        data_pre.pre_stimCoords_Orig   =        stim.coordsOrig;
        data_pre.pre_stimNames         =         stim.stimNames;
    }
    else {
        data_post.post_trialID           =  stim.stimVect.trialID;
        data_post.post_stimExemplar      = stim.stimVect.exemplar;
        data_post.post_stimBranchLevel   =   stim.stimVect.branch;
        data_post.post_stimLeafLevel     =     stim.stimVect.leaf;
        data_post.post_stimCoords_Final  =       stim.coordsFinal;
        data_post.post_stimCoords_Orig   =        stim.coordsOrig;
        data_post.post_stimNames         =         stim.stimNames;
    }

}


function exp_exportData() {
/*
	exports data to JSON file
*/

// first, build data structure
var data = {
    trialID:           stim.stimVect.trialID,
    stimExemplar:     stim.stimVect.exemplar,
    stimBranchLevel:    stim.stimVect.branch,
    stimLeafLevel:        stim.stimVect.leaf,
    stimCoords_Final:       stim.coordsFinal,
    stimCoords_Orig:         stim.coordsOrig,
    stimNames:                stim.stimNames
};

// second, convert data to JSON and send to backend
$.ajax({
    type : "POST",
    url : "../php/io.php",
    data : {
        json : JSON.stringify(data)
    }
});

}
