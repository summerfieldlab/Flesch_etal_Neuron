// -----------------------------------------------------------------------------
// EXPERIMENT
// -----------------------------------------------------------------------------
function newExperiment() {
  /*
  sets up a new experiment by setting all parameters and
  displaying the instructions for the first phase
  */
  setExperiment();
  edata.exp_starttime = getTimestamp();
  coding.phase = 1; // coding.phase = 1
  startedexperiment    =  true;
  finishedexperiment   = false;
  instructPhase();
}

function startExperiment() {
  /*
  starts the Experiment by adding key bindings
  drawing the contents and
  launching a new phase
  */
    board = {};

    // BIND KEYS
    // jwerty.key('←',handleLeft);
    // jwerty.key('→',handleRight);
    jwerty.key('f',handleLeft);
    jwerty.key('j',handleRight);
    jwerty.key('space',startBlock);

    // START
    addPaper();
    newSession();

}


function stopExperiment() {
  if(startedexperiment){
    // set flags
    finishedexperiment = true;
    // remove
    removeFeedback();
    removeStimuli();
    removeInstructions();
    removeCountdown();
    removePaper();
    removeCue();
  }
}

function finishExperiment_resize() {
  //instructions screen
  if(!isFullscreen() && $('#startButton').length>0){
    document.getElementById('startButton').disabled = true;
  }
  //task screen
  if(!isFullscreen() && startedexperiment && !finishedexperiment) {
    stopExperiment();
    saveExperiment("data/resize");
    goWebsite(html_errscreen);
  }
   else if(!isFullscreen() && startedinstructions && !finishedinstructions) {
    saveExperiment("data/resize");
    goWebsite(html_errscreen);

  }
}


function finishExperiment_noresponse() {
  // stop the experiment
  edata.exp_finishtime = getTimestamp();
  stopExperiment();
  // send the data
  saveExperiment("data/noresponse");
  goWebsite(html_errnoresp);
}


function finishExperiment_data() {
  // stop the experiment
  console.log('finished experiment');
  edata.exp_finishtime = getTimestamp();
  //stopExperiment();
  // send the data
  goWebsite(html_sending);
  saveExperiment("data/data");
  goWebsite(html_vercode);
}



function saveExperiment(path_data){
  //set the data to be saved
  var path_tmp  = "data/tmp";

  var alldata = {
      task:       participant_task,
      path:       path_tmp,
      taskID:     participant_taskID,
      subj:       participant_subject,
      id:         participant_id,
      task_id:     JSON.stringify(parameters.task_id),
      // key_assign:  JSON.stringify(parameters.keyStr),
      sdata:      JSON.stringify(sdata),
      edata:      JSON.stringify(edata),
      parameters: JSON.stringify(parameters),
  };

  if(finishedexperiment) {
    alldata.move = path_data;
  }
  //send it to the back-end
  logWrite(alldata);
}
