
function startDissimRatingExperiment() {
    // clean div
    goWebsite(html_dissim);
    finishedinstructions =  true;
    startedinstructions  = false;
    starteddissimjudge   =  true;
    finisheddissimjudge  = false;
    // set parameters
    setExperiment(); //expt_parameters.js
    // window.location.href ="stimeval_page.html"
    if (instr_id=='dissimrating_pre') {
      edata.exp_starttime = getTimestamp();
    }
    // run the experiment
    runDissimJudgeExp();
    // edata.exp_starttime = getTimestamp();


}


function finishDissimRatingExperiment_pre() {
  saveExperiment();
  instr_id='treetask_main';
  // setInstructions(instr_id);
  // changeInstructions();
  starteddissimjudge   = false;
  finisheddissimjudge  =  true;
  startedinstructions  =  false;
  finishedinstructions = true;
  finishExperiment_data();
}

function finishDissimRatingExperiment_post() {
  saveExperiment();
  starteddissimjudge   =  false;
  finisheddissimjudge  =   true;
  startedinstructions  =  false;
  finishedinstructions =   true;

  finishExperiment_data();
}

function startMainExperiment() {

  finishedinstructions = true;
  startedinstructions =  false;

  // set new variables
  setExperiment(); //expt_parameters.js

  newExperiment();
}


function newExperiment() {

  // clean div
  goWebsite(html_task);
  // set flags
   startedexperiment  = true;
  finishedexperiment = false;

  // run the experiment
  //edata.exp_starttime = getTimestamp();
  runExperiment(); //expt_run.js
}


function finishMainExperiment() {
  saveExperiment();
  instr_id='dissimrating_post';
  startedexperiment  = false;
  finishedexperiment = true;
  startedinstructions = true;
  finishedinstructions = false;
  setInstructions(instr_id);
  changeInstructions();
  goWebsite(html_taskinstr);

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
  else if(!isFullscreen() && starteddissimjudge && !finisheddissimjudge) {
    arena_removeUI();
    finisheddissimjudge = true;
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


function saveExperiment(path_data){
  //set the data to be saved
  var path_tmp  = "data/tmp";

  var alldata = {
      task:       participant_task,
      taskID:     participant_taskID,
      subj:       participant_subject,
      path:       path_tmp,
      id:         participant_id,
      task_id:     JSON.stringify(parameters.task_id),
      key_assign:  JSON.stringify(parameters.keyStr),
      sdata:      JSON.stringify(sdata),
      edata:      JSON.stringify(edata),
      parameters: JSON.stringify(parameters),
      pre_dissimData: JSON.stringify(data_pre),
      params_dissimexp: JSON.stringify(params_exp),
      params_dissimvis: JSON.stringify(params_vis),
      params_dissimui: JSON.stringify(params_ui)
  };

  if(finisheddissimjudge) {
    alldata.move = path_data;
  }
  //send it to the back-end
  logWrite(alldata);
}
