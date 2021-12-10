
/* **************************************************************************************

Launches blocks and trials
(c) Timo Flesch, 2016 [timo.flesch@gmail.com]
Summerfield Lab, Experimental Psychology Department, University of Oxford
************************************************************************************** */


// -----------------------------------------------------------------------------
// PHASES
// -----------------------------------------------------------------------------
function startPhase() {

  finishedinstructions =  true;
  startedinstructions  = false;
  startedexperiment    =  true;
  finishedexperiment   = false;

  goWebsite(html_task);

  board = {};

  // BIND KEYS
  // jwerty.key('←',handleLeft);
  // jwerty.key('→',handleRight);
  jwerty.key('f',handleLeft);
  jwerty.key('j',handleRight);
  jwerty.key('space',startBlock);

  // START
  addPaper();
  hideTrial();
  hideFeedback();
  hideCue();
  hideKeys();

  showStartMessage(); //please press space to start experiment blah blah blah
  coding.newblock  = true; // allows participants to start block (call startBlock by pressing space)
}

function stopPhase() {
// removes everything from canvas and
//saves phase progress as tmp file

saveExperiment();
removeFeedback();
removeStimuli();
removeCountdown();
removePaper();
removeCue();
}

function nextPhase() {
  // increments coding.phase and calls instructions
  coding.phase++;
}

function instructPhase() {
// shows instructions for new phase.
// startPhase is called once subject has finished reading instructions
setInstructions(); // sets instruction data structure
changeInstructions(); // injects data structure (see above) into html file
goWebsite(html_taskinstr);
startedinstructions  =  true;
finishedinstructions = false;
}


// -----------------------------------------------------------------------------
// BLOCKS
// -----------------------------------------------------------------------------

function startBlock() {
  /*
    begins with first trial in new block (called when space is pressed during showBlock())
  */
  if (coding.newblock){
    coding.newblock  = false;
    hideBlock();
    newTrial();
  }
}


// -----------------------------------------------------------------------------
// TRIALS
// -----------------------------------------------------------------------------

function nextTrial() {
  /*
    changes to next trial
  */

  // INCREMENT TRIAL
  coding.index++;
  coding.trial++;

  // INCREMENT PHASE
  if (coding.trial==parameters.nb_trialsPerPhase[coding.phase-1]) { // if last trial in phase
    // if not last phase, start next phase
    if (coding.phase < Math.max.apply(null,sdata.expt_phase)) {
      stopPhase(); // remove everyting, save tmp file
      nextPhase(); // increment coding.phase by 1
      instructPhase(); // display instructions for next phase , which ultimately calls startPhase() (which in turn calls startExperiment)
      coding.trial = 0;
    } // else stop experiment
    else {
      stopExperiment();
      finishExperiment_data();
    }
  }
  // if it's not last trial of phase or experiment, but last trial of block
  else if ((sdata.expt_block[coding.index] != sdata.expt_block[coding.index-1]) && coding.index >1 && coding.index != parameters.nb_trialsTotal) {
    coding.block++;
    // show break text and begin new block NOTE: break text is curric specific. e.g. if blocked and there is context switch, it should say that context is going to switch. otherwise it should just say that there is a break
    coding.newblock  = true;
    hideTrial();
    hideFeedback();
    hideCue();
    hideKeys();
    showBreakMessage();
  }
  else { // if there's nothing special about this transition, just start a new trial
  // NEW TRIAL
  newTrial();
  }
}

function newTrial() {
/*
  !!most important function, defines what happens during a trial!!
*/

  if (!startedexperiment) { return; }

  // step 1: remove old crap
  removeCue();        //delete old rect
  removeStimuli();    // remove old tree
  removeKeys();       // remove keys
  removeFeedback();   // remove feedback (rects and values)

  // step 2: update content
  updateCue();       //draw new context rect
  updateStimuli();   // draw new tree
  updateKeys();      // draw new key assignmenth
  updateFeedback();  // draw new feedback

  // step 3: hide new content
  hideCue();
  hideStimuli();
  hideKeys();
  hideFeedback();
  hideFixation(board.fixation);

  // step 4: brief iti, then show cue/context
  setTimeout(beginTrial,parameters.iti_timeout);


}
