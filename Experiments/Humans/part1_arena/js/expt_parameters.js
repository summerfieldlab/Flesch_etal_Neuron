
/* **************************************************************************************

Sets all the important parameters
(c) Timo Flesch, 2016 [timo.flesch@gmail.com]
Summerfield Lab, Experimental Psychology Department, University of Oxford
Based on earlier version of script by jdobalaguer@gmail.com
************************************************************************************** */

// globals
var sdata;
var edata;
var parameters;
var board;
var coding;
var instr_id;
var participant_taskID;
var participant_subject;

function setExperiment() {


  // EDATA ----------------
  edata = {};
  // expt
  edata.expt_subject = participant_id;
  edata.expt_sex     = participant_gender;
  edata.expt_age     = participant_age;
  edata.expt_task    = participant_task;
  edata.expt_taskid  = participant_taskID;
  edata.expt_subj    = participant_subject;
  // edata.expt_turker  = participant_turker;

  // PARAMETERS -----------
  parameters = {};

  // TIMINGS
  parameters.cue_timein       =  0;
  parameters.cue_timeout      =  500;
  parameters.stimulus_timein  =  500;
  parameters.stimulus_timeout =  2500;
  parameters.response_timeout =  2500;  // response time
  parameters.warnings_timeout =  2500;  // response warning time
  parameters.feedback_timein  =   500;  // delay after action until feedback is displayed
  parameters.feedpos_timeout  =  1500;  // feedback time (good)
  parameters.feedneg_timeout  =  1500;  // feedback time (bad)


  // TASK
  set_subjParams();  // obtains subject-specific params from URL (index.html?id=NUMS)

  parameters.keyStr           =         (parameters.keyassignment)? (['left: accept',' right: reject']) : (['right: accept',' left: reject']);
  parameters.gardenURL        =         "orchards/"
  parameters.treeURL          =         "stims/";
  parameters.keyURL           =         "lib/png/";
  parameters.exemplar_ids_train =       "a,b,c,d".split(',');
  parameters.exemplar_ids_test =        "e,f,g,h".split(',');


  parameters.nb_branchiness   =         5; // how many levels?
  parameters.nb_leafiness     =         5; // how many levels?
  parameters.nb_reps          =         1; // how many repetitions within each block?
  parameters.nb_reps_test     =         1; // how many reps of each task within test ?
  parameters.nb_tasks_test    =         2; // how many tasks within test block? needs to be 2 to cover both tasks!!
  parameters.nb_unique        =         parameters.exemplar_ids_train.length; // 4 unique exemplars
  parameters.nb_trials_train  =         parameters.nb_branchiness*parameters.nb_leafiness*parameters.nb_unique*parameters.nb_reps; //200 trials per training task
  parameters.nb_trials_test   =         parameters.nb_branchiness*parameters.nb_leafiness*parameters.nb_unique*parameters.nb_tasks_test; // 100 trials per training task
  parameters.nb_blocks        =         2; // has to be at least 2 (both tasks
  parameters.nb_blocks_test   =         1; // .. + test)
  parameters.nb_total_train   =         (parameters.nb_blocks)*parameters.nb_trials_train; // 400 training trials in total
  parameters.nb_total_test    =         parameters.nb_trials_test; // 200 training trials in total
  parameters.val_categories   =         [-1,-1,0,1,1];
  parameters.val_rewards      =         [-50,-25,0,25,50];


  // VISUALS
  parameters.visuals              =        {};
  // size
  parameters.visuals.size         =        {};
  parameters.visuals.size.garden  =        [400,400]; // context
  parameters.visuals.size.stim    =        [200,200]; // stimulus tree
  parameters.visuals.size.fbt     =        [150,150]; // feedback tree
  parameters.visuals.size.keyIMG  =        [75,75];   // size of key image

  // colors
  parameters.visuals.cols         =        {};
  parameters.visuals.cols.fbn_pos =        "#080";  // positive feedback
  parameters.visuals.cols.fbn_neg =        "#D00";  // negative feedback
  parameters.visuals.cols.fbn_neu =        "black"; // neutral feedback
  parameters.visuals.cols.fb_bg   =        "grey";  // feedback background

  // misc
  parameters.visuals.blurlvl      =             5;  // how much blur?


  // TEXT
  parameters.txt                          =         {};
  parameters.txt.trainBreakBlocked        =     "A couple of seconds' break. \n \n In the next block, you'll do a different task! \n \n There will be a new orchard and different features of the trees indicate their growth success. \n \n Thus, you need to learn from scratch. \n \n Press the SPACE bar when you're ready to continue";

  parameters.txt.trainInstrBlocked        =     "This is the first block of the experiment. \n \n The orchard will be the same for the entire block and a picture of it will be presented on each trial before you'll see the tree. \n \n Via trial and error, you'll learn the rule that tells you which trees grow best in this particular garden. \n \n Press the " + parameters.keyStr[0].split(':').slice(0,1)[0].toUpperCase() + " arrow key if you would like to plant the tree in your orchard, \n \n or the " + parameters.keyStr[1].split(':').slice(0,1)[0].toUpperCase() + " arrow key if you don't want to plant the tree. \n \n Press the SPACE bar when you're ready to continue";

  parameters.txt.trainBreakInterleaved    =     "A couple of seconds' break. \n \n In the next block, you'll do the same experiment! \n \n Nothing has changed. We just want to give you enough time to discover the rules. \n \n Press the SPACE bar when you're ready to continue";

  parameters.txt.trainInstrInterleaved    =     "This is the first block of the experiment. \n \n Please pay attention to the cue at the beginning of each trial that tells you in which orchard you're currently in! \n \n The cue is just an image of one of the two gardens (North or South). \n \n The cue will change every few trials!. \n \n You need to learn two rules: \n \n Which trees grow best in the North garden, and \n \n Which trees grow best in the South garden? \n \n Press the " + parameters.keyStr[0].split(':').slice(0,1)[0].toUpperCase() + " arrow key if you would like to plant the tree in your orchard, \n \n or the " + parameters.keyStr[1].split(':').slice(0,1)[0].toUpperCase() + " arrow key if you don't want to plant the tree. \n \n Press the SPACE bar when you're ready to continue";

  parameters.txt.testInstructions         =     "Now let's see how well you've learned the rules for both gardens! \n \n In the next block, you'll have to plant trees in both gardens, using the knowledge that you've acquired so far! \n \n At the beginning of each trial, you'll see an image of the orchard you're currently in. (North or South). \n \n Then, you'll see a tree and have to decide whether to plant it or not. \n \n  WE WON'T GIVE YOU FEEDBACK. THIS IS THE TEST SESSION. \n \n Press the SPACE bar when you're ready to continue";




  // SDATA ----------------
  sdata = {};
  // expt
  sdata.expt_index        = []; // trial IDX (total)
  sdata.expt_trial        = []; // trial IDX (within block)
  sdata.expt_block        = []; // block IDX
  sdata.expt_branchIDX    = []; // level of branchiness
  sdata.expt_leafIDX      = []; // level of leafiness
  sdata.expt_rewardIDX    = []; // reward: neg & pos
  sdata.expt_catIDX       = []; // category: plant vs don't plant
  sdata.expt_contextIDX   = []; // task: north vs south
  createSdata();

  // vbxi
  sdata.vbxi_category     = [];



  // resp
  sdata.resp_timestamp    = []; // time when button was pressed
  sdata.resp_reactiontime = []; // RT wrt stimulus ONSET
  sdata.resp_category     = []; // responded category
  sdata.resp_correct      = []; // trial-wise accuracy
  sdata.resp_reward       = []; // trial-wise received reward
  sdata.resp_return       = []; // trial-wise cummulative reward (=return)

  // BOARD ----------------
  board = {}; // will contain the canvas

  // CODING ---------------
  coding = {}; // exp logic
  // index
  coding.task   = 0;
  coding.index  = 0;
  coding.trial  = 0;
  coding.block  = 0;
  coding.return = 0;
  // other
  coding.answering = false;
  coding.timestamp = NaN;
}

function set_subjParams() {
  /*
    grabs params from url argument
    input structure:
    ?subject=1&task=1
    subject: participant number
    task: task number (1 = arena, 2 = main)
    id: only for main, this is defines task order and other parameters (e.g. key mapping)
  */
  input = getQueryParams();

  participant_taskID = input.task; // note: participant_task is hard coded
  participant_subject = input.subject;

  if (typeof(input.id)=="undefined" ) {
    parameters.task_id       = 'blocked-A-B'.split('-'); // blocked north south
    parameters.val_rewAssignment = 	  1; // no flipped assignments, cardinal boundary
    parameters.keyassignment =            0; // l-no r-yes
    parameters.blockiness    =          200; // how many trials of one task per block?
  }

  else {
    input.id = input.id.split('').map(Number);
    // 1. curriculum
    switch (input.id[0]) {
      case  1:
        parameters.task_id = 'blocked-A-B'.split('-');
        break;
      case 2:
        parameters.task_id = 'blocked-B-A'.split('-');
        break;
      case 3:
        parameters.task_id = 'interleaved-A-B'.split('-');
        break;
      default:
        parameters.task_id = 'blocked-A-B'.split('-');
        break;
    }

    // 2. reward & boundary
    parameters.val_rewAssignment   = input.id[1]; // second and third items

    // 3. keys
    parameters.keyassignment =    (input.id[2]==3) ? (randi(2,1)) : (input.id[2]); // 0: l-no r-yes, 1: l-yes r-no

    // 4. blockiness (becomes irrelevant for interleaved design)
    switch (input.id[3]) {
	case 1:
		parameters.blockiness = 2;
		break;
	case 2:
		parameters.blockiness = 20;
		break;
	case 3:
		parameters.blockiness = 200;
		break;
	default:
		parameters.blockiness = 200;
		break;
    }
    // however, if interleaved, set back to 200
    if(input.id[0]==3){
	    parameters.blockiness = 200;
    }
  }
  console.log(parameters.task_id)
  console.log(parameters.blockiness)
  console.log(parameters.val_rewAssignment)
}
