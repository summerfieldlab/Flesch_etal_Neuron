
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

function setExperiment() {


  // EDATA ----------------
  edata = {};
  // expt
  edata.expt_subject = participant_id;
  edata.expt_sex     = participant_gender;
  edata.expt_age     = participant_age;
  edata.expt_task    = participant_task;
  // edata.expt_turker  = participant_turker;

  // PARAMETERS -----------
  parameters = {};

  // TIMINGS
  parameters.cue_timein       =  0;
  parameters.cue_timeout      =  1000;
  parameters.stimulus_timein  =  1000;
  parameters.stimulus_timeout =  1500;
  parameters.response_timeout =  1500;  // response time
  parameters.warnings_timeout =  1500;  // response warning time
  parameters.feedback_timein  =   500;  // delay after action until feedback is displayed
  parameters.feedpos_timeout  =  1000;  // feedback time (good)
  parameters.feedneg_timeout  =  1000;  // feedback time (bad)
  parameters.iti_timeout      =  1000;


  // TASK
  set_subjParams();  // obtains subject-specific params from URL (index.html?id=NUMS)

  // parameters.keyStr           =         (parameters.keyassignment)? (['left: accept',' right: reject']) : (['right: accept',' left: reject']);
  parameters.gardenURL        =         "orchards/";
  parameters.treeURL          =         "stims/";
  parameters.keyURL           =         "lib/png/";

  // range of feature dimensions
  parameters.nb_branchiness   =         5; // how many levels?
  parameters.nb_leafiness     =         5; // how many levels?


  // curriculum
  parameters.blockiness       =   25;// 450; // MAIN phase (p3) numbers of consecutive trials within one context

  // numbers of exemplar sets (full 5x5) per phase
  // day 1 mini training
  // day 1 baseline test
  // day 1 main training
  // day 2 refresher (matlab, behav)
  // day 2 main test (matlab, scanner)
  // max tree id (I suspect I've generated 50 per level)

  parameters.nb_treesPerPhase      =  [1,4,18];
  // parameters.nb_treesPerPhase      =  [1,1,2];
  parameters.scan_nb_treesPerPhase =    [1,12]; // first entry is refresher outside of scanner, second entry is inside scanner
  parameters.nb_treesToSample      =        50;

  // assignment of exemplars to phases
  allTreeIDs   =       genExemplarSets(parameters.nb_treesPerPhase.concat(parameters.scan_nb_treesPerPhase),parameters.nb_treesToSample);
  parameters.treeIDsPerPhase = allTreeIDs.slice(0,parameters.nb_treesPerPhase.length);
  parameters.scan_treeIDsPerPhase = allTreeIDs.slice(parameters.nb_treesPerPhase.length);

  // number of trials per phase
  // parameters.nb_trialsPerPhase = [50,50,50];// [25*1*2,25*4*2,25*18*2,25*1*2,25*12*2]; // trials for each phase
  parameters.nb_trialsPerPhase = [25*parameters.nb_treesPerPhase[0]*2,25*parameters.nb_treesPerPhase[1]*2,25*parameters.nb_treesPerPhase[2]*2]; // trials for each phase

  parameters.nb_trialsTotal =  sum(parameters.nb_trialsPerPhase);

 // number of blocks per phase (breaks are between blocks)
 // parameters.nb_blocksPerPhase =    [2,2,2];//[1,1,4,1,4];
 parameters.nb_blocksPerPhase =    [1,1,4];



  parameters.val_categories   =     [-1,-1,0,1,1];
  parameters.val_rewards      = [-50,-25,0,25,50];


  // VISUALS
  parameters.visuals              =        {};
  // size
  parameters.visuals.size         =        {};
  parameters.visuals.size.garden  =        [550,550]; // context
  parameters.visuals.size.stim    =        [250,250]; // stimulus tree
  parameters.visuals.size.fbt     =        [150,150]; // feedback tree
  parameters.visuals.size.keyIMG  =          [75,75];   // 75,75 size of key image

  // colors
  parameters.visuals.cols           =     {};
  parameters.visuals.cols.fbn_pos   =     "#0F0";  // 080 positive feedback
  parameters.visuals.cols.fbn_neg   =     "#F00";  // D00 negative feedback
  parameters.visuals.cols.fbn_neu   =    "black"; // neutral feedback
  parameters.visuals.cols.fb_bg     =     "grey";  // feedback background
  parameters.visuals.cols.ctx_north =     "blue";
  parameters.visuals.cols.ctx_south =   "orange";

  // misc
  parameters.visuals.blurlvl      =             5;  // how much blur?

  // TEXT
  parameters.txt                  = {};
  parameters.txt.break_taskSwitch = "Descanso de unos segundos. \n \n EN EL PROXIMO BLOQUE ESTARÁS EN OTRO HUERTO!! :) \n \n Características diferentes determinan el crecimiento del árbol en este huerto.  \n \n Pulsa la banda espaciadora para continuar";
  parameters.txt.break_taskStay   = "Descanso de unos segundos.  \n \n Pulsa la banda espaciadora para continuar";
  parameters.txt.startMessage     = " Pulsa la banda espaciadora para empezar el experimento.";

  parameters.txt.accept = "aceptar";
  parameters.txt.reject = "rechazar";




  // SDATA ----------------
  sdata = {};
  // expt
  sdata.expt_index        = []; // trial IDX (total)
  sdata.expt_trial        = []; // trial IDX (within phases)
  sdata.expt_phase        = []; // phase IDX
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
  coding.phase  = 0;
  // other
  coding.answering = false;
  coding.timestamp = NaN;
}



function set_subjParams() {
  /*
    grabs params from url argument
    note: this is task 2 !!!
    input structure: subject=1&task=1&id=ab
    ?id=ab
    a = 1/2/3 = bAB,bBA,interl, default bAB (1)
    b = 1:4   = cardinal: nn,ff,fn,nf
    subject: participant number
    task: task number (1 = arena, 2 = main)
*/
  input = getQueryParams();

  participant_taskID = input.task; // note: participant_task is hard coded
  participant_subject = input.subject;
  parameters.keyassignment     =            1; // always random

  if (typeof(input.id)=="undefined" ) {
    parameters.task_id           = 'blocked-A-B'.split('-'); // blocked north south
    parameters.val_rewAssignment = 	  1; // no flipped assignments, cardinal boundary
  }

  else {
    input.id = input.id.split('').map(Number);
    // 1. curriculum
    switch (input.id[0]) {
      case  1:
        parameters.task_id = 'blocked-A-B'.split('-'); // north, south (leaf,branch)
        break;
      case 2:
        parameters.task_id = 'blocked-B-A'.split('-'); // south, north (branch,leaf)
        break;
      case 3:
        parameters.task_id = 'interleaved-A-B'.split('-');
        break;
      default:
        parameters.task_id = 'blocked-A-B'.split('-');
        break;
    }

    // 2. reward
    parameters.val_rewAssignment   = input.id[1]; // second entry in vect
  }
  console.log(parameters.task_id);
  console.log(parameters.blockiness);
  console.log(parameters.val_rewAssignment);

}
