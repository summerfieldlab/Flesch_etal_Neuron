/* **************************************************************************************

Parameters for  Arena Task
(c) Timo Flesch, 2016/17
Summerfield Lab, Experimental Psychology Department, University of Oxford

************************************************************************************** */

var FLAG_DEBUG = 0;

var params_vis 	= {};
var params_exp 	= {};
var params_ui   = {};
var stim 		= {};
var numbers 	= {};
var board 		= {};
var data_pre    = {};
var data_post   = {};


function arena_setParams() {

	// CANVAS
	board.width   =                    window.innerWidth;
	board.height  =                   window.innerHeight;
	board.centre  =   [Math.floor(0.5*window.innerWidth),
	                 Math.floor(0.5*window.innerHeight)];
	board.radius  =                    0.40*board.height;
	board.rect 	  =       [0,0,board.width,board.height];
	board.paper   = {};

	// VISUALS
	params_vis.stimSize           =  0.25*board.radius;
	params_vis.circle             =                {};
	params_vis.circle.colour      =            'grey';
	params_vis.circle.opacity     =               '1';
	params_vis.drawBB 	          =                 0;


	// USER INTERFACE
	params_ui.button              =           {};
	params_ui.button.fill         = 'lightgreen';
	params_ui.button.stroke       =    '#3b4449';
	params_ui.button.width        =            2;
	params_ui.button.font         =  "Helvetica";
	params_ui.button.fontsize     =         "12";

	params_ui.button.glow         =           {};
	params_ui.button.glow.opacity =       '0.85';
	params_ui.button.glow.colour  =      'green';
	params_ui.button.glow.width   =          '2';


	// EXPERIMENT
	// params_exp.startTime = getTimestamp();
	params_exp.treeDir     =                                './stims/';
	params_exp.exemplars   =          ['a','b','c','d','e','f','g','h'];
	params_exp.numTrials   =                                          6;
	params_exp.numStimuli  =                                         25;
	params_exp.numTotal    = params_exp.numTrials*params_exp.numStimuli;


	// STIMULI
	stim.obj         = []; // container for stim objects
	stim.coordsOrig  = []; // saves coordinates of stim objects
	stim.coordsFinal = []; // submitted stimulus coordinates

	stim.stimVect  =  set_exp_stimVect();
	stim.stimNames = set_exp_fileNames();


	// TRIAL COUNTERS
	numbers.trialCount = 1;
	numbers.stimCount  = 0;
}
