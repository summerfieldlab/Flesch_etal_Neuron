/* **************************************************************************************

User Interface
(c) Timo Flesch, 2016/17
Summerfield Lab, Experimental Psychology Department, University of Oxford

************************************************************************************** */




function arena_setUI() {

// CANVAS
// set up circular canvas
board.paper.object    =   drawPaper(board.rect);
board.circle   =   board.paper.object.circle(board.centre[0],
	board.centre[1], board.radius).attr({'fill': params_vis.circle.colour,
	'opacity': params_vis.circle.opacity});


// BUTTON
// create button to go to next trial
board.buttonBox = board.paper.object.rect(board.centre[0]-50,
	board.centre[1]+window.innerHeight*0.42, 100, 25, 5).attr({
fill: params_ui.button.fill,
stroke: params_ui.button.stroke,
'stroke-width': params_ui.button.width
});
board.buttonText = board.paper.object.text(board.buttonBox.attrs.x +
	board.buttonBox.attrs.width / 2, board.buttonBox.attrs.y +
	board.buttonBox.attrs.height / 2, 'Próxima prueba').attr({
"font-family": params_ui.button.font,
"font-size": params_ui.button.fontsize,
});

board.buttonText.node.setAttribute("class","donthighlight");
board.buttonObject = board.paper.object.set().attr({
cursor: 'pointer'
});
board.buttonObject.push(board.buttonBox);
board.buttonObject.push(board.buttonText);

board.buttonObject.mouseover(function (event) {
	this.oGlow = board.buttonBox.glow({
	    opacity: params_ui.button.glow.opacity,
	    color:   params_ui.button.glow.colour,
	    width:   params_ui.button.glow.width
});
}).mouseout(function (event) {
	this.oGlow.remove();
}).click(function (e) {
	if (numbers.trialCount < params_exp.numTrials) {
		gotoNextTrial();
	}
	else{
		gotoNextTask();
	}
});

}

function arena_removeUI() {

	board.paper.object.remove();
}
