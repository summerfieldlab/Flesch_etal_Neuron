
function drawClock(centre) {
  board.clock = {};
  board.clock.centre = centre;
  board.clock.text   = '%02d : %02d : %02d';
  board.clock.object = drawText(board.paper.object,board.clock.centre,'');
  board.clock.object.attr({"font-size": board.font_medsize});
  updateClock();
 }

function updateClock() {
  // update text
  var gs      = getSecs(params_exp.startTime);
  var hours   = floor(gs / 3600);
  gs = gs - (3600 * hours);
  var minutes = floor(gs / 60);
  gs = gs - (  60 * minutes);
  var seconds = gs; 
  board.clock.object.attr({"text": sprintf(board.clock.text,hours,minutes,seconds)});
  // set timeout
  board.clock.timeout = setTimeout(updateClock,1000);
}

function stopClock() {
  clearTimeout(board.clock.timeout);
}

function showClock() {
  board.clock.object.attr({"opacity": 1});
}

function hideClock() {
  board.clock.object.attr({"opacity": 0});
}

function removeClock(){
  board.clock.object.remove();
}
