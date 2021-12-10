
function pathTriangle(points) {
  var path  =   "M" + points[0][0] + "," + points[0][1]
              + "L" + points[1][0]  + "," + points[1][1]
              + "L" + points[2][0]  + "," + points[2][1]
              + "Z"
  return path;
}

function getCenter(rect) {
  var center = [rect[0]+.5*rect[2],
                rect[1]+.5*rect[3]];
  return center;
}

function setCentre(object,centre) {
  var width  = object.attrs.width;
  var height = object.attrs.height;
  var x = centre[0] - .5 * width;
  var y = centre[1] - .5 * height;
  object.attr({"x": x, "y": y});
}

<!-- Draw Methods -->

function drawPaper(rect) {
  var object = Raphael(rect[0],rect[1],rect[2],rect[3]);
  return object;
}

function drawRect(paper,rect) {
  var object = paper.rect(rect[0],rect[1],rect[2],rect[3]);
  return object;
}

function drawImage(paper,src,rect) {
  var object = paper.image(src,rect[0],rect[1],rect[2],rect[3]);
  return object;
}

function drawText(paper,center,text) {
  var object = paper.text(center[0],center[1],text);
  return object;
}

function drawPath(paper,path) {
  var object = paper.path(path);
  return object;
}

function drawEllipsoid(paper,rect) {
  var object = paper.ellipse(rect[0]+rect[2],rect[1]+rect[3],2*rect[2],2*rect[3]);
  return object;
}
