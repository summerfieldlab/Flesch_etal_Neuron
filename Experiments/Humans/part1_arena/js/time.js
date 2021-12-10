
<!-- Time methods  -->
function getTimestamp() {
  var ts = (new Date()).getTime();
  return ts;
}

function getSecs(ts){
  var tn = (new Date()).getTime();
  var td = 0.001*(tn-ts);
  return td;
}
