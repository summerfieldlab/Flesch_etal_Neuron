
function index(v,ii) {
  if(isempty(ii)) { return v; }
  // number
  if(typeof(ii)=="number") { return v[ii]; }
  // array
  var y = [];
  var j = 0;
  for(var i=0; i<ii.length; i++) {
    if(typeof(ii[i])=="number") {
      y[j] = v[ii[i]];
      j++;
    } else if (typeof(ii[i])=="boolean") {
      if(ii[i]) {
        y[j] = v[i];
        j++;
      }
    } else {
      error("index not recognised");
    }
  }
  return y;
}

function applyMatrix() {
  // variables
  var func = arguments[0];
  var mats = [arguments[1]];
  for(var i=1; i<arguments.length-1; i++) { mats[i] = arguments[i+1]; }
  // assert
  assert(typeof(func)=="function",'applyMatrix: error. not a function');
  var s = size(mats[0]);
  for(var i=1; i<mats.length; i++) { assert(isequal(s,size(mats[i])),'applyMatrix: error. size not consistent'); }
  // number
  var m = matrix();
  if (typeof(mats[0])=="number" || typeof(mats[0])=="boolean"){
    return func.apply(this,mats);
  }
  // vector
  for(var i=0; i<s[0]; i++){
    var subarguments = [func];
    for(var j=0; j<mats.length; j++){
      subarguments[j+1]= mats[j][i];
    }
    m[i] = applyMatrix.apply(this,subarguments);
  }
  return m;
}
