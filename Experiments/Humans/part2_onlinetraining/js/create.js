
function matrix(s,k) {
  if(typeof(s)=="undefined") { s=1;   }
  if(typeof(k)=="undefined") { k=NaN; }
  var s = [].concat(s);
  var t = [].concat(s);
  var n = t.splice(0,1);
  var y = [];
  for(var i=0; i<n; i++){
    if(t.length){
      y[i] = matrix(t,k);
    } else {
      y[i] = k;
    }
  }
  return y;
}

function clone(array) {
  if(array == null || typeof(array) != 'object') { return array; }
  var temp = array.constructor();
  for(var key in array) { temp[key] = clone(array[key]); }
  return temp;
};

function repmat(x,k) {
  x = [].concat(x);
  var y = x;
  var i;
  for(i=1; i<k; i++) {
    y = y.concat(x);
  }
  return y;
}

function zeros(s) { return matrix(s,0); }
function ones(s)  { return matrix(s,1); }

function linspace(x1,x2,n) {
  var v_x = [x1];
  for (var i_x = 1; i_x < n; i_x++) { v_x.push(x1 + i_x*(x2-x1)/(n-1)); }
  return v_x;
}

function colon(x1,x2,s_x) {
  if(s_x==undefined) { s_x = 1; }
  if(x2<x1)          { return []; }
  return linspace(x1, x2, 1+(x2-x1)/s_x);
}
