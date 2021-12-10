
function isempty(v) {
  if(typeof(v)=="number") {
    return 0;
  }
  if(typeof(v)=="object" || typeof(v)=="string") {
    if(v.length) { return 0; }
    else         { return 1; }
  }
}

function size(v,d) {
  var get_size = function(v) { var s = [1]; if(typeof(v)=="object") { s = [v.length].concat(get_size(v[0])); } return s; };
  var s = get_size(v);
  if(s.length>1) { s.pop(); }
  if(typeof(d)=="undefined") { return s; }
  return s[d];
}

function numel(v) {
  return prod(size(v));
}

function length(v) {
  return max([size(v,0),size(v,1)]);
}
