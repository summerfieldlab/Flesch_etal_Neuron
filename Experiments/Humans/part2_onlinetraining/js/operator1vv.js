
function floor(v)     { return applyMatrix(Math.floor,v);                }
function ceil(v)      { return applyMatrix(Math.ceil,v);                 }
function round(v)     { return applyMatrix(Math.round,v);                }
function abs(v)       { return applyMatrix(Math.abs,v);                  }
function cos(v)       { return applyMatrix(Math.cos,v);                  }
function acos(v)      { return applyMatrix(Math.acos,v);                 }
function sin(v)       { return applyMatrix(Math.sin,v);                  }
function asin(v)      { return applyMatrix(Math.asin,v);                 }
function tan(v)       { return applyMatrix(Math.tan,v);                  }
function atan(v)      { return applyMatrix(Math.atan,v);                 }
function atan2(v)     { return applyMatrix(Math.atan2,v);                }
function exp(v)       { return applyMatrix(Math.exp,v);                  }
function log(v)       { return applyMatrix(Math.log,v);                  }
function isnan(v)     { return applyMatrix(isNaN,v);                     }
function mod(v,m)     { return applyMatrix(function(v){return v%m;},v);  }
function sign(v)      { return applyMatrix(Math.sign,v);                 }
function gamma(v)     { return applyMatrix(Math.gamma,v);                }
function factorial(v) { return applyMatrix(Math.factorial,v);            }
function sqrt(v)      { return applyMatrix(Math.sqrt,v);                 }
function pow2(v)      { return power(2,v);                               }
function vneg(v)      { return applyMatrix(function(v){return -v;},v);   }
function vinv(v)      { return applyMatrix(function(v){return 1/v;},v);  }

function diff(v) {
  assert(size(v).length==1,'sort: error. not a vector');
  var y = matrix(v.length-1);
  for(var i=0;i<y.length;i++) {y[i]=v[i+1]-v[i];}
  return y;
}

function sort(v) {
  assert(size(v).length==1,'sort: error. not a vector');
  var y = clone(v);
  var sortNumber = function(a,b){ return a-b; };
  return y.sort(sortNumber);
}

function normalize(v) {
  assert(size(v).length==1,'sort: error. not a vector');
  var y = clone(v);
  var m = mean(y);
  var s = std(y);
  y = applyMatrix(function(x){return (x-m)/s;},y);
  return y;
}

function shuffle(v,r) {
  if(typeof(r)=="undefined") { r = randperm(v.length); }
  return index(clone(v),r);
}

function vPrecision(v,n) { return applyMatrix(function(v) {return toPrecision(v,n);}, v); }
