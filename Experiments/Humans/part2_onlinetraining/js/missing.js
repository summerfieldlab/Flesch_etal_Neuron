
function assert(condition, message) { if (!condition) { throw message || "Assertion failed"; } }
function error(message)             { throw message || "Error"; }
function disp(input)                { console.log(input); }
function fprintf()                  { console.log(sprintf.apply(this,arguments)); }

Math.gamma = function(n) {
  var g = 7;
  var p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if(n < 0.5) { return Math.PI / Math.sin(n * Math.PI) / Math.gamma(1 - n); }
  else {
    n--;
    var x = p[0];
    for(var i = 1; i < g + 2; i++) { x += p[i] / (n + i); }
    var t = n + g + 0.5;
    return Math.sqrt(2 * Math.PI) * Math.pow(t, (n + 0.5)) * Math.exp(-t) * x;
  }
}

Math.factorial = function(n) {
  return Math.gamma(n + 1);
}

Math.sign  = function(x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }
