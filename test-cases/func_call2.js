function b(x) { return x + 2; }
function a(x) {
  let y = x + b(x);
  return y;
}

// Prepare both functions
%PrepareFunctionForOptimization(b);
%PrepareFunctionForOptimization(a);

// Warm-up loop to meet TurboFan hotness
for (let i = 0; i < 10000; i++) a(i);

// Force optimization
%OptimizeFunctionOnNextCall(b);
b(1);

%OptimizeFunctionOnNextCall(a);
a(1);

