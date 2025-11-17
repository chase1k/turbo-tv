function b(x) { return x+2;}
function a(x) {
	y = x+b(x);
	return y;
}


%PrepareFunctionForOptimization(b);
%PrepareFunctionForOptimization(a);
a(1);
%OptimizeFunctionOnNextCall(b);
b(1);
%OptimizeFunctionOnNextCall(a);
a(1);

