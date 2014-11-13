#pragma once
#include <algorithm>

enum class ACTIVATEFUNC
{
	TANH,
	SIGMOID,
	IDENTITY,
	RECTLINEAR
};
typedef double(*functype)(double);
double evalTanh(double input)
{
	double temp = std::exp(2*input);
	double result = 1 - 2 / (temp + 1);
	return result;
}
double diffTanh(double input)
{
	return 1 - input*input;
}
double evalSigmoid(double input)
{
	double temp = std::exp(-1*input);
	double result = 1 / (1 + temp);
	return result;
}
double diffSigmoid(double input)
{
	return (1 - input)*input;
}
double evalIdentity(double input)
{
	return input;
}
double diffIdentity(double input)
{
	return 1;
}
double evalRectifiedLinear(double input)
{
	return input > 0.0 ? input : 0.0;
}
double diffRectifiedLinear(double input)
{
	return input > 0.0 ? 1: 0.0;
}
functype evalFunc[4] = { &evalTanh, &evalSigmoid ,&evalIdentity,&evalRectifiedLinear};
functype diffFunc[4] = { &diffTanh, &diffSigmoid, &diffIdentity, &diffRectifiedLinear };
 class activateFunc
{
private:
	functype currentEvalFunc;
	functype currentDiffFunc;
public:
	activateFunc(ACTIVATEFUNC currentFuncType):currentDiffFunc(diffFunc[(int)currentFuncType]),currentEvalFunc(evalFunc[(int)currentFuncType])
	{

	}
	double operator()(double input)const
	{
		return (*currentEvalFunc)(input);
	}
	double eval(double input)const
	{
		return (*currentEvalFunc)(input);
	}
	double diff(double input)const
	{
		return (*currentDiffFunc)(input);
	}
};
