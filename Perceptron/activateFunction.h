#include <algorithm>
#pragma once
enum class ACTIVATEFUNC
{
	TANH,
	SIGMOID,
	IDENTITY,
	RECTLINEAR
};
typedef float(*functype)(float);
float evalTanh(float input)
{
	float temp = std::exp(2*input);
	float result = 1 - 2 / (temp + 1);
	return result;
}
float diffTanh(float input)
{
	return 1 - input*input;
}
float evalSigmoid(float input)
{
	float temp = std::exp(input);
	float result = 1 / (1 + temp);
	return result;
}
float diffSigmoid(float input)
{
	return (1 - input)*input;
}
float evalIdentity(float input)
{
	return input;
}
float diffIdentity(float input)
{
	return 1;
}
float evalRectifiedLinear(float input)
{
	return input > 0.0 ? input : 0.0;
}
float diffRectifiedLinear(float input)
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
	float operator()(float input)const
	{
		return (*currentEvalFunc)(input);
	}
	float eval(float input)const
	{
		return (*currentEvalFunc)(input);
	}
	float diff(float input)const
	{
		return (*currentDiffFunc)(input);
	}
};
