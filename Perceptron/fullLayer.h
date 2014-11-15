#include "singleLayer.h"
class fullLayer :public singleLayer
{
public:
	std::vector<double>  isRemained;//if node i is dropouted then isDropouted[i] =1,else 0
	int remainNumber;//the number of nodes to dropout
	fullLayer(int inDim, ACTIVATEFUNC currentFuncType) :singleLayer(inDim, currentFuncType), isRemained(inDim, 1), remainNumber(inDim)
	{

	}
	void dropoutNodes(int numberToRemain)
	{
		vector<int> forShuffle(dim);
		remainNumber = numberToRemain;
		for (int i = 0; i < dim; i++)
		{
			forShuffle[i] = i;
		}
		std::default_random_engine dre;
		std::shuffle(forShuffle.begin(), forShuffle.end(), dre);
		for (int i = 0; i < dim - numberToRemain; i++)
		{
			isRemained[forShuffle[i]] = 0.0;
		}
	}
	void dropoutRestore()
	{
		for (int i = 0; i < dim; i++)
		{
			isRemained[i] = 1.0;
		}
		remainNumber = dim;
	}
	virtual void forwardPropagate()
	{
		double scale = dim*1.0 / remainNumber;
		for (int i = 0; i < dim; i++)//we can use sse
		{
			outputValue[i] = scale*currentFunc(inputValue[i] + bias[i])*isRemained[i];
			inputValue[i] = 0;
		}
	}
	virtual void backPropagate()
	{
		for (int i = 0; i < dim; i++)
		{
			if (isRemained[i] == 1.0)
			{
				delta[i] = outputGradient[i] * currentFunc.diff(outputValue[i]);
				biasGradient[i] = delta[i];
				batchBiasGradient[i] += biasGradient[i];
			}
			outputGradient[i] = 0;

		}
	}
	void resetOutputAndGradient()
	{

	}
};