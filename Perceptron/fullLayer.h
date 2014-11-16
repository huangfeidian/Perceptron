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
		for (int i = 0; i < BATCH_SIZE; i++)//we can use sse
		{
			for (int j = 0; j < dim; j++)
			{
				outputValue[i][j] = scale*currentFunc(inputValue[i][j] + bias[j])*isRemained[j];
				inputValue[i][j] = 0;
			}

		}
	}
	virtual void backPropagate()
	{
		for (int i = 0; i < dim; i++)
		{
			if (isRemained[i] == 1.0)
			{
				for (int j = 0; j < BATCH_SIZE; j++)
				{
					delta[j][i] = outputGradient[j][i] * currentFunc.diff(outputValue[j][i]);
					biasGradient[i] += delta[j][i];
				}
				
			}
			for (int j = 0; j < BATCH_SIZE; j++)
			{
				outputGradient[j][i] = 0;
			}

		}
	}

};