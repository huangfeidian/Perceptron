#include <vector>
#include "activateFunction.h"
#include <map>
#include <list>
#include <random>
#include <assert.h>
#include <ppl.h>
using namespace concurrency;
using std::vector;
using std::map;
using std::list;
class layer
{
public:
	std::vector<float> inputValue;
	//current.inputvalue[j]=sum(connection.connectionWeight[i][j]*connection.isConnected[i][j]*pre.outputValue[i]*pre.is_maskerd[i])
	std::vector<float>  outputValue;
	//outputValue[i]=current.currentFunc(current.inputValue[i]+current.bias[i])
	std::vector<float>  isRemained;//if node i is dropouted then isDropouted[i] =1,else 0
	int remainNumber ;//the number of nodes to dropout
	std::vector<float>  weightGradient;//current.weightGradient[i]=sum(next.delta[j]*connection.connectionWeight[i][j]*connection.isConnected[i][j])
	std::vector<float>  delta;//delta[i]=weightGradient[i]*currentFunc.diff(outputValue[i])
	activateFunc currentFunc;//stands for the activate fucntion and the diffrentiation function
	const int dim ;
	layer(int inDim, ACTIVATEFUNC currentFuncType) :dim(inDim), currentFunc(currentFuncType), remainNumber(inDim),
		 inputValue(inDim, 0), isRemained(inDim, 0), outputValue(inDim, 0), delta(inDim, 0), weightGradient(inDim,0)
	{
		
	
		//and other initialtion
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
		for (int i = 0; i < dim-numberToRemain; i++)
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
	void update()
	{
		int scale = dim / remainNumber;
		for (int i = 0; i < dim; i++)//we can use sse
		{
			outputValue[i] = scale*currentFunc(inputValue[i])*isRemained[i];
		}
	}

};
