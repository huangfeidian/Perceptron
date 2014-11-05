#include <vector>
#include "activateFunction.h"
#include <map>
#include <list>
#include <random>
#include <assert.h>
using std::vector;
using std::map;
using std::list;
#pragma once
class singleLayer
{
public:
	std::vector<float> inputValue;
	//current.inputvalue[j]=sum(connection.connectionWeight[i][j]*connection.isConnected[i][j]*pre.outputValue[i]*pre.is_maskerd[i])
	std::vector<float>  outputValue;
	//outputValue[i]=current.currentFunc(current.inputValue[i]+current.bias[i])
	std::vector<float>  isRemained;//if node i is dropouted then isDropouted[i] =1,else 0
	int remainNumber ;//the number of nodes to dropout
	std::vector<float>  outputGradient;//current.outputGradient[i]=sum(next.delta[j]*connection.connectionWeight[i][j]*connection.isConnected[i][j])
	std::vector<float>  delta;//delta[i]=outputGradient[i]*currentFunc.diff(outputValue[i])
	std::vector<float>  bias;// for the bias
	std::vector<float>  biasGradient;//biasGradient[i]=delta[i]
	std::vector<float>  batchBiasGradient;//batch sum of biasGradient[i]

	activateFunc currentFunc;//stands for the activate fucntion and the diffrentiation function
	const int dim ;//for the dimension

	singleLayer(int inDim, ACTIVATEFUNC currentFuncType) :dim(inDim), currentFunc(currentFuncType), remainNumber(inDim),
		inputValue(inDim, 0), isRemained(inDim, 0), outputValue(inDim, 0), delta(inDim, 0), outputGradient(inDim, 0), 
		bias(inDim, 0), biasGradient(inDim, 0), batchBiasGradient(inDim, 0)
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
	//void updateInput(const vector<float>& partInput)//for now this function is not used
	//{
	//	for (int i = 0; i < dim; i++)
	//	{
	//		inputValue[i] += partInput[i];
	//	}
	//}
	virtual void forwardPropagate()
	{
		int scale = dim / remainNumber;
		for (int i = 0; i < dim; i++)//we can use sse
		{
			outputValue[i] = scale*currentFunc(inputValue[i]+bias[i])*isRemained[i];
			inputValue[i] = 0;
		}
	}
	virtual void resetOutputGradient()
	{
		outputGradient = vector<float>(dim, 0);//clear the outputGradient
	}
	virtual void backPropagate()
	{
		for (int i = 0; i < dim; i++)
		{
			delta[i] = outputGradient[i] * currentFunc.diff(outputValue[i]);
			biasGradient[i] = delta[i];
			batchBiasGradient[i] += biasGradient[i];
		}
	}
	virtual void updateBias(float biasStepsize)
	{
		for (int i = 0; i < dim; i++)
		{
			bias[i] -= batchBiasGradient[i]*biasStepsize;
			batchBiasGradient.swap(vector<float>(dim, 0));//clear the batch sum
		}
	}
};
