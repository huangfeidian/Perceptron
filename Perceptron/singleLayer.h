#include <vector>
#include "activateFunction.h"
#include <map>
#include <list>
#include <random>
#include <assert.h>
#include <iostream>
using std::cout;
using std::endl;
using std::vector;
using std::map;
using std::list;
#pragma once
class singleLayer
{
public:
	std::vector<double> inputValue;
	//current.inputvalue[j]=sum(connection.connectionWeight[i][j]*connection.isConnected[i][j]*pre.outputValue[i]*pre.is_maskerd[i])
	std::vector<double>  outputValue;
	//outputValue[i]=current.currentFunc(current.inputValue[i]+current.bias[i])
	std::vector<double>  isRemained;//if node i is dropouted then isDropouted[i] =1,else 0
	int remainNumber ;//the number of nodes to dropout
	std::vector<double>  outputGradient;//current.outputGradient[i]=sum(next.delta[j]*connection.connectionWeight[i][j]*connection.isConnected[i][j])
	std::vector<double>  delta;//delta[i]=outputGradient[i]*currentFunc.diff(outputValue[i])
	std::vector<double>  bias;// for the bias
	std::vector<double>  biasGradient;//biasGradient[i]=delta[i]
	std::vector<double>  batchBiasGradient;//batch sum of biasGradient[i]

	activateFunc currentFunc;//stands for the activate fucntion and the diffrentiation function
	const int dim ;//for the dimension

	singleLayer(int inDim, ACTIVATEFUNC currentFuncType) :dim(inDim), currentFunc(currentFuncType), remainNumber(inDim),
		inputValue(inDim, 0), isRemained(inDim, 1), outputValue(inDim, 0), delta(inDim, 0), outputGradient(inDim, 0), 
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
	//void updateInput(const vector<double>& partInput)//for now this function is not used
	//{
	//	for (int i = 0; i < dim; i++)
	//	{
	//		inputValue[i] += partInput[i];
	//	}
	//}
	virtual void forwardPropagate()
	{
		float scale = dim*1.0 / remainNumber;
		for (int i = 0; i < dim; i++)//we can use sse
		{
			outputValue[i] = scale*currentFunc(inputValue[i]+bias[i])*isRemained[i];
			inputValue[i] = 0;
		}
	}
	virtual void resetOutputGradient()
	{
		outputGradient = vector<double>(dim, 0);//clear the outputGradient
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
		
		}
	}
	virtual void updateBias(float biasStepsize)
	{
		for (int i = 0; i < dim; i++)
		{
			bias[i] -= batchBiasGradient[i]*biasStepsize;
			vector<double> temp(dim, 0);
			batchBiasGradient=temp;//clear the batch sum
		}
	}
	virtual void consoleValueOutput()
	{
		for (int i = 0; i < dim; i++)
		{
			cout << outputValue[i] << ' ';
		}
		cout << endl << "current layer output value" << endl;
	}
	virtual void consoleBiasOutput()
	{
		for (int i = 0; i < dim; i++)
		{
			cout << bias[i] << ' ';
		}
		cout << endl << "current layer  bias" << endl;
	}
};
