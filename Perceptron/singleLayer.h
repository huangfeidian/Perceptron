#include <vector>
#include "activateFunction.h"
#include <map>
#include <list>
#include <random>
#include <assert.h>
#include <iostream>
#include <ctime>
#include <fstream>
using std::cout;
using std::endl;
using std::vector;
using std::map;
using std::list;
using std::ofstream;
using std::ifstream;
#pragma once
class singleLayer
{
public:
	std::vector<double> inputValue;
	//current.inputvalue[j]=sum(connection.connectionWeight[i][j]*connection.isConnected[i][j]*pre.outputValue[i]*pre.is_maskerd[i])
	std::vector<double>  outputValue;
	//outputValue[i]=current.currentFunc(current.inputValue[i]+current.bias[i])
	
	std::vector<double>  outputGradient;//current.outputGradient[i]=sum(next.delta[j]*connection.connectionWeight[i][j]*connection.isConnected[i][j])
	std::vector<double>  delta;//delta[i]=outputGradient[i]*currentFunc.diff(outputValue[i])
	std::vector<double>  bias;// for the bias
	std::vector<double>  biasGradient;//biasGradient[i]=delta[i]
	std::vector<double>  batchBiasGradient;//batch sum of biasGradient[i]

	activateFunc currentFunc;//stands for the activate fucntion and the diffrentiation function
	const int dim ;//for the dimension

	singleLayer(int inDim, ACTIVATEFUNC currentFuncType) :dim(inDim), currentFunc(currentFuncType), 
		inputValue(inDim, 0), outputValue(inDim, 0), delta(inDim, 0), outputGradient(inDim, 0), 
		bias(inDim, 0), biasGradient(inDim, 0), batchBiasGradient(inDim, 0)
	{
		std::default_random_engine dre(clock());
		std::uniform_real_distribution<double> di(-1.0, 1.0);
		for (int i = 0; i < inDim; i++)
		{
			bias[i] = di(dre);
		}
	
		//and other initialtion
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
		for (int i = 0; i < dim; i++)//we can use sse
		{
			outputValue[i] = currentFunc(inputValue[i]+bias[i]);
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

			delta[i] = outputGradient[i] * currentFunc.diff(outputValue[i]);
			biasGradient[i] = delta[i];
			batchBiasGradient[i] += biasGradient[i];
			outputGradient[i] = 0;
		
		}
	}
	virtual void updateBias(double biasStepsize)
	{
		for (int i = 0; i < dim; i++)
		{
			bias[i] -= batchBiasGradient[i]*biasStepsize;
			batchBiasGradient[i]=0;//clear the batch sum
			
			delta[i] = 0;
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
	virtual void fileBiasOutput(ofstream& outFile)
	{
		for (int i = 0; i < dim; i++)
		{
			outFile << bias[i] << ' ';
		}
		outFile << endl;
	}
	virtual void loadBiasFromFile(ifstream& inputFile)
	{
		char temp[100];
		for (int i = 0; i < dim; i++)
		{
			inputFile >> bias[i];
		}
		inputFile.getline(temp,99);
	}
	virtual void dropoutNodes(int nodesToRemain)
	{
		// do nothing
	}
	virtual void dropoutRestore()
	{
		//do nothing
	}
};
