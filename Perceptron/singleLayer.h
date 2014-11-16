#include <vector>
#include "activateFunction.h"
#include <map>
#include <list>
#include <random>
#include <assert.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <iomanip>
#include "config.h"
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
	vector<vector<double>> inputValue;
	//current.inputvalue[j]=sum(connection.connectionWeight[i][j]*connection.isConnected[i][j]*pre.outputValue[i]*pre.is_maskerd[i])
	vector<vector<double>>  outputValue;
	//outputValue[i]=current.currentFunc(current.inputValue[i]+current.bias[i])
	
	vector<vector<double>>  outputGradient;//current.outputGradient[i]=sum(next.delta[j]*connection.connectionWeight[i][j]*connection.isConnected[i][j])
	vector<vector<double>> delta;//delta[i]=outputGradient[i]*currentFunc.diff(outputValue[i])
	vector<double> bias;// for the bias
	vector<double> biasGradient;//biasGradient[i]=delta[i]


	activateFunc currentFunc;//stands for the activate fucntion and the diffrentiation function
	const int dim ;//for the dimension

	singleLayer(int inDim, ACTIVATEFUNC currentFuncType) :dim(inDim), currentFunc(currentFuncType), 
		inputValue(BATCH_SIZE, vector<double>(inDim, 0)), outputValue(BATCH_SIZE, vector<double>(inDim, 0)), delta(BATCH_SIZE, vector<double>(inDim, 0)), 
		outputGradient(BATCH_SIZE, vector<double>(inDim, 0)),
		bias(inDim, 0), biasGradient(inDim, 0)
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
		for (int j = 0; j < BATCH_SIZE; j++)
		{
			for (int i = 0; i < dim; i++)//we can use sse
			{
				outputValue[j][i] = currentFunc(inputValue[j][i] + bias[i]);
				inputValue[j][i] = 0;
			}
		}
		
	}
	virtual void backPropagate()
	{
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < BATCH_SIZE; i++)
			{

				delta[j][i] = outputGradient[j][i] * currentFunc.diff(outputValue[j][i]);
				biasGradient[i]+= delta[j][i];
				outputGradient[j][i] = 0;

			}
		}
		
	}
	virtual void updateBias(double biasStepsize)
	{
		for (int i = 0; i < dim; i++)
		{
			bias[i] -= biasGradient[i]*biasStepsize;
			biasGradient[i]=0;//clear the batch sum
		}
	}
	virtual void consoleValueOutput()
	{
		for (int j = 0; j < BATCH_SIZE; j++)
		{
			for (int i = 0; i < dim; i++)
			{
				cout << outputValue[j][i] << ' ';
			}
			cout << endl << "current layer output value" << endl;
		}
		
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
	void fileResultOutput()
	{
		ofstream outputFile("singleCase.txt", std::ios::app);
		for (int j = 0; j < BATCH_SIZE; j++)
		{
			for (int i = 0; i < dim; i++)
			{
				outputFile << std::setw(13) << outputValue[j][i] << ' ';
			}
			outputFile << endl;
		}
		outputFile.close();
	}
};
