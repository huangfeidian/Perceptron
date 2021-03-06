#include "singleLayer.h"
#include "vectorize.h"
#include "accelerateFor.h"
#include <algorithm>
#include <set>
#include <ctime>
#pragma once
using namespace std;
class singleConnection
{
public:
	std::vector<std::vector<double>> connectWeight;//the weights of connections between layers,currently i don't care the sparse before this demo works
	const int inputDim;
	const int outputDim;
	std::vector<std::vector<bool>> isConnected;//if node i in prev layer and node j in next layer is connected then isConnected[i][j]=1,else isConnected[i][j]=0
	std::vector<std::vector<double>> weightGradient;//for the Gradient of the weight
	std::vector<std::vector<int>> weightFromInput;//weightFromInput[i][j]=connectWeight[i][j]
	std::vector<std::vector<int>> weightToOutput;//weightToOutput[i][j]=connecWeight[j][i]

	int totalConnections;//sum of  all isConnected[i][j]!=0
	

	singleConnection(int inDim, int outDim) :inputDim(inDim), outputDim(outDim), totalConnections(0)
		, weightFromInput(inDim, vector<int>()),
		weightToOutput(outDim, vector< int>())
	{
		
	}
	virtual void setConnected(const vector<vector<bool>>& inputIsConnected)
	{
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				isConnected[i][j] = inputIsConnected[i][j] ;
			}
		}
	}
	virtual void initWeight()
	{
		std::default_random_engine dre(clock());
		std::uniform_real_distribution<double> di(-1.0, 1.0);
		double tempWeight;
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				if (isConnected[i][j] == 1)
				{
					tempWeight = di(dre);
					addConnection(i, j, tempWeight);
				}
			}
		}
	}
	virtual void initResource()
	{
		connectWeight = vector<vector<double>>(inputDim, vector<double>(outputDim, 0));
		isConnected = vector<vector<bool>>(inputDim, vector<bool>(outputDim, false));
		weightGradient = vector<vector<double>>(inputDim, vector<double>(outputDim, 0));
	}
	virtual void addConnection(int fromIndex, int toIndex, double weight)
	{
		totalConnections++;
		connectWeight[fromIndex][toIndex] = weight;
		isConnected[fromIndex][toIndex] = 1;
		weightFromInput[fromIndex].push_back(toIndex);
		weightToOutput[toIndex].push_back(fromIndex);
	}
	virtual void forwardPropagate(const vector<vector<double>>& input, vector<vector<double>>& output)
	{
		for (int i= 0; i<outputDim;i++)//we can use multithread or multithread
		{
			for (int j = 0; j < BATCH_SIZE; j++)
			{
				double propagateResult = 0;
				for (auto singleConnection : weightToOutput[i])//we can use sse or avx
				{
					propagateResult += input[j][singleConnection] * connectWeight[singleConnection][i];
				}
				output[j][i] += propagateResult;
			}
			
		}
	}
	virtual void backPropagate(const vector<vector<double>>& nextLayerDelta, vector<vector<double>>& preLayerGradient, const vector<vector<double>>& preLayerOutput)
	{
		//we can gain more parallel
		for (int i = 0; i < inputDim; i++)//for the layer nodes
		{
			double propagateResult[BATCH_SIZE] = { 0 };
			for (auto singleConnection : weightFromInput[i])
			{
				for (int j = 0; j < BATCH_SIZE; j++)
				{
					propagateResult[j] = nextLayerDelta[j][singleConnection] * connectWeight[i][singleConnection];
					preLayerGradient[j][i] += propagateResult[j];
				}
			}
			
		}
		
		//begin update the weight
		for(int i=0; i< outputDim;i++)
		{
			for (auto singleConnection : weightToOutput[i])
			{
				double temp = 0;
				for (int j = 0; j < BATCH_SIZE; j++)
				{
					temp += nextLayerDelta[j][i] * preLayerOutput[j][singleConnection];

				}
				weightGradient[singleConnection][i] =temp ;
			}
		}
	}
	
	virtual void updateWeight(double stepSize)
	{
		for ( int i=0;i<inputDim;i++)
		{
			for(auto in:weightFromInput[i])
			{
				connectWeight[i][in] -=stepSize*weightGradient[i][in];
				weightGradient[i][in] = 0;//clear the batch sum
			};
		}
	}
	virtual void consoleWeightOutput()
	{
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				cout << connectWeight[i][j] << ' ';
			}
			cout << endl;
		}
		cout << "current connection weight" << endl;
	}
	virtual void fileWeightOutput(ofstream& outFile)
	{
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				outFile << connectWeight[i][j] << ' ';
			}
			outFile << endl;
		}
		outFile<< "**************************" << endl;
	}
	virtual void loadWeightFromFile(ifstream& inputFile)
	{
		//do nothing
	}

};
