#include <vector>
#include "singleConnection.h"
#include <ppl.h>
#include <iostream>
using namespace concurrency;
using namespace std;
class convolution :public singleConnection
{
public:
	const int inDimRow;
	const int inDimColumn;
	const int outDimRow;
	const int outDimColumn;
	const int windowRow;
	const int windowColumn;
	vector<vector<float>> windowWeight;
	vector<vector<float>> windowWeightGradient;
	vector<vector<float>> batchWinWeiGradient;
	convolution(int inRow, int inColumn, int window) :singleConnection(inRow*inColumn, (inColumn - window + 1)*(inRow - window + 1)), windowRow(window)
		, windowColumn(window), inDimColumn(inColumn), inDimRow(inRow), outDimColumn(inColumn + 1 - window), outDimRow(inRow + 1 - window), windowWeight(window)
		, windowWeightGradient(window), batchWinWeiGradient(window)
		//watchout you must ensure inDimRow>=window and inDimColumn>=window
	{
		std::default_random_engine dre;
		std::uniform_real_distribution<float> di(-1.0, 1.0);
		vector<float> tempVec(window, 0);
		for (int i = 0; i < window; i++)
		{
			windowWeight.reserve(window);
			windowWeightGradient[i] = tempVec;
			batchWinWeiGradient[i] = tempVec;
			for (int j = 0; j < window; j++)
			{
				windowWeight[i][j] = di(dre);
			}
		}
		for (int i = 0; i <outDimRow; i++)
		{
			for (int j = 0; j <outDimColumn; j++)
			{
				for (int k = 0; k < windowRow; k++)
				{
					for (int l = 0; l < windowColumn; l++)
					{
						addConnection((i + k)*inDimColumn + j + l, i*outDimColumn + j, windowWeight[k][j]);
					}
				}
			}
		}
	}
	void forwardPropagate(const vector<float>& input, vector<float>& output)
	{
		for (int i = 0; i <outDimRow; i++)
		{
			for (int j = 0; j <outDimColumn; j++)
			{
				for (int k = 0; k < windowRow; k++)
				{
					for (int l = 0; l < windowColumn; l++)
					{
						output[i*outDimColumn + j] += input[(i + k)*inDimColumn + j + l] * windowWeight[k][l];
					}
				}
			}
		}
	}
	void backPropagate(const vector<float>& nextLayerDelta, vector<float>& preLayerGradient, const vector<float>& preLayerOutput)
	{
		for (int i = 0; i < windowRow; i++)
		{
			for (int j = 0; j < windowColumn; j++)
			{
				for (int k = 0; k < outDimRow; k++)
				{
					for (int l = 0; l < outDimColumn; l++)
					{
						windowWeightGradient[i][j] += nextLayerDelta[k*outDimColumn + l] * preLayerOutput[(k + i)*inDimColumn + j + l];
					}
				}
				batchWinWeiGradient[i][j] += windowWeightGradient[i][j];
				windowWeightGradient[i][j] = 0;
			}
		}
		for (int i = 0; i < inputDim; i++)
		{
			float propagateResult = 0;
			for (auto singleConnection : weightFromInput[i])
			{
				propagateResult += nextLayerDelta[singleConnection.first] * connectWeight[i][singleConnection.second];
			}
			preLayerGradient[i] = propagateResult;
		}
	}
	void updateWeight(float stepSize, const vector<float>& isRemained)
	{
		for (int i = 0; i < windowRow; i++)
		{
			for (int j = 0; j < windowColumn; j++)
			{
				windowWeight[i][j] -= stepSize*batchWinWeiGradient[i][j];
				batchWinWeiGradient[i][j] = 0;
			}
		}//update the window
		//then forward the update to the weight matrix
		for (int i = 0; i <outDimRow; i++)
		{
			for (int j = 0; j <outDimColumn; j++)
			{
				for (int k = 0; k < windowRow; k++)
				{
					for (int l = 0; l < windowColumn; l++)
					{
						connectWeight[(i + k)*inDimColumn + j + l][i*outDimColumn + j]= windowWeight[k][j];
					}
				}
			}
		}
	}
	void consoleWeightOutput()
	{
		for (int i = 0; i < windowRow; i++)
		{
			for (int j = 0; j < windowColumn; j++)
			{
				cout<< windowWeight[i][j] << ' ';
			}
			cout<< endl;
		}
		cout << "current convolution connection weight" << endl;
	}
};