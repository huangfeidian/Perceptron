#include <vector>
#include "singleConnection.h"
#include <ppl.h>
#include <iostream>
using namespace concurrency;
using namespace std;
class convolutionConnection :public singleConnection
{
public:
	const int inDimRow;
	const int inDimColumn;
	const int outDimRow;
	const int outDimColumn;
	const int windowRow;
	const int windowColumn;
	vector<vector<double>> windowWeight;
	vector<vector<double>> windowWeightGradient;
	vector<vector<double>> batchWinWeiGradient;
	convolutionConnection(int inRow, int inColumn, int window) :singleConnection(inRow*inColumn, (inColumn - window + 1)*(inRow - window + 1)), windowRow(window)
		, windowColumn(window), inDimColumn(inColumn), inDimRow(inRow), outDimColumn(inColumn + 1 - window), outDimRow(inRow + 1 - window), windowWeight(window, vector<double>(window, 0))
		, windowWeightGradient(window, vector<double>(window, 0)), batchWinWeiGradient(window, vector<double>(window, 0))
		//watchout you must ensure inDimRow>=window and inDimColumn>=window
	{
		std::default_random_engine dre(clock());
		std::uniform_real_distribution<double> di(-1.0, 1.0);
		vector<double> tempVec(window, 0);
		for (int i = 0; i < window; i++)
		{
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
						addConnection((i + k)*inDimColumn + j + l, i*outDimColumn + j, windowWeight[k][l]);
	
					}
				}
			}
		}
	}
	void addConnection(int fromIndex, int toIndex,double currentWeight)
	{
		totalConnections++;
		weightFromInput[fromIndex].push_back(toIndex);
		weightToOutput[toIndex].push_back(fromIndex);
	}
	void forwardPropagate(const vector<double>& input, vector<double>& output)
	{
		accelerateFor( 0, outDimRow,[&](int i)
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
		});
	}
	void backPropagate(const vector<double>& nextLayerDelta, vector<double>& preLayerGradient, const vector<double>& preLayerOutput)
	{
		accelerateFor ( 0,  windowRow,[&](int i)
		{
			for (int j = 0; j < windowColumn; j++)
			{
				windowWeightGradient[i][j] = 0;
				for (int k = 0; k < outDimRow; k++)
				{
					for (int l = 0; l < outDimColumn; l++)
					{
						windowWeightGradient[i][j] += nextLayerDelta[k*outDimColumn + l] * preLayerOutput[(k + i)*inDimColumn + j + l];
					}
				}
				batchWinWeiGradient[i][j] += windowWeightGradient[i][j];
				
			}
		});
		
		accelerateFor(  0,  inputDim,[&](int i)
		{
			int preLayerRowIndex;
			int preLayerColIndex;
			double propagateResult = 0;
			preLayerColIndex = i%inDimColumn;
			preLayerRowIndex = i / inDimColumn;
			int nextLayerRowIndex;
			int nextLayerColIndex;
			for (auto singleConnection : weightFromInput[i])
			{
				nextLayerColIndex = singleConnection%outDimColumn;
				nextLayerRowIndex = singleConnection / outDimColumn;
				propagateResult += nextLayerDelta[singleConnection] * windowWeight[preLayerRowIndex-nextLayerRowIndex][preLayerColIndex-nextLayerColIndex];
			}
			preLayerGradient[i]+= propagateResult;
		});
	}
	void updateWeight(double stepSize)
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
		//for (int i = 0; i <outDimRow; i++)
		//{
		//	for (int j = 0; j <outDimColumn; j++)
		//	{
		//		for (int k = 0; k < windowRow; k++)
		//		{
		//			for (int l = 0; l < windowColumn; l++)
		//			{
		//				connectWeight[(i + k)*inDimColumn + j + l][i*outDimColumn + j]= windowWeight[k][l];
		//			}
		//		}
		//	}
		//}
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
	void fileWeightOutput(ofstream& outFile)
	{
		for (int i = 0; i < windowRow; i++)
		{
			for (int j = 0; j < windowColumn; j++)
			{
				outFile << windowWeight[i][j] << ' ';
			}
			outFile<< endl;
		}
	}
	void loadWeightFromFile(ifstream& inputFile)
	{
		char temp;
		for (int i = 0; i < windowRow; i++)
		{
			for (int j = 0; j < windowColumn; j++)
			{
				inputFile >> windowWeight[i][j];
			}
			inputFile.get(temp);
			inputFile.get(temp);//get the endl and jetison
		}
	}
};