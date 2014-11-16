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
	convolutionConnection(int inRow, int inColumn, int window) :singleConnection(inRow*inColumn, (inColumn - window + 1)*(inRow - window + 1)), windowRow(window)
		, windowColumn(window), inDimColumn(inColumn), inDimRow(inRow), outDimColumn(inColumn + 1 - window), outDimRow(inRow + 1 - window), windowWeight(window, vector<double>(window, 0))
		, windowWeightGradient(window, vector<double>(window, 0))
		//watchout you must ensure inDimRow>=window and inDimColumn>=window
	{
		std::default_random_engine dre(clock());
		std::uniform_real_distribution<double> di(-1.0, 1.0);
		vector<double> tempVec(window, 0);
		for (int i = 0; i < window; i++)
		{
			windowWeightGradient[i] = tempVec;
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
	void forwardPropagate(const vector<vector<double>>& input, vector<vector<double>>& output)
	{
		accelerateFor (0, BATCH_SIZE,[&](int t)
		{
			for (int i = 0; i < outDimRow; i++)
			{
				for (int j = 0; j <outDimColumn; j++)
				{
					for (int k = 0; k < windowRow; k++)
					{
						for (int l = 0; l < windowColumn; l++)
						{
							output[t][i*outDimColumn + j] += input[t][(i + k)*inDimColumn + j + l] * windowWeight[k][l];
						}
					}
				}
			}
		});
		
	}
	void backPropagate(const vector<vector<double>>& nextLayerDelta, vector<vector<double>>& preLayerGradient, const vector<vector<double>>& preLayerOutput)
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
						for (int t = 0; t < BATCH_SIZE; t++)
						{
							windowWeightGradient[i][j] += nextLayerDelta[t][k*outDimColumn + l] * preLayerOutput[t][(k + i)*inDimColumn + j + l];

						}
					}
				}
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
				double tempWeight = windowWeight[preLayerRowIndex - nextLayerRowIndex][preLayerColIndex - nextLayerColIndex];
				for (int j = 0; j < BATCH_SIZE; j++)
				{
					preLayerGradient[j][i] += nextLayerDelta[j][singleConnection] *tempWeight ;
				
				}
			}
			
		});
	}
	void updateWeight(double stepSize)
	{
		for (int i = 0; i < windowRow; i++)
		{
			for (int j = 0; j < windowColumn; j++)
			{
				windowWeight[i][j] -= stepSize*windowWeightGradient[i][j];
				windowWeightGradient[i][j] = 0;
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