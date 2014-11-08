#include "singleConnection.h"
#include <iostream>
#include "accelerateFor.h"
using namespace std;
class poolConnection :public singleConnection
{
public:
	const int poolWindowCol;
	const int poolWindowRow;
	const int inputDimRow;
	const int inputDimCol;
	const int outputDimRow;
	const int outputDimCol;
	vector<vector<int>> preLayerWinMaxIndex;
	poolConnection(int inDimRow, int inDimCol, int poolWinSize) :singleConnection(inDimCol*inDimRow, (inDimCol / poolWinSize)*(inDimRow / poolWinSize)), poolWindowCol(poolWinSize),
		poolWindowRow(poolWinSize)
		, inputDimCol(inDimCol), inputDimRow(inDimRow), outputDimCol(inDimCol / poolWinSize), outputDimRow(inDimRow / poolWinSize)
		, preLayerWinMaxIndex(inDimRow / poolWinSize, vector<int>(inDimCol / poolWinSize,0))
	{
		assert(inDimCol%poolWinSize == 0);
		assert(inDimRow%poolWinSize == 0);
		for (int i = 0; i < inDimRow; i++)
		{
			for (int j = 0; j < inDimCol; j++)
			{
				addConnection(i*inDimRow + j, (i / poolWinSize)*outputDimCol+ j / poolWinSize, 0);
			}
		}
	}
	void  forwardPropagate(const vector<double>& input, vector<double>& output)
	{
		for (int i = 0; i < outputDimRow; i++)
		{
			for (int j = 0; j < outputDimCol; j++)
			{
				double tempmax = input[i*poolWindowRow*inputDimCol + j*poolWindowCol];
				preLayerWinMaxIndex[i][j] = i*poolWindowRow*inputDimCol + j*poolWindowCol;
				for (int k = 0; k < poolWindowRow; k++)
				{
					for (int l = 0; l < poolWindowCol; l++)
					{
						double tempValue = input[(i*poolWindowRow + k)*inputDimCol + j*poolWindowCol + l];
						if (tempValue > tempmax)
						{
							tempmax = tempValue;
							preLayerWinMaxIndex[i][j] = (i*poolWindowRow + k)*inputDimCol + j*poolWindowCol + l;
						}
					}
				}
				output[i*outputDimCol + j] = tempmax;
			}
		}
	}
	void backPropagate(const vector<double>& nextLayerDelta, vector<double>& preLayerGradient, const vector<double>& preLayerOutput)
		//in this implementation the prelayeroutput is not used ,we just dont need it 
	{
		for(int i=0; i<inputDimRow;i++)
		{
			for (int j = 0; j < inputDimCol; j++)
			{
				preLayerGradient[i*inputDimCol + j] = 0;
			}
		}
		for(int i=0; i<outputDimRow; i++)
		{
			for (int j = 0; j < outputDimCol; j++)
			{
				preLayerGradient[preLayerWinMaxIndex[i][j]] = nextLayerDelta[i*outputDimCol + j];
			}
		}
		//we dont record the weight ,because weight is useless in pool connection
	}
	void updateWeight(double stepSize)
	{
		//do nothing because  there are no weights here
	}
	void consoleWeightOutput()
	{
		
		cout << "current connection is pool" << endl;
	}
};