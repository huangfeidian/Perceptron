#include "singleConnection.h"
#include <iostream>
#include "accelerateFor.h"
#include "config.h"
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
	double scale;
	double scaleGradient;
	poolConnection(int inDimRow, int inDimCol, int poolWinSize) :singleConnection(inDimCol*inDimRow, (inDimCol / poolWinSize)*(inDimRow / poolWinSize)), poolWindowCol(poolWinSize),
		poolWindowRow(poolWinSize)
		, inputDimCol(inDimCol), inputDimRow(inDimRow), outputDimCol(inDimCol / poolWinSize), outputDimRow(inDimRow / poolWinSize)
	{
#ifdef CHECK_LEGITIMATE
		assert(inDimCol%poolWinSize == 0);
		assert(inDimRow%poolWinSize == 0);
#endif
		default_random_engine dre(clock());
		uniform_real_distribution<double> ran;
		scale = ran(dre);
		//for (int i = 0; i < inDimRow; i++)
		//{
		//	for (int j = 0; j < inDimCol; j++)
		//	{
		//		addConnection(i*inDimRow + j, (i / poolWinSize)*outputDimCol+ j / poolWinSize, scale);
		//	}
		//}
		scaleGradient = 0;
	}
	virtual void addConnection(int fromIndex, int toIndex, double weight)
	{
		// just do nothing
		//totalConnections++;
		//connectWeight[fromIndex][toIndex] = weight;
		//isConnected[fromIndex][toIndex] = 1;
		//weightFromInput[fromIndex].push_back(toIndex);
		//weightToOutput[toIndex].push_back(fromIndex);
	}
	void  forwardPropagate(const vector<vector<double>>& input, vector<vector<double>>& output)
	{
		for (int t = 0; t < BATCH_SIZE; t++)
		{
			for (int i = 0; i < outputDimRow; i++)
			{
				for (int j = 0; j < outputDimCol; j++)
				{
					double result = 0;
					for (int k = 0; k < poolWindowRow; k++)
					{
						for (int l = 0; l < poolWindowCol; l++)
						{
							result += input[t][(i*poolWindowRow + k)*inputDimCol + (j*poolWindowCol + l)];
						}
					}
					output[t][i*outputDimCol + j] = result*scale;
				}
			}
		}
		

	}
	void backPropagate(const vector<vector<double>>& nextLayerDelta, vector<vector<double>>& preLayerGradient, const vector<vector<double>>& preLayerOutput)
		//in this implementation the prelayeroutput is not used ,we just dont need it 
	{
		scaleGradient = 0;
		for (int i = 0; i < outputDimRow; i++)
		{
			for (int j = 0; j < outputDimCol; j++)
			{

				for (int k = 0; k < poolWindowRow; k++)
				{
					for (int l = 0; l < poolWindowCol; l++)
					{
						for (int t = 0; t < BATCH_SIZE; t++)
						{
							preLayerGradient[t][(i*poolWindowRow + k)*inputDimCol + (j*poolWindowCol + l)] += nextLayerDelta[t][i*outputDimCol + j] * scale;
							scaleGradient += preLayerOutput[t][(i*poolWindowRow + k)*inputDimCol + (j*poolWindowCol + l)] * nextLayerDelta[t][i*outputDimCol + j];
						}
					}
				}
			}
		}

	}
	void updateWeight(double stepSize)
	{
		scale -= stepSize*scaleGradient;
		scaleGradient = 0;
	}
	void consoleWeightOutput()
	{
		cout <<scale<< endl;
	}
	void fileWeightOutput(ofstream& outFile)
	{
		outFile << scale << endl;
	}
	void loadWeightFromFile(ifstream& inputFile)
	{
		inputFile >> scale;
		scale = scale / 4;
		inputFile.get();
	}
};