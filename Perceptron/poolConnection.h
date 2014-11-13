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
	double scaleBatchGradient;
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
		for (int i = 0; i < inDimRow; i++)
		{
			for (int j = 0; j < inDimCol; j++)
			{
				addConnection(i*inDimRow + j, (i / poolWinSize)*outputDimCol+ j / poolWinSize, scale);
			}
		}
		scaleGradient = 0;
		scaleBatchGradient = 0;
	}
	void  forwardPropagate(const vector<double>& input, vector<double>& output)
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
						result += input[(i*poolWindowRow + k)*inputDimCol + (j*poolWindowCol + l)];
					}
				}
				output[i*outputDimCol + j] = result;
			}
		}

	}
	void backPropagate(const vector<double>& nextLayerDelta, vector<double>& preLayerGradient, const vector<double>& preLayerOutput)
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
						preLayerGradient[(i*poolWindowRow + k)*inputDimCol + (j*poolWindowCol + l)] += nextLayerDelta[i*outputDimCol + j] * scale;
						scaleGradient += preLayerOutput[(i*poolWindowRow + k)*inputDimCol + (j*poolWindowCol + l)]*nextLayerDelta[i*outputDimCol+j];
					}
				}
			}
		}
		scaleBatchGradient += scaleGradient;
	}
	void updateWeight(double stepSize)
	{
		scaleBatchGradient = scaleBatchGradient / inputDim;
		scale -= stepSize*scaleBatchGradient;
		scaleBatchGradient = 0;
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
		inputFile.get();
	}
};