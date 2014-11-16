#include "singleLayer.h"
class convolLayer :public singleLayer
{
public:
	const int dimCol;
	const int dimRow;
	double convolBias;
	double convolBiasGradient;
	convolLayer(int inDimRow, int inDimCol, ACTIVATEFUNC currentActiFun) :singleLayer(inDimCol*inDimRow, currentActiFun), dimCol(inDimCol), dimRow(inDimRow)
	{
		default_random_engine dre(clock());
		uniform_real_distribution<double> ran(-1.0, 1.0);
		convolBias = ran(dre);
		convolBiasGradient = 0;
	}
	void forwardPropagate()
	{
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < BATCH_SIZE; j++)
			{
				outputValue[j][i] = currentFunc(inputValue[j][i] + convolBias);
				inputValue[j][i] = 0;
			}
			
		}
	}
	void backPropagate()
	{
		convolBiasGradient = 0;
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < BATCH_SIZE; j++)
			{
				delta[j][i] = outputGradient[j][i] * currentFunc.diff(outputValue[j][i]);
				convolBiasGradient += delta[j][i];
				outputGradient[j][i] = 0;
			}
			
		}
	}
	void updateBias(double biasStepsize)
	{
		convolBias -= biasStepsize*convolBiasGradient;
		convolBiasGradient = 0;
	}
	void consoleBiasOutput()
	{
		cout << convolBias << endl;
	}
	void fileBiasOutput(ofstream& outFile)
	{
		outFile << convolBias << endl;
	}
	void loadBiasFromFile(ifstream& inputFile)
	{
		inputFile >> convolBias;
		inputFile.get();
	}
};