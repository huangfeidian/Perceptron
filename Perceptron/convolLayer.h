#include "singleLayer.h"
class convolLayer :public singleLayer
{
public:
	const int dimCol;
	const int dimRow;
	double convolBias;
	double convolBiasGradient;
	double batchConvolBiasGrad;
	convolLayer(int inDimRow, int inDimCol, ACTIVATEFUNC currentActiFun) :singleLayer(inDimCol*inDimRow, currentActiFun), dimCol(inDimCol), dimRow(inDimRow)
	{
		default_random_engine dre(clock());
		uniform_real_distribution<double> ran(-1.0, 1.0);
		convolBias = ran(dre);
		convolBiasGradient = 0;
		batchConvolBiasGrad = 0;
	}
	void forwardPropagate()
	{
		for (int i = 0; i < dim; i++)
		{
			outputValue[i] = currentFunc(inputValue[i] + convolBias);
			inputValue[i] = 0;
		}
	}
	void backPropagate()
	{
		convolBiasGradient = 0;
		for (int i = 0; i < dim; i++)
		{
			delta[i] = outputGradient[i] * currentFunc.diff(outputValue[i]);
			convolBiasGradient += delta[i];
			outputGradient[i] = 0;
		}
		batchConvolBiasGrad += convolBiasGradient;
	}
	void updateBias(double biasStepsize)
	{
		batchConvolBiasGrad = batchConvolBiasGrad / dim;
		convolBias -= biasStepsize*batchConvolBiasGrad;
		batchConvolBiasGrad = 0;
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