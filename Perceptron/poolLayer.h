#include "singleLayer.h"
using namespace std;
class poolLayer :public singleLayer
{
public:
	const int dimCol;
	const int dimRow;
	double poolBias;
	double poolBiasGradient;
	double batchPoolBiasGrad;
	poolLayer(int inDimRow, int inDimCol,ACTIVATEFUNC currentActiFun) :singleLayer(inDimCol*inDimRow, currentActiFun), dimCol(inDimCol), dimRow(inDimRow)
	{
		default_random_engine dre(clock());
		uniform_real_distribution<double> ran(-1.0, 1.0);
		poolBias = ran(dre);
		poolBiasGradient = 0;
		batchPoolBiasGrad = 0;
	}
	void forwardPropagate()
	{
		for (int i = 0; i < dim; i++)
		{
			outputValue[i] = currentFunc(inputValue[i]+poolBias);
			inputValue[i] = 0;
		}
	}
	void backPropagate()
	{
		poolBiasGradient = 0;
		for (int i = 0; i < dim; i++)
		{
			delta[i] = outputGradient[i]*currentFunc.diff(outputValue[i]);
			poolBiasGradient+= delta[i];
			outputGradient[i] = 0;
		}
		batchPoolBiasGrad += poolBiasGradient;
	}
	void updateBias(double biasStepsize)
	{
		poolBias -= biasStepsize*batchPoolBiasGrad;
		batchPoolBiasGrad = 0;
	}
	void consoleBiasOutput()
	{
		cout <<poolBias<< endl;
	}
	void fileBiasOutput(ofstream& outFile) 
	{
		outFile << poolBias<<endl;
	}
	void loadBiasFromFile(ifstream& inputFile)
	{
		inputFile >> poolBias;
		inputFile.get();
	}
};