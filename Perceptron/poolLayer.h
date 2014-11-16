#include "singleLayer.h"
using namespace std;
class poolLayer :public singleLayer
{
public:
	const int dimCol;
	const int dimRow;
	double poolBias;
	double poolBiasGradient;
	poolLayer(int inDimRow, int inDimCol,ACTIVATEFUNC currentActiFun) :singleLayer(inDimCol*inDimRow, currentActiFun), dimCol(inDimCol), dimRow(inDimRow)
	{
		default_random_engine dre(clock());
		uniform_real_distribution<double> ran(-1.0, 1.0);
		poolBias = ran(dre);
		poolBiasGradient = 0;
	}
	void forwardPropagate()
	{
		for (int i = 0; i < BATCH_SIZE; i++)
		{
			for (int j= 0; j < dim; j++)
			{
				outputValue[i][j] = currentFunc(inputValue[i][j] + poolBias);
				inputValue[i][j] = 0;
			}
			
		}
	}
	void backPropagate()
	{
		poolBiasGradient = 0;
		for (int i = 0; i < BATCH_SIZE; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				delta[i][j] = outputGradient[i][j] * currentFunc.diff(outputValue[i][j]);
				poolBiasGradient += delta[i][j];
				outputGradient[i][j] = 0;
			}
		}
	}
	void updateBias(double biasStepsize)
	{
		poolBias -= biasStepsize*poolBiasGradient;
		poolBiasGradient = 0;
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