#include "singleLayer.h"
using namespace std;
class poolLayer :public singleLayer
{
public:
	const int dimCol;
	const int dimRow;
	poolLayer(int inDimRow, int inDimCol) :singleLayer(inDimCol*inDimRow, ACTIVATEFUNC::IDENTITY), dimCol(inDimCol), dimRow(inDimRow)
	{
		for (int i = 0; i < dim; i++)
		{
			bias[i] = 0;
		}
	}
	void forwardPropagate()
	{
		for (int i = 0; i < dim; i++)
		{
			outputValue[i] = inputValue[i];
			inputValue[i] = 0;
		}
	}
	void backPropagate()
	{
		for (int i = 0; i < dim; i++)
		{
			delta[i] = outputGradient[i];
		}
	}
	void updateBias(double biasStepsize)
	{
		for (int i = 0; i < dim; i++)
		{
			outputGradient[i] = 0;
			delta[i] = 0;
		}
	}
	void consoleBiasOutput()
	{
		cout << "this layer is pool layer ,no bias " << endl;
	}
};