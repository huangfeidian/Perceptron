#include "singleLayer.h"
using namespace std;
class poolLayer :singleLayer
{
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
	void updateBias(float biasStepsize)
	{
		// do nothing
	}
	void consoleBiasOutput()
	{
		cout << "this layer is pool layer ,no bias " << endl;
	}
};