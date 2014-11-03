#include "singleLayer.h"
class multiLayer
{
public:
	vector<layer&> featureMaps;
	int featureMapNumber;
	int outputDim;
	multiLayer(int FeaMapNum, int inRow, int inCol, ACTIVATEFUNC currentFuncType) :outputDim(inRow*inCol), featureMapNumber(FeaMapNum)
	{
		for (int i = 0; i < FeaMapNum; i++)
		{
			layer tempLayer(inRow*inCol,currentFuncType);
			featureMaps.push_back(tempLayer);
		}
	}
	void updateBias(float biasstep)
	{
		for (int i = 0; i < featureMapNumber; i++)
		{
			featureMaps[i].updateBias(biasstep);
		}
	}
	
};