#include "singleLayer.h"
class multiLayer
{
public:
	vector<singleLayer*> featureMaps;
	int featureMapNumber;
	int outputDim;
	multiLayer(int FeaMapNum, int inRow, int inCol, ACTIVATEFUNC currentFuncType) :outputDim(inRow*inCol), featureMapNumber(FeaMapNum)
	{
		for (int i = 0; i < FeaMapNum; i++)
		{
			singleLayer tempLayer(inRow*inCol,currentFuncType);
			featureMaps.push_back(&tempLayer);
		}
	}
	void resetOutputGradient()
	{
		for (int i = 0; i < featureMapNumber; i++)
		{
			featureMaps[i]->resetOutputGradient();//clear the outputGradient
		}
	}
	void forwardPropagate()
	{
		for (int i = 0; i < featureMapNumber; i++)
		{
			featureMaps[i]->forwardPropagate();
		}
	}
	void backPropagate()
	{
		for (int i = 0; i < featureMapNumber; i++)
		{
			featureMaps[i]->backPropagate();
		}
	}
	void updateBias(float biasstep)
	{
		for (int i = 0; i < featureMapNumber; i++)
		{
			featureMaps[i]->updateBias(biasstep);
		}
	}
	
};