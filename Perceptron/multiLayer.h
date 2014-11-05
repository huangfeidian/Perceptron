#include "singleLayer.h"
#pragma once
class multiLayer
{
public:
	vector<singleLayer*> featureMaps;
	const int featureMapNumber;
	const int outputDim;
	multiLayer(int FeaMapNum, int outDim) : featureMapNumber(FeaMapNum), featureMaps(0), outputDim(outDim)
	{
		//do nothing
	}
	void addSingleLayer(singleLayer* inputLayer)
	{
#ifdef CHECK_LEGITIMATE
		assert(outputDim == inputLayer->dim);
#endif 
		featureMaps.push_back(inputLayer);
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
	void consoleValueOutput()
	{
		for (int i = 0; i < featureMapNumber; i++)
		{
			featureMaps[i]->consoleValueOutput();
		}
		cout << "current multiLayer value output finish" << endl<<endl;
	}
	void consoleBiasOutput()
	{
		for (int i = 0; i < featureMapNumber; i++)
		{
			featureMaps[i]->consoleBiasOutput();
		}
		cout << "current multiLayer bias output finish" << endl<<endl;
	}
};