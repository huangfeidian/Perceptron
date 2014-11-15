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

	void forwardPropagate()
	{
		accelerateFor (0,featureMapNumber,[&](int i)
		{
			featureMaps[i]->forwardPropagate();
		});

#ifdef FILE_DEBUG
		fileResultOutput();
#endif
	}
	void backPropagate()
	{
		accelerateFor( 0, featureMapNumber,[&](int i)
		{
			featureMaps[i]->backPropagate();
		});
	}
	void updateBias(double biasstep)
	{
		accelerateFor( 0, featureMapNumber,[&](int i)
		{
			featureMaps[i]->updateBias(biasstep);
		});
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
	void fileBiasOutput(ofstream& outFile)
	{
		outFile << " multimap bias output" << endl;
		for (int i = 0; i < featureMapNumber; i++)
		{
			featureMaps[i]->fileBiasOutput(outFile);
		}
	}
	void loadBiasFromFile(ifstream& inputFile)
	{
		char temp[100];
		inputFile.getline(temp, 99);//eat the illlustration line
		for (int i = 0; i < featureMapNumber; i++)
		{
			featureMaps[i]->loadBiasFromFile(inputFile);
		}
	}
	void fileResultOutput()
	{
		ofstream outFile("singleCase.txt", ios::app);
		outFile << " multimap result output" << endl;
		outFile.close();
		for (int i = 0; i < featureMapNumber; i++)
		{
			featureMaps[i]->fileResultOutput();
		}
		
	}
};