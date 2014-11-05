#include "singleConnection.h"
//#include "accelerateFor.h"
class fullConnection :public singleConnection
{
public:
	vector<vector<float>> reverseWeight;//reverseWeight[i][j]=connectweight[j][i]
	fullConnection(int inDim, int outDim) :singleConnection(inDim, outDim), reverseWeight(outDim, vector<float>(inDim,0))
	{
		initWeight();
	}
	void addConnection(int fromIndex, int toIndex, float weight)
	{
		totalConnections++;
		connectWeight[fromIndex][toIndex] = weight;
		reverseWeight[toIndex][fromIndex] = weight;
		isConnected[fromIndex][toIndex] = 1;
		weightFromInput[fromIndex][toIndex] = toIndex;
		weightToOutput[toIndex][fromIndex] = fromIndex;
	}
	void initWeight()
	{
		std::default_random_engine dre;
		std::uniform_real_distribution<float> di(-1.0, 1.0);
		
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				float tempWeight;
				tempWeight = di(dre);
				addConnection(i, j, tempWeight);				
			}
		}
	}
	void forwardPropagate(const vector<float>& input, vector<float>& output)
	{
		for (int i = 0; i < outputDim; i++)
		{
			float propagateResult = 0;
			propagateResult = avx_product(input, reverseWeight[i]);
			output[i] += propagateResult;
		};
	}
	void backPropagate(const vector<float>& nextLayerDelta, vector<float>& preLayerGradient,const vector<float>& preLayerOutput)
	{
		accelerateFor(0,inputDim,[&](int i)
		{
			float propagateResult = 0;
			propagateResult = avx_product(nextLayerDelta, connectWeight[i]);//using avx
			preLayerGradient[i] = propagateResult;
			std::transform(nextLayerDelta.cbegin(), nextLayerDelta.cend(), weightGradient[i].begin(), [&](float in){return in*preLayerOutput[i]; });
			for (int j = 0; j < outputDim; j++)
			{
				batchWeightGradient[i][j] += weightGradient[i][j];
			}
		});
		//the weightGradient
	}
	void updateWeight(float stepSize, const vector<float>& isRemained)
	{
		accelerateFor(0, inputDim, [&](int i)
		{
			for (int j = 0; j < outputDim; j++)
			{
				connectWeight[i][j] -= isRemained[i] * stepSize*weightGradient[i][j];
				batchWeightGradient[i][j] = 0;//clear the batchsum
			}
		});
	}
	void consoleWeightOutput()
	{
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				cout << connectWeight[i][j] << ' ';
			}
			cout << endl;
		}
		cout << "current full connection weight" << endl;
	}
};