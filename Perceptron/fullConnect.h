#include "singleConnection.h"
#include <ppl.h>
using namespace concurrency;
class fullConnection :public connection
{
public:
	vector<vector<float>> reverseWeight;//reverseWeight[i][j]=connectweight[j][i]
	fullConnection(int inDim, int outDim) :connection(inDim, outDim)
	{
		for (int i = 0; i < inDim; i++)
		{
			for (int j = 0; j < outDim; j++)
			{
				isConnected[i][j] = 1;
			}
		}
	}
	void addConnection(int fromIndex, int toIndex, float weight)
	{
		totalConnections++;
		connectWeight[fromIndex][toIndex] = weight;
		reverseWeight[toIndex][fromIndex] = weight;
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
		parallel_for (0,outputDim,[&](int i)
		{
			float propagateResult = 0;
			propagateResult = avx_product(input, reverseWeight[i]);
			output[i] += propagateResult;
		});
	}
	void backPropagate(const vector<float>& nextLayerDelta, vector<float>& preLayerGradient,const vector<float>& preLayerOutput)
	{
		parallel_for(0,inputDim,[&](int i)
		{
			float propagateResult = 0;
			propagateResult = avx_product(nextLayerDelta, connectWeight[i]);//using avx
			preLayerGradient[i] = propagateResult;
			std::transform(nextLayerDelta.cbegin(), nextLayerDelta.cend(), outputGradient[i].begin(), [&](float in){return in*preLayerOutput[i]; });
			for (int j = 0; j < outputDim; j++)
			{
				batchWeightGradient[i][j] += outputGradient[i][j];
			}
		});
		//the outputGradient
	}
	void updateWeight(float stepSize, const vector<float>& isRemained)
	{
		parallel_for(0, inputDim, [&](int i)
		{
			for (int j = 0; j < outputDim; j++)
			{
				connectWeight[i][j] -= isRemained[i] * stepSize*outputGradient[i][j];
				batchWeightGradient[i][j] = 0;//clear the batchsum
			}
		});
	}
};