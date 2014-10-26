#include "connection.h"
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
				isConnected[i][j] = 1.0;
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
			propagateResult += bias[i];
			output[i] = propagateResult;
		});
	}
	void backPropagate(const vector<float>& nextLayerDelta, vector<float>& preLayerGradient,const vector<float>& preLayerOutput)
	{
		parallel_for(0,inputDim,[&](int i)
		{
			float propagateResult = 0;
			propagateResult = avx_product(nextLayerDelta, connectWeight[i]);//using avx
			preLayerGradient[i] = propagateResult;
			std::transform(nextLayerDelta.cbegin(), nextLayerDelta.cend(), weightGradient[i].begin(), [&](float in){return in*preLayerOutput[i]; });
		});
		biasGradient = nextLayerDelta;//for the bias up to now doesn't support dropout
		//the weightGradient
	}
	void updateBias(float stepSize, const vector<float>& isRemained)
	{
		transform(isRemained.cbegin(), isRemained.cend(), biasGradient.cbegin(), bias.begin(), [&](int a, int b){return a*b*stepSize; });
	}
	void updateWeight(float stepSize, const vector<float>& isRemained)
	{
		parallel_for(0, inputDim, [&](int i)
		{
			for (int j = 0; j < outputDim; j++)
			{
				connectWeight[i][j] -= isRemained[i] * stepSize*weightGradient[i][j];
			}
		});
	}
};