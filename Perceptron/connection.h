#include "layer.h"
#include "vectorize.h"
#include <ppl.h>
#include <algorithm>
using namespace std;
class connection
{
public:
	std::vector<std::vector<float>> connectWeight;//the weights of connections between layers,currently i don't care the sparse before this demo works
	const int inputDim;
	const int outputDim;
	std::vector<std::vector<int>> isConnected;//if node i in prev layer and node j in next layer is connected then isConnected[i][j]=1,else isConnected[i][j]=0
	std::vector<std::vector<float>> outputGradient;//for the Gradient of the weight
	std::vector<std::vector<float>> batchWeightGradient;//for the batch sum of  Gradient of the weight
	std::vector<std::map<int, int>> weightFromInput;//weightFromInput[i][j]=connectWeight[i][j]
	std::vector<std::map<int, int>> weightToOutput;//weightToOutput[i][j]=connecWeight[j][i]

	int totalConnections;//sum of  all isConnected[i][j]!=0
	

	connection(int inDim, int outDim) :inputDim(inDim), outputDim(outDim), totalConnections(0)
	{
		connectWeight.reserve(inDim);
		isConnected.reserve(inDim);
		outputGradient.reserve(inDim);

		weightFromInput.reserve(inDim);
		weightToOutput.reserve(outDim);
		for (int i = 0; i < inDim; i++)
		{
			connectWeight[i].reserve(outDim);
			isConnected[i].reserve(outDim);
			outputGradient.reserve(outDim);
			for (int j = 0; j < outDim; j++)
			{
				isConnected[i][j] = 0;
			}
		}
	}
	void setConnected(const vector<vector<bool>>& inputIsConnected)
	{
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				isConnected[i][j] = inputIsConnected[i][j] == true ? 1 : 0;
			}
		}
	}
	void initWeight()
	{
		std::default_random_engine dre;
		std::uniform_real_distribution<float> di(-1.0, 1.0);
		float tempWeight;
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				if (isConnected[i][j] == 1)
				{
					tempWeight = di(dre);
					addConnection(i, j, tempWeight);
				}
			}
		}
	}
	inline void addConnection(int fromIndex, int toIndex, float weight)
	{
		totalConnections++;
		connectWeight[fromIndex][toIndex] = weight;
		isConnected[fromIndex][toIndex] = 1;
		weightFromInput[fromIndex][toIndex] = toIndex;
		weightToOutput[toIndex][fromIndex] = fromIndex;
	}
	virtual void forwardPropagate(const vector<float>& input, vector<float>& output)
	{
		parallel_for ( 0, outputDim,[&](int i)//we can use multithread or multithread
		{
			float propagateResult = 0;
			for (auto singleConnection : weightToOutput[i])//we can use sse or avx
			{
				propagateResult += input[singleConnection.first] * connectWeight[singleConnection.second][i];
			}
			output[i] += propagateResult;
		});
	}
	virtual void backPropagate(const vector<float>& nextLayerDelta, vector<float>& preLayerGradient)
	{
		//we can gain more parallel
		parallel_for(0,  inputDim,[&](int i)//for the layer nodes
		{
			float propagateResult = 0;
			for (auto singleConnection : weightFromInput[i])
			{
				propagateResult += nextLayerDelta[singleConnection.first] * connectWeight[i][singleConnection.second];
			}
			preLayerGradient[i] = propagateResult;
		});
		
		//begin update the weight
		parallel_for (0,  outputDim,[&](int i)
		{
			for (auto singleConnection : weightToOutput[i])
			{
				float temp = nextLayerDelta[i] * nextLayerDelta[singleConnection.first];
				outputGradient[singleConnection.first][i] =temp ;
				batchWeightGradient[singleConnection.first][i] += temp;
			}
		});
	}
	
	virtual void updateWeight(float stepSize, const vector<float>& isRemained)
	{
		parallel_for ( 0, inputDim,[&](int i)
		{
			for_each(weightFromInput[i].cbegin(), weightFromInput[i].cend(), [&](const pair<int, int>& in)
			{
				connectWeight[i][in.first] -= isRemained[i] * stepSize*outputGradient[i][in.second];
				batchWeightGradient[i][in.first] = 0;//clear the batch sum
			});
		});
	}
};
