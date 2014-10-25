#include "layer.h"
class connection
{
public:
	std::vector<std::vector<int>> connectWeight;//the weights of connections between layers,currently i don't care the sparse before this demo works
	const int inputDim;
	const int outputDim;
	std::vector<std::vector<int>> isConnected;//if node i in prev layer and node j in next layer is connected then isConnected[i][j]=1,else isConnected[i][j]=0
	std::vector<std::vector<float>> weightGradient;//for the weightGradient of the weight
	std::vector<std::map<int, int>> weightFromInput;//weightFromInput[i][j]=connectWeight[i][j]
	std::vector<std::map<int, int>> weightToOutput;//weightToOutput[i][j]=connecWeight[j][i]

	int totalConnections;//sum of  all isConnected[i][j]!=0
	std::vector<float>  bias;// for the bias
	std::vector<float>  biasGradient;//biasGradient[i]=delta[i]
	connection(int inDim, int outDim, const std::vector<std::vector<bool>>& connect) :inputDim(inDim), outputDim(outDim)
	{
		connectWeight.reserve(inDim);
		isConnected.reserve(inDim);
		weightGradient.reserve(inDim);

		weightFromInput.reserve(inDim);
		weightToOutput.reserve(outDim);
		bias.reserve(inDim);
		biasGradient.reserve(inDim);
		totalConnections = 0;
		for (int i = 0; i < inDim; i++)
		{
			connectWeight[i].reserve(outDim);
			isConnected[i].reserve(outDim);
			weightGradient.reserve(outDim);
			for (int j = 0; j < outDim; j++)
			{
				isConnected[i][j] = connect[i][j] == true ? 1 : 0;
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
	void addConnection(int fromIndex, int toIndex, float weight)
	{
		totalConnections++;
		connectWeight[fromIndex][toIndex] = weight;
		isConnected[fromIndex][toIndex] = 1;
		weightFromInput[fromIndex][toIndex] = toIndex;
		weightToOutput[toIndex][fromIndex] = fromIndex;
	}
	void forwardPropagate(const vector<float>& input, vector<float>& output)
	{
		for (int i = 0; i < outputDim; i++)//we can use multithread or multithread
		{
			float propagateResult = 0;
			for (auto singleConnection : weightToOutput[i])//we can use sse or avx
			{
				propagateResult += input[singleConnection.first] * connectWeight[singleConnection.second][i];
			}
			propagateResult += bias[i];
			output[i] = propagateResult;
		}
	}
	void backPropagate(const vector<float>& nextLayerDelta, vector<float>& preLayerGradient)
	{
		for (int i = 0; i < inputDim; i++)//for the layer nodes
		{
			float propagateResult = 0;
			for (auto singleConnection : weightFromInput[i])
			{
				propagateResult += nextLayerDelta[singleConnection.first] * connectWeight[i][singleConnection.second];
			}
			preLayerGradient[i] = propagateResult;
		}
		biasGradient = nextLayerDelta;//for the bias up to now doesn't support dropout
		//begin update the weight
		for (int i = 0; i < outputDim; i++)
		{
			for (auto singleConnection : weightToOutput[i])
			{
				weightGradient[singleConnection.first][i] = nextLayerDelta[i];
			}
		}
	}
	void updateBias(float stepSize, const vector<int>& isRemained)
	{
		for (int i = 0; i < outputDim; i++)//we can parallize it and sse 
		{
			bias[i] -= stepSize* isRemained[i] * biasGradient[i];
		}
	}
	void updateWeight(float stepSize, const vector<int>& isRemained)
	{
		for (int i = 0; i < inputDim; i++)
		{
			for (auto singleConnection : weightFromInput[i])//make use of sse
			{
				connectWeight[i][singleConnection.first] -= isRemained[i] * stepSize*weightGradient[i][singleConnection.first];
			}

		}
	}

};
