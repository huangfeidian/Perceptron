#include "connection.h"
#include "lossFunction.h"
class network
{
public:
	int layerNum;
	const int inputDim;
	int outputDim;
	lossFunc currentLoss;
	vector<float> output;
public:
	std::vector<layer &>allLayers;
	std::vector<connection&> allConnections;
	network(int inputDim, LOSSFUNC currentLossFunctype) :inputDim(inputDim), outputDim(inputDim), layerNum(1), currentLoss(currentLossFunctype), output(inputDim)
	{
		layer inputLayer(inputDim, ACTIVATEFUNC::IDENTITY);
	}
	void addLayerAndConnection(layer& layerToAdd, connection& connectionToAdd)
	{
		layerNum++;
		outputDim = layerToAdd.dim;
		allLayers.push_back(layerToAdd);
		allConnections.push_back(connectionToAdd);
	}
	void dropout(int layerIndex, float dropoutRate)
	{
		assert(dropoutRate < 1 && dropoutRate>0);
		int numberToRemain = allLayers[layerIndex].dim*(1 - dropoutRate);
		allLayers[layerIndex].dropoutNodes(numberToRemain);
	}
	void dropoutRestore(int layerIndex)
	{
		allLayers[layerIndex].dropoutRestore();
	}
	void  singleCaseOutput(const vector<float>& inputCase)
	{
		allLayers[0].outputValue=inputCase;
		for (int i = 0; i < layerNum ; i++)//do something to eliminate the vector copy next time
		{
			allConnections[i].forwardPropagate(allLayers[i].outputValue, allLayers[i].inputValue);
			allLayers[i+1].update();
		}
		output=allLayers[layerNum].outputValue;
	}
	vector<float> setBackGradient(const vector<float>& realResult)
	{
		return currentLoss.diff(output, realResult);
	}
	void singleCaseBackProp(const vector<float>& realResult)
	{
		auto initGradient = currentLoss.diff(output, realResult);
		allLayers[layerNum ].weightGradient.swap(initGradient);
		vector<float> initDelta;
		for (int i = layerNum ; i > 0; i--)
		{
			transform(allLayers[i].weightGradient.cbegin(), allLayers[i].weightGradient.cend(), initDelta.begin(), [&](float input)
			{
				return allLayers[i].currentFunc.diff(input);
			});
			allConnections[i-1].backPropagate(initDelta, allLayers[i - 1].weightGradient);
		}
	}
	void updateNetwork(float stepSize)
	{
		parallel_for(0, layerNum - 1, [&](int index)
		{
			allConnections[index].updateBias(stepSize,allLayers[index].isRemained);
			allConnections[index].updateWeight(stepSize,allLayers[index].isRemained);
		});
	}
};