#include "multiConnection.h"
#include "multiLayer.h"
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
	std::vector<multiLayer &>allLayers;
	std::vector<multiConnection&> allConnections;
	network(int inputDim, LOSSFUNC currentLossFunctype) :inputDim(inputDim), outputDim(inputDim), layerNum(1), currentLoss(currentLossFunctype), output(inputDim)
	{
		layer inputLayer(inputDim, ACTIVATEFUNC::IDENTITY);
	}
	void addLayerAndConnection(multiLayer& layerToAdd, multiConnection& connectionToAdd)
	{
		layerNum++;
		outputDim = layerToAdd.outputDim;
		allLayers.push_back(layerToAdd);
		allConnections.push_back(connectionToAdd);
	}
	void dropout(int layerIndex, float dropoutRate)
	{
		assert(dropoutRate < 1 && dropoutRate>0);
		int numberToRemain = allLayers[layerIndex].outputDim*(1 - dropoutRate);
		for (int i = 0; i < allLayers[layerIndex].featureMapNumber; i++)
		{
			allLayers[layerIndex].featureMaps[i].dropoutNodes(numberToRemain);
		}
		
	}
	void dropoutRestore(int layerIndex)
	{
		for (int i = 0; i < allLayers[layerIndex].featureMapNumber; i++)
		{
			allLayers[layerIndex].featureMaps[i].dropoutRestore();
		}
	}
	void  singleCaseOutput(const vector<float>& inputCase)
	{
		allLayers[0].featureMaps[0].outputValue=inputCase;
		for (int i = 0; i < layerNum ; i++)//do something to eliminate the vector copy next time
		{
			allConnections[i].forwardPropagate(allLayers[i].outputValue, allLayers[i+1].inputValue);
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
		allLayers[layerNum ].outputGradient.swap(initGradient);
		vector<float> initDelta;
		for (int i = layerNum ; i > 0; i--)
		{
			transform(allLayers[i].outputGradient.cbegin(), allLayers[i].outputGradient.cend(), initDelta.begin(), [&](float input)
			{
				return allLayers[i].currentFunc.diff(input);
			});
			allConnections[i-1].backPropagate(initDelta, allLayers[i - 1].outputGradient);
		}
	}
	void updateNetwork(float biasStepSize,float weightStepSize)
	{
		parallel_for(0, layerNum - 1, [&](int index)
		{
			allLayers[index+1].updateBias(biasStepSize);
			allConnections[index].updateWeight(weightStepSize,allLayers[index].isRemained);
		});
	}
	void trainbatch(const vector<vector<float>>& batchInput,const vector<vector<float>>& batchResult, int beginIndex,int batchSize)
	{
		int totalSize = batchInput.size();
		int currentBatch = (totalSize - beginIndex) > batchSize ? batchSize : (totalSize - beginIndex);
		for (int i = 0; i < currentBatch; i++)
		{
			singleCaseOutput(batchInput[beginIndex + i]);
			singleCaseBackProp(batchResult[beginIndex + i]);
		}
	}

};