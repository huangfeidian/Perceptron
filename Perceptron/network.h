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
	std::vector<multiLayer*>allLayers;
	std::vector<multiConnection*> allConnections;
	network(int inputDim, LOSSFUNC currentLossFunctype) :inputDim(inputDim), outputDim(inputDim), layerNum(1), currentLoss(currentLossFunctype), output(inputDim)
	{
		singleLayer* inputLayer=new singleLayer(inputDim, ACTIVATEFUNC::IDENTITY);
		multiLayer* firstMultiLayer = new multiLayer(1, inputDim);
		firstMultiLayer->addSingleLayer(inputLayer);
		allLayers.push_back(firstMultiLayer);
	}
	void addLayerAndConnection(multiLayer* layerToAdd, multiConnection* connectionToAdd)
	{
#ifdef CHECK_LEGITIMATE
		for (int i = 0; i < connectionToAdd->connectionNumber; i++)
		{
			auto connectFromTo = connectionToAdd->connectionRelation[i];
			assert(((allLayers[layerNum - 1])->featureMaps[connectFromTo.first])->dim == connectionToAdd->feaMapConnect[i]->inputDim);
			assert(layerToAdd->featureMaps[connectFromTo.first]->dim == connectionToAdd->feaMapConnect[i]->outputDim);
		}
#endif
		layerNum++;
		outputDim = layerToAdd->outputDim;
		allLayers.push_back(layerToAdd);
		allConnections.push_back(connectionToAdd);
	}
	void dropout(int layerIndex, float dropoutRate)
	{
		assert(dropoutRate < 1 && dropoutRate>0);
		int numberToRemain = allLayers[layerIndex]->outputDim*(1 - dropoutRate);
		for (int i = 0; i < allLayers[layerIndex]->featureMapNumber; i++)
		{
			allLayers[layerIndex]->featureMaps[i]->dropoutNodes(numberToRemain);
		}
		
	}
	void dropoutRestore(int layerIndex)
	{
		for (int i = 0; i < allLayers[layerIndex]->featureMapNumber; i++)
		{
			allLayers[layerIndex]->featureMaps[i]->dropoutRestore();
		}
	}
	void  singleCaseOutput(const vector<float>& inputCase)
	{
		allLayers[0]->featureMaps[0]->outputValue=inputCase;
		for (int i = 0; i < layerNum-1 ; i++)//do something to eliminate the vector copy next time
		{
			allConnections[i]->forwardPropagate(allLayers[i], allLayers[i+1]);
			allLayers[i + 1]->forwardPropagate();
		}
	}
	vector<float> setBackGradient(const vector<float>& realResult)
	{
		return currentLoss.diff(output, realResult);
	}
	void singleCaseBackProp(const vector<float>& realResult)
	{
		auto initGradient = currentLoss.diff(output, realResult);
		allLayers[layerNum ]->featureMaps[0]->outputGradient.swap(initGradient);
		for (int i = layerNum-1 ; i > 0; i--)
		{
			allLayers[i]->backPropagate();
			allConnections[i - 1]->backPropagate(allLayers[i], allLayers[i - 1]);
			allLayers[i]->resetOutputGradient();
		}
	}
	void updateNetwork(float biasStepSize,float weightStepSize)
	{
		for(int index=0; index<layerNum - 1;index++)
		{
			allLayers[index+1]->updateBias(biasStepSize);
			allConnections[index]->updateWeight(weightStepSize,allLayers[index+1]);
		}
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