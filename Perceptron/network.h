#include "multiConnection.h"
#include "multiLayer.h"
#include "lossFunction.h"
#include "convolConnection.h"
#include "poolConnection.h"
#include "poolLayer.h"
#include "fullConnection.h"
#include "convolLayer.h"
#include "fullLayer.h"
class network
{
public:
	int layerNum;
	const int inputDim;
	int outputDim;
	lossFunc currentLoss;
	vector<double> output;
public:
	std::vector<multiLayer*>allLayers;
	std::vector<multiConnection*> allConnections;
	network(int inputDim, int finalOutDim ,LOSSFUNC currentLossFunctype) :inputDim(inputDim), outputDim(inputDim), layerNum(1), currentLoss(currentLossFunctype), output(finalOutDim,0)
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
			assert(layerToAdd->featureMaps[connectFromTo.second]->dim == connectionToAdd->feaMapConnect[i]->outputDim);
		}
#endif
		layerNum++;
		outputDim = layerToAdd->outputDim;
		allLayers.push_back(layerToAdd);
		allConnections.push_back(connectionToAdd);
	}
	void addConvolutionLayerAndConnection(int preLayerImageSize,int windowSize, vector<vector<bool>>& connectionRelation,ACTIVATEFUNC currentActivateFunction)
	{
		//todo
#ifdef CHECK_LEGITIMATE
		assert(connectionRelation.size() == allLayers[layerNum - 1]->featureMapNumber);
#endif

		int nextLayerMapNumber = connectionRelation[0].size();
		int preLayerMapNumber = connectionRelation.size();
		int currentLayerOutDim = (preLayerImageSize - windowSize + 1)*(preLayerImageSize - windowSize + 1);
		multiLayer* currentConvolutionLayer = new multiLayer(nextLayerMapNumber,currentLayerOutDim);
		multiConnection* currentConvolutionConnection = new multiConnection(preLayerMapNumber, nextLayerMapNumber);
		for (int i = 0; i < nextLayerMapNumber; i++)
		{
			convolLayer* currentSingleConvolutionLayer = new convolLayer(preLayerImageSize - windowSize + 1, preLayerImageSize - windowSize + 1,currentActivateFunction);
			currentConvolutionLayer->addSingleLayer(currentSingleConvolutionLayer);
		}
		for (int i = 0; i < preLayerMapNumber; i++)
		{
			for (int j = 0; j < nextLayerMapNumber; j++)
			{
				if (connectionRelation[i][j] == true)
				{
					convolutionConnection* currentSingleConvolutionConnection = new convolutionConnection(preLayerImageSize, preLayerImageSize, windowSize);
					currentConvolutionConnection->addConnection(i, j, currentSingleConvolutionConnection);
				}
			}
		}
		addLayerAndConnection(currentConvolutionLayer, currentConvolutionConnection);
	}
	void addPoolLayerAndConnection(int preLayerImageSize, int windowSize, ACTIVATEFUNC currentActivateFunction)
	{

#ifdef CHECK_LEGITIMATE
		assert(preLayerImageSize%windowSize == 0);
#endif

		int currentLayerOutDim = (preLayerImageSize / windowSize)*(preLayerImageSize / windowSize);
		int preLayerMapNumber = allLayers[layerNum - 1]->featureMapNumber;
		multiLayer* currentPoolLayer = new multiLayer(preLayerMapNumber, currentLayerOutDim);
		multiConnection* currentPoolConnnection = new multiConnection(preLayerMapNumber, preLayerMapNumber);
		for (int i = 0; i < preLayerMapNumber; i++)
		{
			poolLayer* currentSinglePoolLayer = new poolLayer(preLayerImageSize / windowSize, preLayerImageSize / windowSize,currentActivateFunction);
			poolConnection* currentSinglePoolConnection = new poolConnection(preLayerImageSize, preLayerImageSize, windowSize);
			currentPoolLayer->addSingleLayer(currentSinglePoolLayer);
			currentPoolConnnection->addConnection(i, i, currentSinglePoolConnection);
		}
		addLayerAndConnection(currentPoolLayer, currentPoolConnnection);
	}
	void addFullLayerAndConnection(int fullLayerDim, ACTIVATEFUNC currentActivateFunction)
	{

		multiConnection* currentFullConnection = new multiConnection(allLayers[layerNum - 1]->featureMapNumber, 1);
		multiLayer* currentFullLayer = new multiLayer(1, fullLayerDim);
		fullLayer* currentSingleFullLayer = new fullLayer(fullLayerDim, currentActivateFunction);
		currentFullLayer->addSingleLayer(currentSingleFullLayer);
		//todo
		for (int i = 0; i < allLayers[layerNum - 1]->featureMapNumber; i++)
		{
			fullConnection* currentSingleFullConnection = new fullConnection(outputDim, fullLayerDim);
			currentFullConnection->addConnection(i, 0, currentSingleFullConnection);
		}
		addLayerAndConnection(currentFullLayer, currentFullConnection);
	}
	
	void dropout(int layerIndex, double dropoutRate)
	{
#ifdef CHECK_LEGITIMATE
		assert(dropoutRate < 1 && dropoutRate>0);
#endif
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
	void  singleCaseOutput(const vector<double>& inputCase)
	{
		allLayers[0]->featureMaps[0]->outputValue=inputCase;
		for (int i = 0; i < layerNum-1 ; i++)
		{
			allConnections[i]->forwardPropagate(allLayers[i], allLayers[i+1]);
			allLayers[i + 1]->forwardPropagate();
		}
		for (int i = 0; i < outputDim; i++)
		{
			output[i] = allLayers[layerNum - 1]->featureMaps[0]->outputValue[i];
		}
		
	}
	vector<double> setBackGradient(const vector<double>& realResult)
	{
		return currentLoss.diff(output, realResult);
	}
	void singleCaseBackProp(const vector<double>& realResult)
	{
		auto initGradient = currentLoss.diff(output, realResult);
		allLayers[layerNum-1 ]->featureMaps[0]->outputGradient.swap(initGradient);
		for (int i = layerNum-1 ; i > 0; i--)
		{
			allLayers[i]->backPropagate();
			allConnections[i - 1]->backPropagate(allLayers[i], allLayers[i - 1]);
		}
	}
	void updateNetwork(double biasStepSize,double weightStepSize)
	{
		for(int index=0; index<layerNum - 1;index++)
		{
			allLayers[index+1]->updateBias(biasStepSize);
			allConnections[index]->updateWeight(weightStepSize,allLayers[index]);
		}
	}
	void trainbatch(const vector<vector<double>>& batchInput,const vector<vector<double>>& batchResult, int beginIndex,int batchSize)
	{
		int totalSize = batchInput.size();
		int currentBatch = (totalSize - beginIndex) > batchSize ? batchSize : (totalSize - beginIndex);
		for (int i = 0; i < currentBatch; i++)
		{
			singleCaseOutput(batchInput[beginIndex + i]);
			singleCaseBackProp(batchResult[beginIndex + i]);
		}
	}
	void fileNetworkOutput(ofstream& OutFile)
	{
		for (int i = 0; i < layerNum - 1; i++)
		{
			allConnections[i]->fileWeightOutput(OutFile);
			allLayers[i + 1]->fileBiasOutput(OutFile);
		}
	}
	void loadNetworkFromFile(ifstream& inputFile)
	{
		for (int i = 0; i < layerNum - 1; i++)
		{
			allConnections[i]->loadWeightFromFile(inputFile);
			allLayers[i + 1]->loadBiasFromFile(inputFile);
		}
	}
};