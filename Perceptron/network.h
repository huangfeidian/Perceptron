#include "connection.h"
class network
{
private:
	int layerNum;
	const int inputDim;
	int outputDim;
public:
	std::vector<layer &>allLayers;
	std::vector<connection&> allConnections;
	network(int inputDim) :inputDim(inputDim), outputDim(inputDim), layerNum(0)
	{

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
	vector<float> singleCaseOutput(const vector<float>& inputCase)
	{
		vector<float> inputVector(inputCase);
		for (int i = 0; i < layerNum - 1; i++)//do something to eliminate the vector copy next time
		{
			allConnections[i].forwardPropagate(inputVector, allLayers[i].inputValue);
			allLayers[i].update();
			inputVector = move(vector<float>(allLayers[i].outputValue));
		}
		return inputVector;

	}
};