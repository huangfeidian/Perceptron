#include "singleConnection.h"
#include "multiLayer.h"
class multiConnection
{
public:
	const int connectionNumber;//
	const int preFeaMapNumber;//afterFeaMapNumber=connectionNumber
	const int afterFeaMapNumber;//
	vector<connection*> feaMapConnect;//feaMapConnect.size()=connectionNumber
	const int windowRow;
	const int windowColumn;
	const int preMapRow;
	const int preMapColumn;
	vector<vector<int>> preToAfter;//the connection between pre layer map to next layer map[i] mapToMapCon[i]={a,b,c}
	vector<vector<int>> afterFromPre;//the connection between pre layer map[a] to next layer map afterFromPre[a]={i} etc
	multiConnection(int pfmNumber, int windowSize, int afmNumber,int preMapSize, const vector<vector<bool>>& layerConnect) :connectionNumber(afterFeaMapNumber), afterFeaMapNumber(afmNumber),
		windowRow(windowSize), windowColumn(windowSize), preFeaMapNumber(pfmNumber), preMapRow(preMapSize), preMapColumn(preMapSize)
	{
		feaMapConnect.reserve(afterFeaMapNumber);
		vector<vector<bool>> connected(preMapSize*preMapSize);
		//set connect matrix
		for (int i = 0; i < preMapSize; i++)
		{
			for (int j = 0; j < preMapSize; j++)
			{
				connected[i*preMapSize + j]=vector<bool>((preMapSize - windowSize + 1)*(preMapSize - windowSize + 1),false);
			}
		}
		for (int i = 0; i < preMapSize - windowSize + 1; i++)
		{
			for (int j = 0; j < preMapSize - windowSize + 1; i++)
			{
				for (int k = 0; k < windowSize; k++)
				{
					for (int l = 0; l < windowSize; l++)
					{
						connected[(i + k)*preMapSize + j + l][i*(preMapSize - windowSize + 1) + j] = true;
					}
				}
			}
		}
		for (int i = 0; i < connectionNumber; i++)
		{
			feaMapConnect[i] = new connection(preMapSize*preMapSize, (preMapSize - windowSize + 1)*(preMapSize - windowSize + 1));	
			feaMapConnect[i]->setConnected(connected);
		}
		//set connect map
		preToAfter.reserve(pfmNumber);
		afterFromPre.reserve(afmNumber);

		for (int i = 0; i < pfmNumber; i++)
		{
			for (int j = 0; j < afmNumber; j++)
			{
				if (layerConnect[i][j] == true)
				{
					preToAfter[i].push_back(j);
					afterFromPre[j].push_back(i);
				}
			}
		}
	}
	void forwardPropagate(const multiLayer& preLayers, const multiLayer& afterLayers)
	{
		for (int i = 0; i < afterFeaMapNumber; i++)
		{
			for (auto preMapIndex : afterFromPre[i])
			{
				feaMapConnect[i]->forwardPropagate(preLayers.featureMaps[preMapIndex].outputValue, afterLayers.featureMaps[i].inputValue);
			}
		}
	}
	void backPropagate(const multiLayer& afterLayers, multiLayer preLayers)
	{
		for (int i = 0; i < preFeaMapNumber; i++)
		{
			for (auto afterMapIndex : preToAfter[i])
			{
				feaMapConnect[afterMapIndex]->backPropagate(afterLayers.featureMaps[afterMapIndex].delta, preLayers.featureMaps[i].outputGradient);
			}
		}
	}
	void updateWeight(float stepSize,const multiLayer& nextLayer)
	{
		for (int i = 0; i < connectionNumber; i++)
		{
			feaMapConnect[i]->updateWeight(stepSize,nextLayer.featureMaps[i].isRemained);
		}
	}
};