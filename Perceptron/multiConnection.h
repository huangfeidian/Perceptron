#include "singleConnection.h"
#include "multiLayer.h"
#include "accelerateFor.h"


class multiConnection
{
public:
	const int preFeaMapNumber;//afterFeaMapNumber=connectionNumber
	const int afterFeaMapNumber;//
	int connectionNumber;//how many connections have been added
	vector<singleConnection*> feaMapConnect;//feaMapConnect.size()=connectionNumber
	vector<map<int,int>> preToAfter;//the connection between pre layer map to next layer map[i] mapToMapCon[i]={a,b,c}
	vector<map<int,int>> afterFromPre;//the connection between pre layer map[a] to next layer map afterFromPre[a]={i} etc
	vector<pair<int, int>> connectionRelation;
	multiConnection(int pfmNumber, int afmNumber) :afterFeaMapNumber(afmNumber), preFeaMapNumber(pfmNumber), connectionNumber(0)
		, preToAfter(pfmNumber, map<int, int>()), afterFromPre(afmNumber, map<int, int>())
	{
		
	}
	void addConnection(int preMapIndex, int afterMapindex, singleConnection* currrentConnection)
	{

		feaMapConnect.push_back(currrentConnection);
		preToAfter[preMapIndex][afterMapindex] = connectionNumber;
		afterFromPre[afterMapindex][preMapIndex] = connectionNumber;
		connectionNumber++;
		connectionRelation.push_back(make_pair(preMapIndex, afterMapindex));
	}
	void forwardPropagate(const multiLayer* preLayers, const multiLayer* afterLayers)
	{
		for (int i = 0; i < afterFeaMapNumber; i++)
		{
			for (auto currentConnect : afterFromPre[i])
			{
				feaMapConnect[currentConnect.second]->forwardPropagate(preLayers->featureMaps[currentConnect.first]->outputValue, 
					afterLayers->featureMaps[i]->inputValue);
			}
		}
	}
	void backPropagate(const multiLayer* afterLayers, multiLayer* preLayers)
	{
		for (int i = 0; i < preFeaMapNumber; i++)
		{
			for (auto currentConnect : preToAfter[i])
			{
				feaMapConnect[currentConnect.second]->backPropagate(afterLayers->featureMaps[currentConnect.first]->delta, preLayers->featureMaps[i]->outputGradient
					,preLayers->featureMaps[i]->outputValue);
			}
		}
	}
	void updateWeight(float stepSize,const multiLayer* preLayer)
	{
		for (int i = 0; i < connectionNumber; i++)
		{
			feaMapConnect[i]->updateWeight(stepSize,preLayer->featureMaps[connectionRelation[i].first]->isRemained);
		}
	}
	void consoleWeightOutput()
	{
		for (int i = 0; i < connectionNumber; i++)
		{
			feaMapConnect[i]->consoleWeightOutput();
			auto temp = connectionRelation[i];
			cout << "this connection connects the premap " << temp.first << " to aftermap " << temp.second << endl;
		}
		cout << "current multiConnection weight output finish" << endl<<endl;
	}
};