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
	vector<vector<bool>> isMapConnected;
	multiConnection(int pfmNumber, int afmNumber) :afterFeaMapNumber(afmNumber), preFeaMapNumber(pfmNumber), connectionNumber(0)
		, preToAfter(pfmNumber, map<int, int>()), afterFromPre(afmNumber, map<int, int>()), isMapConnected(pfmNumber, vector<bool>(afmNumber,false))
	{
		
	}
	void addConnection(int preMapIndex, int afterMapindex, singleConnection* currrentConnection)
	{

		feaMapConnect.push_back(currrentConnection);
		isMapConnected[preMapIndex][afterMapindex] = true;
		preToAfter[preMapIndex][afterMapindex] = connectionNumber;
		afterFromPre[afterMapindex][preMapIndex] = connectionNumber;
		connectionNumber++;
		connectionRelation.push_back(make_pair(preMapIndex, afterMapindex));
	}
	void forwardPropagate(const multiLayer* preLayers, const multiLayer* afterLayers)
	{
		accelerateFor (  0, afterFeaMapNumber,[&](int i)
		{
			for (auto currentConnect : afterFromPre[i])
			{
				feaMapConnect[currentConnect.second]->forwardPropagate(preLayers->featureMaps[currentConnect.first]->outputValue, 
					afterLayers->featureMaps[i]->inputValue);
			}
		});
	}
	void backPropagate(const multiLayer* afterLayers, multiLayer* preLayers)
	{
		accelerateFor ( 0,preFeaMapNumber, [&](int i)
		{
			for (auto currentConnect : preToAfter[i])
			{
				feaMapConnect[currentConnect.second]->backPropagate(afterLayers->featureMaps[currentConnect.first]->delta, preLayers->featureMaps[i]->outputGradient
					,preLayers->featureMaps[i]->outputValue);
			}
		});
	}
	void updateWeight(double stepSize,const multiLayer* preLayer)
	{
		accelerateFor( 0, connectionNumber,[&](int i)
		{
			feaMapConnect[i]->updateWeight(stepSize);
		});
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
	void fileWeightOutput(ofstream& outFile)
	{
		for (int i = 0; i < preFeaMapNumber; i++)
		{
			for (int j = 0; j < afterFeaMapNumber; j++)
			{
				if (isMapConnected[i][j] == true)
				{
					outFile << "this connection is from " << i << " to " << j << endl;
					feaMapConnect[preToAfter[i][j]]->fileWeightOutput(outFile);
				}
			}
		}
	}
	void loadWeightFromFile(ifstream& inputFile)
	{
		char temp[100];
		for (int i = 0; i < preFeaMapNumber; i++)
		{
			for (int j = 0; j < afterFeaMapNumber; j++)
			{
				if (isMapConnected[i][j] == true)
				{
					inputFile.getline(temp,100);//eat the illustration line
					feaMapConnect[preToAfter[i][j]]->loadWeightFromFile(inputFile);
				}
			}
		}
	}
};