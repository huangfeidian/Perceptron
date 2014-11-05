
#include <iostream>
#include "network.h"
#include "fullConnection.h"
using namespace std;
int main()
{
	network currentNet(10, LOSSFUNC::MSE);
	fullConnection  hehe(10, 5);
	singleLayer thefirstconnect(5, ACTIVATEFUNC::SIGMOID);
	multiLayer thefirstlayer(1,5);
	thefirstlayer.addSingleLayer(&thefirstconnect);
	multiConnection thefirstConnection(1, 1);
	thefirstConnection.addConnection(0, 0, &hehe);
	currentNet.addLayerAndConnection(&thefirstlayer, &thefirstConnection);
	fullConnection  haha(5, 1);
	singleLayer thesecondconnect(1, ACTIVATEFUNC::SIGMOID);
	multiLayer thesecondlayer(1, 1);
	thesecondlayer.addSingleLayer(&thesecondconnect);
	multiConnection thesecondConnection(1, 1);
	thesecondConnection.addConnection(0, 0, &haha);
	currentNet.addLayerAndConnection(&thesecondlayer, &thesecondConnection);
	vector<float> lalallatest{ 1, 1,1,1,1,1,1,1,1,1 };
	currentNet.singleCaseOutput(lalallatest);
	int i = 1;
	for (int i = 1; i < 3; i++)
	{
		currentNet.allLayers[i]->consoleValueOutput();
	}
	cout << "**************************" << endl;
	for (int i = 1; i < 3; i++)
	{
		currentNet.allLayers[i]->consoleBiasOutput();
	}
	currentNet.allConnections[0]->consoleWeightOutput();
	currentNet.allConnections[1]->consoleWeightOutput();
	cout << currentNet.allLayers[2]->featureMaps[0]->outputValue[0]<<endl;
}