
#include <iostream>
#include "network.h"
#include "fullConnection.h"
using namespace std;
int main()
{
	network currentNet(4, LOSSFUNC::MSE);
	fullConnection  hehe(4, 1);
	singleLayer out(1, ACTIVATEFUNC::SIGMOID);
	multiLayer theonlylayer(1,1);
	theonlylayer.addSingleLayer(&out);
	multiConnection theonlyConnection(1, 1);
	theonlyConnection.addConnection(0, 0, &hehe);
	currentNet.addLayerAndConnection(theonlylayer, theonlyConnection);
	vector<float> lalallatest{ 1, 2, 3, 4 };
	currentNet.singleCaseOutput(lalallatest);
	cout << currentNet.allLayers[1].featureMaps[0]->outputValue[0];
}