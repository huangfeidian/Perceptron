
#include "headIntro.h"
using namespace std;
int main()
{
	vector<int> trainLabels, testLabels;
	vector<vector<double>> trainImages, testImages;
	//parse_mnist_labels("train-labels.idx1-ubyte", &trainLabels);
	//parse_mnist_images("train-images.idx3-ubyte", &trainImages);
	//parse_mnist_labels("t10k-labels.idx1-ubyte", &testLabels);
	//parse_mnist_images("t10k-images.idx3-ubyte", &testImages);
	network currentNet(32*32, LOSSFUNC::MSE);
	fullConnection thefirstsingleconnection(32*32,16*16);
	multiConnection thefirstmulticonnection(1, 1);
	thefirstmulticonnection.addConnection(0, 0, &thefirstsingleconnection);
	singleLayer thefirstsinglelayer(16*16,ACTIVATEFUNC::SIGMOID);
	multiLayer thefirstmultilayer(1,16*16);
	thefirstmultilayer.addSingleLayer(&thefirstsinglelayer);
	currentNet.addLayerAndConnection(&thefirstmultilayer, &thefirstmulticonnection);
	fullConnection  thesecondsingleconnection(16*16, 8*8);
	multiConnection thesecondmulticonnection(1, 1);
	thesecondmulticonnection.addConnection(0, 0, &thesecondsingleconnection);
	singleLayer thesecondsinglelayer(8*8, ACTIVATEFUNC::SIGMOID);
	multiLayer thesecondmultilayer(1, 8*8);
	thesecondmultilayer.addSingleLayer(&thesecondsinglelayer);
	currentNet.addLayerAndConnection(&thesecondmultilayer, &thesecondmulticonnection);
	fullConnection  thethirdsingleconnection(8*8, 10);
	multiConnection thethirdmulticonnection(1, 1);
	thethirdmulticonnection.addConnection(0, 0, &thethirdsingleconnection);
	singleLayer thethirdsinglelayer(10, ACTIVATEFUNC::SIGMOID);
	multiLayer thethirdmultilayer(1, 10);
	thethirdmultilayer.addSingleLayer(&thethirdsinglelayer);
	currentNet.addLayerAndConnection(&thethirdmultilayer, &thethirdmulticonnection);
	vector<double> lalallatest(32*32,1);
	currentNet.singleCaseOutput(lalallatest);
	vector<double> result{ 1,0,0,0,0,0,0,0,0,0 };
	for (int i = 0; i < 40; i++)
	{
		currentNet.singleCaseBackProp(result);
		currentNet.updateNetwork(0.8, 0.8);
		cout << "after backPropagate" << endl;
		currentNet.singleCaseOutput(lalallatest);
	}

}