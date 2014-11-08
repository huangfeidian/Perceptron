
#include "network.h"
#include "mnistParser.h"
#include <ctime>
#include <cmath>
#include <iomanip>
using namespace std;
#define O true
#define X false
int maxOut(vector<double>& result)
{
	double max = 0;
	int maxIndex = 0;
	int size = result.size();
	for (int i = 0; i < size; i++)
	{
		if (result[i]>max)
		{
			maxIndex = i;
			max = result[i];
		}
	}
	return maxIndex;
}
int main()
{
	clock_t begin, end;
	vector<vector<double>> trainImages, testImages, trainLabels, testLabels;
	begin = clock();
	parse_mnist_labels("train-labels.idx1-ubyte", trainLabels);
	parse_mnist_images("train-images.idx3-ubyte", trainImages);
	parse_mnist_labels("t10k-labels.idx1-ubyte", testLabels);
	parse_mnist_images("t10k-images.idx3-ubyte", testImages);
	ofstream resultOutFile("hehehehehhe.txt", ios::binary);
	end = clock();
	cout << (end - begin)  << " ms passed in reading the file" << endl;
	begin = clock();
	network currentNet(32*32,10, LOSSFUNC::MSE);
	//vector<vector<bool>> theFirstConvolutionConnection = { { O, O, O, O, O, O } };
	//vector<vector<bool>> theSecondConvolutionConnection = {
	//		{O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O},
	//		{O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O},
	//		{O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O},
	//		{X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O},
	//		{X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O},
	//		{X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O}
	//};
	//currentNet.addConvolutionLayerAndConnection(32, 5, theFirstConvolutionConnection,ACTIVATEFUNC::SIGMOID);
	//currentNet.addPoolLayerAndConnection(28, 2);
	//currentNet.addConvolutionLayerAndConnection(14, 5, theSecondConvolutionConnection, ACTIVATEFUNC::SIGMOID);
	//currentNet.addPoolLayerAndConnection(10, 2);
	//currentNet.addFullLayerAndConnection(120, ACTIVATEFUNC::SIGMOID);
	//currentNet.addFullLayerAndConnection(84, ACTIVATEFUNC::SIGMOID);
	currentNet.addFullLayerAndConnection(100, ACTIVATEFUNC::TANH);
	currentNet.addFullLayerAndConnection(10, ACTIVATEFUNC::TANH);
	int trainCaseNumber=trainLabels.size();
	int testCaseNumber = testLabels.size();
	end = clock();
	cout << (end - begin) << " ms passed in construct the network" << endl;
	begin = clock();
	double totalPrecision = 0;
	double step = 2.0;
	bool nantest = false;
	//for (int j = 0; j <10; j++)
	//{
	//	resultOutFile << std::setw(13) << trainLabels[0][j] << ' ';
	//}
	//resultOutFile<< endl;
	for (int k= 0; k <10; k++)
	{
		for (int i = 0; i < trainCaseNumber; i++)
		{
			currentNet.singleCaseOutput(trainImages[i]);
			//for (int j = 0; j <10; j++)
			//{
			//	resultOutFile <<std::setw(13)<<currentNet.output[j] << ' ';
			//}
			////nantest=isnan(currentNet.output[0]);
			//resultOutFile<< endl;
			currentNet.singleCaseBackProp(trainLabels[i]);
			currentNet.updateNetwork(step, step);
			
		}
		step = step*0.85;
		//there is 
		int rightResults = 0;
		for (int i = 0; i < trainCaseNumber; i++)
		{
			currentNet.singleCaseOutput(trainImages[i]);
			if (trainLabels[i][maxOut(currentNet.output)] > 0.5)
			{
				rightResults++;
			}
		}
		cout << " the whole precision is " << (1.0*rightResults) / trainCaseNumber << endl;
	}
	for (int i = 0; i < trainCaseNumber; i++)
	{
		currentNet.singleCaseOutput(trainImages[i]);
		/*for (int j = 0; j <10; j++)
		{
			resultOutFile << std::setw(13) << trainLabels[i][j] << ' ';
		}*/
		//nantest=isnan(currentNet.output[0]);
		resultOutFile << endl;
		for (int j = 0; j <10; j++)
		{
			resultOutFile <<std::setw(13)<<currentNet.output[j] << ' ';
		}
		//nantest=isnan(currentNet.output[0]);
		resultOutFile<< endl;
	}
	end = clock();
	cout << (end - begin) << " ms passed in training" << endl;

}