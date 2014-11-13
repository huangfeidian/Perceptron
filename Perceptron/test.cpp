
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
	ofstream weightOutFile("weight.txt");
	ofstream biasOutFile("bias.txt");
	ofstream trainResult("trainResult.txt");
	end = clock();
	cout << (end - begin)  << " ms passed in reading the file" << endl;
	begin = clock();
	network currentNet(32*32,10, LOSSFUNC::MSE);
	vector<vector<bool>> theFirstConvolutionConnection = { { O, O, O, O, O, O } };
	vector<vector<bool>> theSecondConvolutionConnection = {
			{O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O},
			{O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O},
			{O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O},
			{X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O},
			{X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O},
			{X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O}
	};
	currentNet.addConvolutionLayerAndConnection(32, 5, theFirstConvolutionConnection, ACTIVATEFUNC::SIGMOID);
	currentNet.addPoolLayerAndConnection(28, 2, ACTIVATEFUNC::SIGMOID);
	currentNet.addConvolutionLayerAndConnection(14, 5, theSecondConvolutionConnection, ACTIVATEFUNC::SIGMOID);
	currentNet.addPoolLayerAndConnection(10, 2, ACTIVATEFUNC::SIGMOID);
	currentNet.addFullLayerAndConnection(120, ACTIVATEFUNC::SIGMOID);
	currentNet.addFullLayerAndConnection(84, ACTIVATEFUNC::SIGMOID);
	currentNet.addFullLayerAndConnection(10, ACTIVATEFUNC::SIGMOID);
	int trainCaseNumber=trainLabels.size();
	int testCaseNumber = testLabels.size();
	end = clock();
	cout << (end - begin) << " ms passed in construct the network" << endl;
	begin = clock();
	double totalPrecision = 0;
	int rightResults = 0;
	double step = 0.1;
	bool nantest = false;
	for (int i = 0; i < 10; i++)
	{
		trainResult << std::setw(13) << trainLabels[1][i]<< ' ';
	}
	trainResult << endl;
	for (int k = 0; k < 1; k++)
	{
	
		step = step*0.9;
		rightResults = 0;
		for (int i = 0; i < trainCaseNumber; i++)
		{
			currentNet.singleCaseOutput(trainImages[i]);
			currentNet.singleCaseBackProp(trainLabels[i]);
			if (i % 8 == 0)
			{
				currentNet.updateNetwork(step, step);
			}	
		}
	}
	//int i = 0;
	//for (int j = 0; j < currentNet.outputDim; j++)
	//{
	//	trainResult << setw(13) << trainLabels[0][j] << ' ';
	//}
	//trainResult << endl;
	//while (i<1000)
	//{
	//	for (int j = 0; j < currentNet.outputDim; j++)
	//	{
	//		trainResult << setw(13)<<currentNet.output[j] << ' ';
	//	}
	//	trainResult << endl;
	//	currentNet.singleCaseBackProp(trainLabels[0]);
	//	currentNet.updateNetwork(step,step);
	//	i++;
	//	currentNet.singleCaseOutput(trainImages[0]);
	//}
	//cout << "after " << i << "rounds" << endl;
	//currentNet.fileNetworkOutput(weightOutFile);


	//for (int i = 0; i < testCaseNumber; i++)
	//{
	//	currentNet.singleCaseOutput(testImages[i]);
	//	for (int j = 0; j <10; j++)
	//	{
	//		trainResult <<std::setw(13)<<currentNet.output[j] << ' ';
	//	}
	//	trainResult << endl;
	//	if (testLabels[i][maxOut(currentNet.output)] > 0.5)
	//	{
	//		rightResults++;
	//	}
	//}
	//cout << " the whole test precision is " << (1.0*rightResults) / testCaseNumber << endl;
	end = clock();
	cout << (end - begin) << " ms passed in training" << endl;

}