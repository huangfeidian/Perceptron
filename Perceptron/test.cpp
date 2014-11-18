
#include "network.h"
#include "mnistParser.h"
#include <ctime>
#include <cmath>
#include <iomanip>
#include <boost\timer.hpp>
#include <boost\progress.hpp>

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
	ifstream weightInputFile("preWeight.txt");
	ofstream weightOutputFile("afterWeight.txt");
	ofstream biasOutFile("bias.txt");
	ofstream trainResult("trainResult.txt");
	ifstream tinycnn("LeNet-weights");
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
	currentNet.addConvolutionLayerAndConnection(32, 5, theFirstConvolutionConnection, ACTIVATEFUNC::TANH);
	currentNet.addPoolLayerAndConnection(28, 2, ACTIVATEFUNC::TANH);
	currentNet.addConvolutionLayerAndConnection(14, 5, theSecondConvolutionConnection, ACTIVATEFUNC::TANH);
	currentNet.addPoolLayerAndConnection(10, 2, ACTIVATEFUNC::TANH);
	currentNet.addFullLayerAndConnection(120, ACTIVATEFUNC::TANH);
	currentNet.addFullLayerAndConnection(80, ACTIVATEFUNC::TANH);
	currentNet.addFullLayerAndConnection(10, ACTIVATEFUNC::TANH);
	currentNet.loadNetworkFromFile(weightInputFile);
	int trainCaseNumber=trainLabels.size();
	int testCaseNumber = testLabels.size();
	end = clock();
	cout << (end - begin) << " ms passed in construct the network" << endl;
	begin = clock();
	//load the bias and weight from tiny cnn
	
	//load end
	int rightResults = 0;
	double step = 0.005;
	//for (int i = 0; i < 10; i++)
	//{
	//	trainResult << std::setw(13) << trainLabels[1][i]<< ' ';
	//}
	//trainResult << endl;
	boost::progress_display currentProgress(trainCaseNumber);
	boost::timer timeElapsed;
	for (int k = 0; k <1; k++)
	{
		
		step = step*0.9;
		rightResults = 0;
		for (int i = 0; i < trainCaseNumber/BATCH_SIZE; i++)
		{
			currentNet.singleCaseOutput(trainImages, BATCH_SIZE*i);
			currentNet.singleCaseBackProp(trainLabels, BATCH_SIZE*i);
			currentNet.updateNetwork(step, step);
			currentProgress += BATCH_SIZE;
		}
		cout << timeElapsed.elapsed() << " s elapsed." << endl;
		for (int i = 0; i < testCaseNumber / BATCH_SIZE; i++)
		{
			currentNet.singleCaseOutput(testImages, BATCH_SIZE*i);
			/*for (int j = 0; j <10; j++)
			{
				trainResult << std::setw(13) << currentNet.output[j] << ' ';
			}*/
			//trainResult << endl;
			for (int j = 0; j < BATCH_SIZE; j++)
			{
				if (testLabels[BATCH_SIZE*i + j][maxOut(currentNet.output[j])] > 0.5)
				{
					rightResults++;
				}
			}
			
		}
		cout << " the whole test precision is " << (1.0*rightResults) / testCaseNumber << endl;
		timeElapsed.restart();
		currentProgress.restart(trainCaseNumber);
	}
	
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
	//currentNet.singleCaseOutput(testImages[0]);
	//ofstream testcaseone("testcase1.txt");
	//auto& tempMap = currentNet.allLayers[1]->featureMaps;
	//for (int j = 0; j < 28; j++)
	//{
	//	for (int i = 0; i < 28; i++)
	//	{
	//		testcaseone << setw(13) << tempMap[0]->outputValue[28*i+j] << ' ';
	//	}
	//	testcaseone << endl;
	//}
	//
	//testcaseone.close();
	cout << " the whole test precision is " << (1.0*rightResults) / testCaseNumber << endl;
	end = clock();
	cout << (end - begin) << " ms passed in training" << endl;
	currentNet.fileNetworkOutput(weightOutputFile);
}