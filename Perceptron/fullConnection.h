#include "singleConnection.h"
//#include "accelerateFor.h"
class fullConnection :public singleConnection
{
public:
	vector<vector<double>> reverseWeight;//reverseWeight[i][j]=connectweight[j][i]
	fullConnection(int inDim, int outDim) :singleConnection(inDim, outDim), reverseWeight(outDim, vector<double>(inDim,0))
	{
		initWeight();
	}
	void addConnection(int fromIndex, int toIndex, double weight)
	{
		totalConnections++;
		connectWeight[fromIndex][toIndex] = weight;
		reverseWeight[toIndex][fromIndex] = weight;
		isConnected[fromIndex][toIndex] = 1;
		weightFromInput[fromIndex].push_back(toIndex);
		weightToOutput[toIndex].push_back(fromIndex);
	}
	void initWeight()
	{
		std::default_random_engine dre(clock());
		std::uniform_real_distribution<double> di(-1.0, 1.0);
		
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				double tempWeight;
				tempWeight = di(dre);
				addConnection(i, j, tempWeight);				
			}
		}
	}
	void forwardPropagate(const vector<double>& input, vector<double>& output)
	{
		accelerateFor(0, outputDim,[&](int i)
		{
			double propagateResult = 0;
			propagateResult = avx_product(input, reverseWeight[i]);
			output[i] += propagateResult;
		});
	}
	void backPropagate(const vector<double>& nextLayerDelta, vector<double>& preLayerGradient,const vector<double>& preLayerOutput)
	{
		accelerateFor(0, inputDim, [&](int i)
		{
			double propagateResult = 0;
			propagateResult = avx_product(nextLayerDelta, connectWeight[i]);//using avx
			preLayerGradient[i] += propagateResult;
			for (int j = 0; j < outputDim; j++)
			{
				weightGradient[i][j] = preLayerOutput[i] * nextLayerDelta[j];
				batchWeightGradient[i][j] += weightGradient[i][j];
			}
		});
		//the weightGradient
	}
	void updateWeight(double stepSize)
	{
		for(int i=0; i<inputDim;i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				connectWeight[i][j] -= stepSize*weightGradient[i][j];
				batchWeightGradient[i][j] = 0;//clear the batchsum
			}
		}
	}
	void consoleWeightOutput()
	{
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				cout << connectWeight[i][j] << ' ';
			}
			cout << endl;
		}
		cout << "current full connection weight" << endl;
	}
	void fileWeightOutput(ofstream& outFile)
	{
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				outFile << connectWeight[i][j] << ' ';
			}
			outFile << endl;
		}
		outFile << "current full connection weight" << endl;
	}
	void loadWeightFromFile(ifstream& inputFile)
	{
		for (int i = 0; i < inputDim; i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				inputFile >> connectWeight[i][j];
			}
			inputFile.get();
			inputFile.get();//eat the new line
		}
	}
};