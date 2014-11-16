#include "singleConnection.h"
//#include "accelerateFor.h"
class fullConnection :public singleConnection
{
public:
	vector<vector<double>> reverseWeight;//reverseWeight[i][j]=connectweight[j][i]
	fullConnection(int inDim, int outDim) :singleConnection(inDim, outDim), reverseWeight(outDim, vector<double>(inDim,0))
	{
		connectWeight = vector<vector<double>>(inDim, vector<double>(outDim, 0));
		weightGradient = vector<vector<double>>(inDim, vector<double>(outDim, 0));
		initWeight();
	}
	void addConnection(int fromIndex, int toIndex, double weight)
	{
		totalConnections++;
		connectWeight[fromIndex][toIndex] = weight;
		reverseWeight[toIndex][fromIndex] = weight;
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
	void forwardPropagate(const vector<vector<double>>& input, vector<vector<double>>& output)
	{
		accelerateFor(0, BATCH_SIZE, [&](int i)
		{
			for (int j = 0; j < outputDim; j++)
			{
				double propagateResult = 0;
				propagateResult = avx_product(input[i], reverseWeight[j]);
				output[i][j] += propagateResult;
			}
			
		});
	}
	void backPropagate(const vector<vector<double>>& nextLayerDelta, vector<vector<double>>& preLayerGradient,const vector<vector<double>>& preLayerOutput)
	{
		accelerateFor(0, BATCH_SIZE, [&](int i)
		{
			for (int j = 0; j <inputDim ; j++)
			{
				double propagateResult = 0;
				propagateResult = avx_product(nextLayerDelta[i], connectWeight[j]);//using avx
				preLayerGradient[i][j] += propagateResult;
			}
			
		});
		//the weightGradient
		accelerateFor(0, inputDim, [&](int i)
		{
			for (int j= 0; j < outputDim; j++)
			{
				for (int k = 0; k < BATCH_SIZE; k++)
				{
					weightGradient[i][j] += preLayerOutput[k][i] * nextLayerDelta[k][j];
				}
			}
		});
	}
	void updateWeight(double stepSize)
	{
		for(int i=0; i<inputDim;i++)
		{
			for (int j = 0; j < outputDim; j++)
			{
				connectWeight[i][j] -= stepSize*weightGradient[i][j];
				reverseWeight[j][i] = connectWeight[i][j];
				weightGradient[i][j] = 0;//clear the batchsum
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