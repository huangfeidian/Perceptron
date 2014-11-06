#include <vector>
#include <list>
using std::vector;
using std::list;

class input
{
public:
	int totalCases;//how many input we have
	const int inputDimension;//for a single case input the lenght of input vector
	const int outputDimension;//the length of desired output vector;
	vector<vector<double>> totalOutput;
	vector<vector<double>> totalInput;
	input(int inDim, int outDim) :inputDimension(inDim), outputDimension(outDim)
	{
		totalCases = 0;
	}
	void addCase(vector<double>&& singleInputCase,vector<double>&& singleOutputCase)
	{
		totalCases++;
		totalInput.push_back(singleInputCase);
		totalOutput.push_back(singleOutputCase);
	}
	
};