#include <vector>
#include "activateFunction.h"
#include <map>
template<int Dimension,ACTIVATEFUNC ActivateFuncIndex>class layer
{
public:
	double inputValue[Dimension];
	double outputValue[Dimension];
	double bias[Dimension];
	activateFunc<ActivateFuncIndex> currentActiFunc;
	const int dim = Dimension;
};
template<int InDim, int outDim>class connection
{
public:
	double connectWeight[InDim][outDim];//the weights of connections between layers,currently i don't care the sparse before this demo works
	const int inputDim = InDim;
	const int outputDim = outDim;

};
class layers
{
private:
	int layerNum;
	int inputDim;
	int outputDim;
public:
	template<int,ACTIVATEFUNC> layer allLayers;
	template<int, int> connection allConnections;
	
};