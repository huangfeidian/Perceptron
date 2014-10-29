#include "layer.h"
class multiLayer
{
public:
	vector<layer&> featureMaps;
	int featureMapNumber;
	multiLayer(int FeaMapNum, int inRow, int inCol, ACTIVATEFUNC currentFuncType)
	{
		for (int i = 0; i < FeaMapNum; i++)
		{
			layer tempLayer(inRow*inCol,currentFuncType);
			featureMaps.push_back(tempLayer);
		}
	}
};