#include <vector>
#include "connection.h"
#include <ppl.h>
using namespace concurrency;
using namespace std;
class convolution :public connection
{
public:
	const int inDimRow;
	const int inDimColumn;
	const int outDimRow;
	const int outDimColumn;
	const int windowRow;
	const int windowColumn;
	convolution(int inRow,int inColumn,  int window) :connection(inRow*inColumn, (inColumn- window+1)*(inRow-window+1)), windowRow(window)
		, windowColumn(window), inDimColumn(inColumn), inDimRow(inDimRow), outDimColumn(inColumn + 1 - window), outDimRow(inRow+1-window)
		//watchout you must ensure inDimRow>=window and inDimColumn>=window
	{
		setConnected();
	}
	convolution(int inRow, int inColumn, int windowRow,int windowColumn) :connection(inRow*inColumn, (inColumn - windowColumn + 1)*(inRow - windowRow + 1)), windowRow(windowRow)
		, windowColumn(windowColumn), inDimColumn(inColumn), inDimRow(inDimRow), outDimColumn(inColumn + 1 - windowColumn), outDimRow(inRow + 1 - windowRow)
		//watchout you must ensure inRow>=windowRow and inColumn>=windowColumn
	{
		setConnected();
	}
	void setConnected()
	{
		std::default_random_engine dre;
		std::uniform_real_distribution<float> di(-1.0, 1.0);
		for (int i = 0; i < inDimRow-windowRow; i++)
		{
			for (int j = 0; j < inDimColumn-windowColumn; j++)
			{
				int begink = i - windowRow + 1 > 0 ? i - windowRow + 1 : 0;
				int endk = i + windowRow - 1 > inDimRow ? inDimRow : i + windowRow - 1;
				int beginl = j - windowColumn + 1 > 0 ? j - windowColumn + 1 : 0;
				int endl = j + windowColumn - 1 > inDimColumn ? inDimColumn : j + windowColumn - 1;
				for (begink; begink <= endk; begink++)
				{
					for (beginl; beginl < endl; beginl++)
					{
						addConnection(i*inDimColumn + j, begink*(inDimColumn - windowColumn + 1)+beginl,di(dre));
					}
				}
			}
		}
	}

};