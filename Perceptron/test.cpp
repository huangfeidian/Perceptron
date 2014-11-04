
#include <iostream>
#include "network.h"
#include "fullConnection.h"
using namespace std;
int main()
{
	network currentNet(4, LOSSFUNC::MSE);
	fullConnection  hehe(4, 1);
	singleLayer out(1, ACTIVATEFUNC::SIGMOID);
}