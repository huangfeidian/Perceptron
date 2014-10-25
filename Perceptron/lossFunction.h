#include <vector>
#include <algorithm>
#include <assert.h>
using namespace std;
enum class LOSSFUNC
{
	MSE,
	CROSSENTROPHY
};

typedef double(*evalFunctype)(const vector<double>& A, const vector<double>& B);
typedef double(*diffFunctype)(const vector<double>& A, const vector<double>& B,int index);
//diffrentiation is done for A not for B ,watch out
double evalMse(const vector<double>& A, const vector<double>& B)
{
	int sizeA, sizeB;
	sizeA = A.size();
	sizeB = B.size();
	double result = 0;
	assert(sizeA == sizeB);
	for (int i = 0; i < sizeA; i++)
	{
		double temp = A[i] - B[i];
		result += temp*temp;
	}
	return result;
}
double diffMse(const vector<double>& A, const vector<double>& B, int index)
{
	int sizeA, sizeB;
	sizeA = A.size();
	sizeB = B.size();
	double result = 0;
	assert(sizeA == sizeB);
	assert(index >= 0 && index < sizeA);
	return 2*(A[index] - B[index]);
}
double evalCrossentrophy(const vector<double>& A, const vector<double>& B)
{
	int sizeA, sizeB;
	sizeA = A.size();
	sizeB = B.size();
	double result = 0;
	assert(sizeA == sizeB);
	for (int i = 0; i < sizeA; i++)
	{
		double temp = A[i] * log(B[i]) + (1 - A[i])*log(1 - B[i]);
		result += -temp;
	}
	return result;
}
double diffCrossentrophy(const vector<double>& A, const vector<double>& B, int index)
{
	int sizeA, sizeB;
	sizeA = A.size();
	sizeB = B.size();
	double result = 0;
	assert(sizeA == sizeB);
	assert(index >= 0 && index < sizeA);
	return  (A[index] - B[index])/(A[index]*(1-A[index]));
	
}

evalFunctype lossFunc[2] = { &evalMse, &evalCrossentrophy };
diffFunctype diffFunc[2] = { &diffMse, &diffCrossentrophy };
