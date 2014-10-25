#include <vector>
#include <algorithm>
#include <assert.h>
using namespace std;
enum class LOSSFUNC
{
	MSE,
	CROSSENTROPHY
};

typedef float(*evalFunctype)(const vector<float>& A, const vector<float>& B);
typedef float(*diffFunctype)(const vector<float>& A, const vector<float>& B,int index);
//diffrentiation is done for A not for B ,watch out
float evalMse(const vector<float>& A, const vector<float>& B)
{
	int sizeA, sizeB;
	sizeA = A.size();
	sizeB = B.size();
	float result = 0;
	assert(sizeA == sizeB);
	for (int i = 0; i < sizeA; i++)
	{
		float temp = A[i] - B[i];
		result += temp*temp;
	}
	return result;
}
float diffMse(const vector<float>& A, const vector<float>& B, int index)
{
	int sizeA, sizeB;
	sizeA = A.size();
	sizeB = B.size();
	float result = 0;
	assert(sizeA == sizeB);
	assert(index >= 0 && index < sizeA);
	return 2*(A[index] - B[index]);
}
float evalCrossentrophy(const vector<float>& A, const vector<float>& B)
{
	int sizeA, sizeB;
	sizeA = A.size();
	sizeB = B.size();
	float result = 0;
	assert(sizeA == sizeB);
	for (int i = 0; i < sizeA; i++)
	{
		float temp = A[i] * log(B[i]) + (1 - A[i])*log(1 - B[i]);
		result += -temp;
	}
	return result;
}
float diffCrossentrophy(const vector<float>& A, const vector<float>& B, int index)
{
	int sizeA, sizeB;
	sizeA = A.size();
	sizeB = B.size();
	float result = 0;
	assert(sizeA == sizeB);
	assert(index >= 0 && index < sizeA);
	return  (A[index] - B[index])/(A[index]*(1-A[index]));
	
}

evalFunctype lossFunc[2] = { &evalMse, &evalCrossentrophy };
diffFunctype diffFunc[2] = { &diffMse, &diffCrossentrophy };
