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
typedef vector<float> (*diffFunctype)(const vector<float>& A, const vector<float>& B);
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
vector<float> diffMse(const vector<float>& A, const vector<float>& B)//we can use avx
{
	int sizeA, sizeB;
	sizeA = A.size();
	sizeB = B.size();
	vector<float> result(sizeA);
	int index = 0;
	assert(sizeA == sizeB);
	for (int i = 0; i < sizeA; i++)
	{
		result[index] = 2 * (A[index] - B[index]);
	}
	return result;
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
vector<float>diffCrossentrophy(const vector<float>& A, const vector<float>& B)
{
	int sizeA, sizeB;
	sizeA = A.size();
	sizeB = B.size();
	vector<float> result(sizeA);
	int index = 0;
	assert(sizeA == sizeB);
	for (int i = 0; i < sizeA; i++)
	{
		result[index]=(A[index] - B[index]) / (A[index] * (1 - A[index]));
	}
	return result;
}

evalFunctype evalFunc[2] = { &evalMse, &evalCrossentrophy };
diffFunctype diffFunc[2] = { &diffMse, &diffCrossentrophy };
class lossFunc
{
private:
	evalFunctype currentEvalFunc;
	diffFunctype currentDiffFunc;
public:
	lossFunc(LOSSFUNC currentFuncType) :currentDiffFunc(diffFunc[(int) currentFuncType]), currentEvalFunc(evalFunc[(int) currentFuncType])
	{

	}
	float operator()(vector<float>& trainResult,vector<float> realResult)const
	{
		return (*currentEvalFunc)(trainResult,realResult);
	}
	float eval(vector<float>& trainResult, vector<float> realResult)const
	{
		return (*currentEvalFunc)(trainResult, realResult);
	}
	vector<float> diff(vector<float>& trainResult, vector<float> realResult)const
	{
		return (*currentDiffFunc)( trainResult, realResult);
	}
};