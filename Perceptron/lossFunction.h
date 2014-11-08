#pragma once
#include <iostream>
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
typedef vector<double>(*diffFunctype)(const vector<double>& A, const vector<double>& B);
//diffrentiation is done for A not for B ,watch out


class lossFunc
{
private:
	LOSSFUNC currentLossType;
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
	vector<double> diffMse(const vector<double>& A, const vector<double>& B)//we can use avx
	{
		int sizeA, sizeB;
		sizeA = A.size();
		sizeB = B.size();
		vector<double> result(sizeA);
		assert(sizeA == sizeB);
		for (int i = 0; i < sizeA; i++)
		{
			result[i] = 2 * (A[i] - B[i]);
		}
		return result;
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
			double temp = B[i] * log(A[i]) + (1 - B[i])*log(1 - A[i]);
			result -= temp;
		}
		return result;
	}
	vector<double> diffCrossentrophy(const vector<double>& A, const vector<double>& B)
	{
		int sizeA, sizeB;
		sizeA = A.size();
		sizeB = B.size();
		vector<double> result(sizeA,0);
		assert(sizeA == sizeB);
		for (int i = 0; i < sizeA; i++)
		{
			//result[i] = B[i] / A[i] -(1 - B[i]) / (1 - A[i]);//beware there is something to nan 

		/*	double temp_1, temp_2, temp_3;
			temp_1 =1+ A[i];
			temp_2 = A[i]*A[i];
			temp_3 = (1+temp_2)*temp_1;
			temp_2 = temp_2*temp_2;
			temp_3 = temp_3*(1 + temp_2);
			result[i] = B[i] / A[i] - (1 - B[i] )* temp_3;*/
			result[i] = (A[i] - B[i]) / (A[i] * (1 - A[i]));
		}
		return result;
	}
public:
	lossFunc(LOSSFUNC inFuncType):currentLossType(inFuncType)
	{
	}
	double operator()(const vector<double>& trainResult,const vector<double>& realResult)
	{
		if (currentLossType == LOSSFUNC::MSE)
		{
			return evalMse(trainResult, realResult);
		}
		else
		{
			return evalCrossentrophy(trainResult, realResult);
		}
		
	}
	double eval(vector<double>& trainResult, vector<double> realResult)
	{
		if (currentLossType == LOSSFUNC::MSE)
		{
			return evalMse(trainResult, realResult);
		}
		else
		{
			return evalCrossentrophy(trainResult, realResult);
		}
	}
	vector<double> diff(vector<double>& trainResult,const vector<double> realResult)
	{
		if (currentLossType == LOSSFUNC::MSE)
		{
			return diffMse(trainResult, realResult);
		}
		else
		{
			return diffCrossentrophy(trainResult, realResult);
		}
	}
};