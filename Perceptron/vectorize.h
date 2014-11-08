#include <immintrin.h>
#include <vector>
#include <assert.h>
#include <numeric>
#include "config.h"
using std::vector;
#ifdef USE_AVX
double  avx_product(const vector<double>& origin1,const vector<double>& origin2)//2 vector product
{
	int size1 = origin1.size();
	int size2 = origin2.size();
	assert(size1 == size2);
	int blocks = size1 / 4;//because 256bit =8*sizeof(double)
	__m256d result256,temp1,temp2;
	double result=0;
	result256=_mm256_setzero_pd();
	for (int i = 0; i < blocks; i++)
	{
		temp1 = _mm256_loadu_pd(&(origin1[0])+i*4);
		temp2 = _mm256_loadu_pd(&(origin2[0]) + i * 4);
		result256 = _mm256_add_pd(result256, _mm256_mul_pd(temp1, temp2));
	}
	const double* temp = (const double*) &result256;
	for(int i=0;i<4;i++)
	{
		result+=temp[i];
	}
	for (int i = (size1 >> 2) << 2; i < size1; i++)
	{
		result += origin1[i] * origin2[i];
	}
	return result;
	
}
double avx_product(const vector<double>& origin1, const vector<double>& origin2, const vector<double>& origin3)//3 vector product
{
	int size1 = origin1.size();
	int size2 = origin2.size();
	int size3 = origin3.size();
	assert(size1 == size2);
	assert(size1 == size3);
	int blocks = size1 / 4;//because 256bit =8*sizeof(double)
	__m256d result256, temp1, temp2,temp3;
	double result=0;
	result256 = _mm256_setzero_pd();
	for (int i = 0; i < blocks; i++)//maybe we can align these vector or just replace these vector with struct 
	{
		temp1 = _mm256_loadu_pd(&(origin1[0]) + i * 4);
		temp2 = _mm256_loadu_pd(&(origin2[0]) + i * 4);
		temp3 = _mm256_loadu_pd(&(origin3[0]) + i * 4);
		result256 = _mm256_add_pd(result256, _mm256_mul_pd(_mm256_mul_pd(temp1, temp2),temp3));
	}
	const double* temp = (const double*) &result256;
	for(int i=0;i<4;i++)
	{
		result+=temp[i];
	}
	for (int i = (size1 >> 2) << 2; i < size1; i++)
	{
		result += origin1[i] * origin2[i]*origin3[i];
	}
	return result;
}
double avx_product(const vector<double>& origin1, const vector<double>& origin2, const vector<double>& origin3, const vector<double>& origin4)//4 vector product
{
	int size1 = origin1.size();
	int size2 = origin2.size();
	int size3 = origin3.size();
	int size4 = origin4.size();

	assert(size1 == size2);
	assert(size1 == size4);
	assert(size1 == size3);
	int blocks = size1 / 4;//because 256bit =8*sizeof(double)
	__m256d result256, temp1, temp2, temp3,temp4;
	double result=0;
	result256 = _mm256_setzero_pd();
	for (int i = 0; i < blocks; i++)//maybe we can align these vector or just replace these vector with struct 
	{
		temp1 = _mm256_loadu_pd(&(origin1[0]) + i * 4);
		temp2 = _mm256_loadu_pd(&(origin2[0]) + i * 4);
		temp3 = _mm256_loadu_pd(&(origin3[0]) + i * 4);
		temp4 = _mm256_loadu_pd(&(origin4[0]) + i * 4);

		result256 = _mm256_add_pd(result256, _mm256_mul_pd(_mm256_mul_pd(temp1, temp2), _mm256_mul_pd(temp3,temp4)));
	}
	const double* temp = (const double*) &result256;
	for(int i=0;i<4;i++)
	{
		result+=temp[i];
	}
	for (int i = (size1 >> 2) << 2; i < size1; i++)
	{
		result += origin1[i] * origin2[i] * origin3[i]*origin4[i];
	}
	return result;
}
#else
double  avx_product(const vector<double>& origin1,const vector<double>& origin2)
{
	double result=0;
	int size1 = origin1.size();
	int size2 = origin2.size();
#ifdef CHECK_LEGITIMATE
	assert(size1 == size2);
#endif
	for(int i=0;i<size1;i++)
	{
		result += origin1[i] * origin2[i];
	}
	return result;
}
#endif

