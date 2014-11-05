#include <immintrin.h>
#include <vector>
#include <assert.h>
#include <numeric>
#include "config.h"
using std::vector;
#ifdef USE_AVX
float  avx_product(const vector<float>& origin1,const vector<float>& origin2)//2 vector product
{
	int size1 = origin1.size();
	int size2 = origin2.size();
	assert(size1 == size2);
	int blocks = size1 / 8;//because 256bit =8*sizeof(float)
	__m256 result256,temp1,temp2;
	float result=0;
	result256=_mm256_setzero_ps();
	for (int i = 0; i < blocks; i++)
	{
		temp1 = _mm256_loadu_ps(&(origin1[0])+i*8);
		temp2 = _mm256_loadu_ps(&(origin2[0]) + i * 8);
		result256 = _mm256_add_ps(result256, _mm256_mul_ps(temp1, temp2));
	}
	const float* temp = (const float*) &result256;
	for(int i=0;i<8;i++)
	{
		result+=temp[i];
	}
	for (int i = (size1 >> 3) << 3; i < size1; i++)
	{
		result += origin1[i] * origin2[i];
	}
	return result;
	
}
float avx_product(const vector<float>& origin1, const vector<float>& origin2, const vector<float>& origin3)//3 vector product
{
	int size1 = origin1.size();
	int size2 = origin2.size();
	int size3 = origin3.size();
	assert(size1 == size2);
	assert(size1 == size3);
	int blocks = size1 / 8;//because 256bit =8*sizeof(float)
	__m256 result256, temp1, temp2,temp3;
	float result=0;
	result256 = _mm256_setzero_ps();
	for (int i = 0; i < blocks; i++)//maybe we can align these vector or just replace these vector with struct 
	{
		temp1 = _mm256_loadu_ps(&(origin1[0]) + i * 8);
		temp2 = _mm256_loadu_ps(&(origin2[0]) + i * 8);
		temp3 = _mm256_loadu_ps(&(origin3[0]) + i * 8);
		result256 = _mm256_add_ps(result256, _mm256_mul_ps(_mm256_mul_ps(temp1, temp2),temp3));
	}
	const float* temp = (const float*) &result256;
	for(int i=0;i<8;i++)
	{
		result+=temp[i];
	}
	for (int i = (size1 >> 3) << 3; i < size1; i++)
	{
		result += origin1[i] * origin2[i]*origin3[i];
	}
	return result;
}
float avx_product(const vector<float>& origin1, const vector<float>& origin2, const vector<float>& origin3, const vector<float>& origin4)//4 vector product
{
	int size1 = origin1.size();
	int size2 = origin2.size();
	int size3 = origin3.size();
	int size4 = origin4.size();

	assert(size1 == size2);
	assert(size1 == size4);
	assert(size1 == size3);
	int blocks = size1 / 8;//because 256bit =8*sizeof(float)
	__m256 result256, temp1, temp2, temp3,temp4;
	float result=0;
	result256 = _mm256_setzero_ps();
	for (int i = 0; i < blocks; i++)//maybe we can align these vector or just replace these vector with struct 
	{
		temp1 = _mm256_loadu_ps(&(origin1[0]) + i * 8);
		temp2 = _mm256_loadu_ps(&(origin2[0]) + i * 8);
		temp3 = _mm256_loadu_ps(&(origin3[0]) + i * 8);
		temp4 = _mm256_loadu_ps(&(origin4[0]) + i * 8);

		result256 = _mm256_add_ps(result256, _mm256_mul_ps(_mm256_mul_ps(temp1, temp2), _mm256_mul_ps(temp3,temp4)));
	}
	const float* temp = (const float*) &result256;
	for(int i=0;i<8;i++)
	{
		result+=temp[i];
	}
	for (int i = (size1 >> 3) << 3; i < size1; i++)
	{
		result += origin1[i] * origin2[i] * origin3[i]*origin4[i];
	}
	return result;
}
#else
float  avx_product(const vector<float>& origin1,const vector<float>& origin2)
{
	float result=0;
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

