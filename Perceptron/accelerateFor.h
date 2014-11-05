//this head file encapsulate  plantform dependent parallel library
#include "config.h"
#include <functional>
#pragma once
#ifdef WINDOWS_PPL
#include <ppl.h>
using namespace concurrency;
void accelerateFor(int beginIndex, int endIndex, std::function<void(int)> f)
{
	parallel_for(beginIndex, endIndex, f);
}
#else
void accelerateFor(int beginIndex,int endIndex,std::function<void(int)> f)
{
	for(int i=beginIndex;i<endIndex;i++)
	{
		f(i);
	}
}
#endif
