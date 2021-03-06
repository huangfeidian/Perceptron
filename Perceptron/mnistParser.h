#include "config.h"
#include <fstream>
#include <cstdint>
#include <boost/detail/endian.hpp>
#include <vector>
#ifdef CHECK_LEGITIMATE
#include <assert.h>
#endif
//前面的两个有何意义？ 现在所有的平台都默认int是4个字节 double是8个字节 
template<typename T> T* reverse_endian(T* p) 
{
	std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) +sizeof(T));//改变大小端？？？？？ 有什么意义么
	//上次看到改变大小端的时候还是在MD5里面
	return p;
}
void parse_mnist_labels(const std::string& label_file, std::vector<vector<double>>& labels)
{
	std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);//基本就没有文本格式来打开文件 

#ifdef CHECK_LEGITIMATE
	assert((ifs.bad() || ifs.fail())==false);
#endif

	unsigned int magic_number, num_items;
	ifs.read((char*) &magic_number, 4);
	ifs.read((char*) &num_items, 4);

#if defined(BOOST_LITTLE_ENDIAN)
	//这个宏挺好玩的啊 话说这里为什么一定要转换为大端啊
	reverse_endian(&magic_number);
	reverse_endian(&num_items);
#endif

#ifdef CHECK_LEGITIMATE
	assert((magic_number == 0x00000801 &&num_items > 0));
#endif
	//oh I see
	vector<double> temp{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	for (unsigned int i = 0; i < num_items; i++)
	{
		char label;
		ifs.read((char*) &label, 1);
		temp[label] = 1;
		labels.push_back(temp);
		temp[label] = 0;
	}
}

struct mnist_header
{
	unsigned int  magic_number;
	unsigned int  num_items;
	unsigned int num_rows;
	unsigned int num_cols;
};

void parse_mnist_header(std::ifstream& ifs, mnist_header& header) 
{
	ifs.read((char*) &(header.magic_number), 4);
	ifs.read((char*) &(header.num_items), 4);
	ifs.read((char*) &(header.num_rows), 4);
	ifs.read((char*) &(header.num_cols), 4);
#if defined(BOOST_LITTLE_ENDIAN)
	reverse_endian(&(header.magic_number));
	reverse_endian(&(header.num_items));
	reverse_endian(&(header.num_rows));
	reverse_endian(&(header.num_cols));
#endif

#ifdef CHECK_LEGITIMATE
	assert((header.magic_number == 0x00000803 && header.num_items > 0));
	assert((ifs.fail() || ifs.bad())==false);
#endif
	
	
}

void parse_mnist_image(std::ifstream& ifs,
	const mnist_header& header,
	double scale_min,
	double scale_max,
	int x_padding,
	int y_padding,
	vector<double>& dst) 
{
	const int width = header.num_cols + 2 * x_padding;
	const int height = header.num_rows + 2 * y_padding;

	std::vector<unsigned char> image_vec(header.num_rows * header.num_cols);

	ifs.read((char*) &image_vec[0], header.num_rows * header.num_cols);

	dst.resize(width * height, scale_min);

	for (unsigned int y = 0; y < header.num_rows; y++)
	{
		for (unsigned int x = 0; x < header.num_cols; x++)
		{
			dst[width * (y + y_padding) + x + x_padding]= (image_vec[y * header.num_cols + x] / 255.0) * (scale_max - scale_min) + scale_min;
		}
	}
		
}

void parse_mnist_images(const std::string& image_file,
						std::vector<std::vector<double>>&images,
						double scale_min = -1.0,
						double scale_max = 1.0,
						int x_padding = 2,
						int y_padding = 2) 
{
	std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

#ifdef CHECK_LEGITIMATE
	assert((ifs.bad() || ifs.fail()) == false);
#endif

	mnist_header header;

	parse_mnist_header(ifs, header);

	for (unsigned int  i = 0; i < header.num_items; i++)
	{
		std::vector<double> image;
		parse_mnist_image(ifs, header, scale_min, scale_max, x_padding, y_padding, image);
		images.push_back(image);
	}
}