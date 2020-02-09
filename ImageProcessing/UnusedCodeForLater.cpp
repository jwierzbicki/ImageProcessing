#include <inttypes.h>

typedef struct {
	uint8_t Blue;
	uint8_t Green;
	uint8_t Red;
} Pixel;

void SumAveragePixels(Pixel *input, Pixel *output, int row, int col, int imageHeight, int imageWidth, int color)
{
	uint16_t bottomRow = 0, middleRow = 0, topRow = 0;
	uint8_t blocksAdded = 0;

	// --- Bottom row
	if (row >= 1)
	{
		if (col >= 1)
		{
			// bottom left
			switch (color)
			{
			case 0: bottomRow += input[(row - 1) * imageWidth + (col - 1)].Blue; break;
			case 1: bottomRow += input[(row - 1) * imageWidth + (col - 1)].Green; break;
			case 2: bottomRow += input[(row - 1) * imageWidth + (col - 1)].Red; break;
			}
			blocksAdded++;
		}

		// bottom center
		switch (color)
		{
		case 0:	bottomRow += input[(row - 1) * imageWidth + col].Blue; break;
		case 1:	bottomRow += input[(row - 1) * imageWidth + col].Green; break;
		case 2:	bottomRow += input[(row - 1) * imageWidth + col].Red; break;
		}
		blocksAdded++;

		if (col <= imageWidth - 2)
		{
			// bottom right
			switch (color)
			{
			case 0:	bottomRow += input[(row - 1) * imageWidth + (col + 1)].Blue; break;
			case 1: bottomRow += input[(row - 1) * imageWidth + (col + 1)].Green; break;
			case 2: bottomRow += input[(row - 1) * imageWidth + (col + 1)].Red; break;
			}
			blocksAdded++;
		}
	}

	// --- Middle row
	 // middle center
	switch (color)
	{
	case 0: middleRow += input[row * imageWidth + col].Blue; break;
	case 1: middleRow += input[row * imageWidth + col].Green; break;
	case 2: middleRow += input[row * imageWidth + col].Red;	break;
	}
	blocksAdded++;

	if (col >= 1)
	{
		// middle left
		switch (color)
		{
		case 0: middleRow += input[row * imageWidth + (col - 1)].Blue; break;
		case 1: middleRow += input[row * imageWidth + (col - 1)].Green; break;
		case 2: middleRow += input[row * imageWidth + (col - 1)].Red; break;
		}
		blocksAdded++;
	}

	if (col <= imageWidth - 2)
	{
		// middle right
		switch (color)
		{
		case 0: middleRow += input[row * imageWidth + (col + 1)].Blue; break;
		case 1: middleRow += input[row * imageWidth + (col + 1)].Green; break;
		case 2: middleRow += input[row * imageWidth + (col + 1)].Red; break;
		}
		blocksAdded++;
	}

	// --- Top row
	if (row <= imageHeight - 2)
	{
		if (col >= 1)
		{
			// top left
			switch (color)
			{
			case 0: topRow += input[(row + 1) * imageWidth + (col - 1)].Blue; break;
			case 1: topRow += input[(row + 1) * imageWidth + (col - 1)].Green; break;
			case 2: topRow += input[(row + 1) * imageWidth + (col - 1)].Red; break;
			}
			blocksAdded++;
		}

		// top center
		switch (color)
		{
		case 0: topRow += input[(row + 1) * imageWidth + col].Blue; break;
		case 1: topRow += input[(row + 1) * imageWidth + col].Green; break;
		case 2: topRow += input[(row + 1) * imageWidth + col].Red; break;
		}
		blocksAdded++;

		if (col <= imageWidth - 2)
		{
			// top right
			switch (color)
			{
			case 0: topRow += input[(row + 1) * imageWidth + (col + 1)].Blue; break;
			case 1: topRow += input[(row + 1) * imageWidth + (col + 1)].Green; break;
			case 2: topRow += input[(row + 1) * imageWidth + (col + 1)].Red; break;
			}
			blocksAdded++;
		}
	}

	switch (color)
	{
	case 0: output[row * imageWidth + col].Blue = (bottomRow + middleRow + topRow) / blocksAdded; break;
	case 1: output[row * imageWidth + col].Green = (bottomRow + middleRow + topRow) / blocksAdded; break;
	case 2: output[row * imageWidth + col].Red = (bottomRow + middleRow + topRow) / blocksAdded; break;
	}
}