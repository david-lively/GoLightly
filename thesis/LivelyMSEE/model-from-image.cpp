void ModelProvider::FromImage(
FieldDescriptor &media
 ,unsigned char *bytes
 ,const unsigned int width
 ,const unsigned int height
 ,const unsigned int channels
 ,const float n
 ,vector<unsigned int> &sourceOffsets
 ,const unsigned int pmlLayers
 )
{
	for(int j = 0; j < media.Size.y; j++)
	{
		int sourceY = j * height / media.Size.y;
		for(int i = 0; i < media.Size.x; i++)
		{
			int sourceX = i * width / media.Size.x;
			unsigned int sourceOffset = channels * (sourceY * width + sourceX);
			unsigned int mediaOffset = j * media.Size.x + i;
			unsigned char red =   bytes[sourceOffset + 0];
			unsigned char green = bytes[sourceOffset + 1];
			unsigned char blue =  bytes[sourceOffset + 2];
			// is this pixel part of a source?
			if (red > 128)
				sourceOffsets.push_back(mediaOffset);

			/// fill default waveguide material (parameter n)
			if (green > 0)
			{
				// interpolate n based on green value.
				media.HostArray[mediaOffset] = n * green * 1.f / 255;
			}
		}
	}

	if (sourceOffsets.size() == 0)
	{
		cerr << "No sources found\n";
		exit(EXIT_FAILURE);
	}
}