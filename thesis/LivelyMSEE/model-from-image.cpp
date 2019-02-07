for(int j = 0; j < media.Size.y; ++j)
{
	int sourceY = j * height / media.Size.y;
	for(int i = 0; i < media.Size.x; ++i)
	{
		int sourceX = i * width / media.Size.x;
		unsigned int sourceOffset = colorChannels * (sourceY * width + sourceX);
		unsigned int mediaOffset = j * media.Size.x + i;
		unsigned char sourceID =   imageBytes[sourceOffset + 0]; // red
		unsigned char epsilonR = imageBytes[sourceOffset + 1]; // green
		unsigned char monitorID =  imageBytes[sourceOffset + 2]; // blue
		// is this pixel part of a source?
		if (sourceID > 128)
			sourceOffsets.push_back(mediaOffset);

		/// fill default waveguide material (parameter n)
		if (epsilonR > 0)
		{
			// interpolate n based on green value.
			media.HostArray[mediaOffset] = epsilonMax * epsilonR * 1.f / 255;
		}
		
		if (monitorID > 0)
		{
			// add this to the list of cells with this monitor ID.
			unsigned int monitorOffset = j*m_fields[FieldType::Ez]->Size.x+i;
			monitorPositions[monitorId].push_back(monitorOffset);
		}
		
	}
}
