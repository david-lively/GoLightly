/*
Scan the image and generate a list of monitors.

Basically:
1. Loop through all pixels, looking for anything with a BLUE component, which indicates that the
pixel is part of a monitor. The value of the blue component is a unique identifier indicating to which monitor instance the pixel belongs
2. Add the linear offset of the given monitor pixel to the collection for that monitor (see map monitorPositions below)
3. Have the monitor Initialize itself with the collected positions.

*/
void Simulator::BuildMonitors(
	const unsigned char *bytes
	, const unsigned int width
	, const unsigned int height
	, const unsigned int channels
	, const unsigned int framesToAllocate
	)
{
	map<unsigned char, vector<unsigned int>> monitorPositions;

	auto layers = m_configuration.PmlLayers;

	for (unsigned int j = layers; j < height - 1 - layers; ++j)
	{
		for (unsigned int i = layers; i < width - 1 - layers; ++i)
		{
			unsigned int pixelOffset = channels  * (j * width + i);

			/// get third byte for this pixel (blue component), which is the monitor ID if this is a monitor cell.
			unsigned char monitorId = bytes[pixelOffset + 2];

			if (monitorId > 0)
			{
				auto &mp = monitorPositions[monitorId];

				unsigned int monitorOffset = j * m_fields[FieldType::Ez]->Size.x + i;
				mp.push_back(monitorOffset);
			}

		}
	}

	for (auto it = begin(monitorPositions); it != end(monitorPositions); ++it)
	{
		vector<unsigned int> &positions = it->second;

		auto ptr = make_shared<Monitor>();
		ptr->Id = it->first;

		m_monitors[it->first] = ptr;

		ptr->Initialize(m_cuda, positions, 1);
	}
}
