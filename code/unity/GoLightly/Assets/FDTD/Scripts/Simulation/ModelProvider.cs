using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;

namespace GoLightly
{
	public class ModelProvider
	{
		private static float length(float x, float y)
		{
			return Mathf.Sqrt(x * x + y * y);
		}
		public static void Cylinder(uint2 center, float radius, float width, int domainWidth, float material, float[] cb)
		{
			var tl = new int2((int)(center.x - radius), (int)(center.y - radius));
			var br = new int2((int)(center.x + radius), (int)(center.y + radius));

			for (var j = -radius; j <= radius; ++j)
			{
				for (var i = -radius; i <= radius; ++i)
				{
					var d = length(i, j);
					if (d <= radius && d >= radius - width)
					{
						var x = (int)(i + center.x);
						var y = (int)(j + center.y);
						var offset = y * domainWidth + x;
						cb[offset] = material;
					}

				}
			}

		}


		/*
		void ModelProvider::Cylinder(FieldDescriptor &media, float n, float radius, int centerX, int centerY, float width)
		{
			dim3 center(centerX, centerY);

			if (center.x < 0)
			{
				center.x = media.Size.x / 2;
			}

			if (center.y < 0)
			{
				center.y = media.Size.y / 2;
			}

			if (radius < 0)
			{
				radius = min(center.x, center.y);
			}

			if (width < 0)
				width = radius + 1;

			auto topLeft = center - radius;
			auto bottomRight = center + radius;

			for (unsigned int j = topLeft.y; j < bottomRight.y; j++)
			{
				for (unsigned int i = topLeft.x; i < bottomRight.x; i++)
				{
					float r = length(dim3(center.x - i, center.y - j));
					if (r <= radius && r >= radius - width)
					{
						media.HostArray[j * media.Size.x + i] = n;
					}

				}
			}

		}*/
	}
}