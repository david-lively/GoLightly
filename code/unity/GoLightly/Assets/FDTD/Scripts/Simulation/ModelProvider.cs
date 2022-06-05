using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.Mathematics;

namespace GoLightly
{
    public class ModelProvider : MonoBehaviour
    {

        public Texture2D tile;
        private Simulation _simulation;
        // Start is called before the first frame update
        void Start()
        {
            _simulation = GetComponent<Simulation>();
            Assert.IsNotNull(_simulation, $"Could not find {nameof(Simulation)} component.");
            _simulation.onGenerateModels = generateModels;

        }

        // Update is called once per frame
        void Update()
        {

        }

        void generateModels(float[] data)
        {

            if (null == tile)
            {
                Debug.Log($"No Tile is set. Cb array will be empty.");
                return;
            }

            var size = _simulation.domainSize;
            if (size.x % tile.width != 0 || size.y % tile.height != 0)
            {
                Debug.LogWarning($"Domain size is not evenly divisble by tile size. Truncation may occur");
            }
            Debug.Log($"Tile dimensions: {tile.width}x{tile.height}");

            var textureData = tile.GetRawTextureData<Color32>();

            var dt = _simulation.parameters.dt;
            var dx = _simulation.parameters.dx;
            var cbDefault = _simulation.parameters.cb;

            
            var epsMax = 9;
            /*
             * cb = dt/dx * 1/eps;
             */
            for (var j = 0; j < size.y; ++j)
            {
                var tileY = j % tile.height;
                for (var i = 0; i < size.x; ++i)
                {
                    var tileX = i % tile.width;
                    var textureOffset = j * size.x + i;

                    var color = textureData[tileY * tile.width + tileX];

                    var g = color.g;



                    if (g > 0)
                    {
                        var c = dt / dx * 1.0f / 9;

                        data[textureOffset] = c;
                    }
                    else
                    {
                        data[textureOffset] = cbDefault;
                    }
                }
            }

            //cb.SetData(data);
        }


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

    }
}