using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.Mathematics;

namespace GoLightly
{
    public class ModelProvider : MonoBehaviour
    {

        public Texture2D tileTexture;
        private Simulation _simulation;
        private SimulationParameters Parameters => _simulation.parameters;

        // Start is called before the first frame update
        void Start()
        {
            _simulation = GetComponent<Simulation>();
            Assert.IsNotNull(_simulation, $"Could not find {nameof(Simulation)} component.");
            //_simulation.onGenerateModels = generateModels;
            _simulation.onGenerateModels = generateCrystal;
            _simulation.onGenerateSources = generateSources;

        }

        // Update is called once per frame
        void Update()
        {

        }

        void generateCrystal(float[] data)
        {
            /* assume a source value of 600nm
             a = 1.2um
             r = 0.2 * a
             epsR = 8.9

             dx = lambda / 10 = 62nm
             
             assume dx = 

            */
            float cellWidth = 62; // nm




            


            


        }


        void generateSources(List<Simulation.Source> sources)
        {
            var size = _simulation.domainSize;
            var boundary = 0;
            var s = sources[0];
            sources.Clear();

            var left = 128;

            for(var j = boundary; j < size.y - boundary; ++j)
            {
                var s2 = s;
                var p = s.position;
                p.x = left;
                p.y = j;
                s2.position = p;
                sources.Add(s2);
            }

        }


        void generateModels(float[] data)
        {

            if (null == tileTexture)
            {
                Debug.Log($"No Tile is set. Cb array will be empty.");
                return;
            }

            var size = _simulation.domainSize;
            if (size.x % tileTexture.width != 0 || size.y % tileTexture.height != 0)
            {
                Debug.LogWarning($"Domain size is not evenly divisble by tile size. Truncation may occur");
            }
            Debug.Log($"Tile dimensions: {tileTexture.width}x{tileTexture.height}");

            var textureData = tileTexture.GetRawTextureData<Color32>();

            var dt = _simulation.parameters.dt;
            var dx = _simulation.parameters.dx;
            var cbDefault = _simulation.parameters.cb;

            
            var epsMax = 9;
            /*
             * cb = dt/dx * 1/eps;
             */
            for (var j = 0; j < size.y; ++j)
            {
                var tileY = j % tileTexture.height;
                for (var i = 0; i < size.x; ++i)
                {
                    var tileX = i % tileTexture.width;
                    var textureOffset = j * size.x + i;

                    var color = textureData[tileY * tileTexture.width + tileX];

                    var g = color.g;

                    if (g > 0)
                    //if (false)
                    {
                        //var c = dt / dx * 1.0f / 9;
                        var epsR = g * epsMax / 255.0f;
                        var c = dt / dx * 1.0f / epsR;
                        data[textureOffset] = c;
                    }
                    else
                    {
                        var n = dt / dx;
                        data[textureOffset] = n;
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