using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

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

        void generateModels(ComputeBuffer cb)
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

            using var textureData = tile.GetRawTextureData<Color32>();

            var cbDefault = _simulation.parameters.cb;

            var data = new float[size.x * size.y];
            for (var j = 0; j < size.y; ++j)
            {
                var tileY = j % tile.height;
                for (var i = 0; i < size.x; ++i)
                {
                    var tileX = i % tile.width;

                    var textureOffset = j * size.x + i;



                }
            }

        }
    }
}