using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.Mathematics;
using UnityEditor;

namespace GoLightly
{
    [DisallowMultipleComponent]
    public class ModelProvider : MonoBehaviour
    {

        public Texture2D tileTexture;
        private Simulation _simulation;
        private SimulationParameters Parameters => _simulation.parameters;
        public bool enableDielectric = true;


        #region rod lattice parameters

        /* rod lattice parameters */
        public int rodSpacingNm = 1200;
        // public int rodDiameterNm => (int)(2 * rodSpacingNm * 0.2f);
        public float rodDiameterNm = 1200 * 0.2f * 2;
        public int cellDivisor = 10;
        public float lambdaNm = 1500;
        public bool includeDefect = false;

        #endregion

        #region calculated values

        public float nmPerYeeCell;
        public float lambdaNorm;
        public int rodSpacingCells;
        public int rodDiameterCells;
        public float dx;
        public float dt;

        /// <summary>
        /// nm = norm * (10 * nmPerYeeCell)
        /// </summary>
        public float lambdaScalar;

        #endregion        


        // Start is called before the first frame update
        void Awake()
        {
            _simulation = FindObjectOfType<Simulation>();
            Assert.IsNotNull(_simulation, $"Could not find {nameof(Simulation)} component.");
            if (isActiveAndEnabled)
                _simulation.onGenerateModels = generateRodArray2;

        }

        // Update is called once per frame
        void Update()
        {

        }

        void generateSources(List<Simulation.Source> sources)
        {
            var size = _simulation.domainSize;
            var boundary = 0;
            var s = sources[0];
            sources.Clear();

            var left = 128;

            for (var j = boundary; j < size.y - boundary; ++j)
            {
                var s2 = s;
                var p = s.position;
                p.x = left;
                p.y = j;
                s2.position = p;
                sources.Add(s2);
            }

        }

#if false
        void generateModels(float[] data)
        {
            if (!isActiveAndEnabled)
                return;
            Debug.Log($"ModelProvider.GenerateModels");
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


            var epsMax = 8.9f;
            /*
             * cb = dt/dx * 1/eps;
             */
            var maxEpsR = float.MinValue;

            var mn = 17 * 30;
            var mx = mn + 30;

            var a = 150;
            var r = 30;


            var tileOffset = Vector2Int.zero; //new Vector2Int(tileTexture.width / 2, tileTexture.height / 2);
            // tileOffset.x = 62+15+30;
            // tileOffset.y = tileTexture.height - 37*2+12-12+6+6;

            for (var j = 0; j < size.y; ++j)
            {
                var tileY = (j + tileOffset.y) % tileTexture.height;

                for (var i = 0; i < size.x; ++i)
                {
                    var tileX = (i + tileOffset.x) % tileTexture.width;
                    var materialAddress = j * size.x + i;

                    var color = textureData[tileY * tileTexture.width + tileX];

                    var g = color.g;

                    // if ((j >= mn && j <= mx) || g <= 0)
                    if (g <= 0 || !enableDielectric)
                    {
                        var n = dt / dx;
                        data[materialAddress] = n;
                    }
                    else
                    {
                        //var c = dt / dx * 1.0f / 9;
                        var epsR = g * epsMax / 255.0f;
                        maxEpsR = Mathf.Max(epsR, maxEpsR);
                        var c = dt / dx * 1.0f / epsR;
                        data[materialAddress] = c;
                    }
                }
            }

            Debug.Log($"Max epsilon R = {maxEpsR}");


            //cb.SetData(data);
        }
#endif
        private static float length(float x, float y)
        {
            return Mathf.Sqrt(x * x + y * y);
        }

        public static void Cylinder(int2 center, float radius, float width, int domainWidth, int domainHeight, float material, float[] cb)
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

                        if (x >= domainWidth || x < 0 || y >= domainHeight || y < 0)
                            continue;
                        var offset = y * domainWidth + x;
                        cb[offset] = material;
                    }

                }
            }

        }

        public void generateRodArray(float[] cb)
        {
            var dt = _simulation.parameters.dt;
            var dx = _simulation.parameters.dx;
            var cbDefault = _simulation.parameters.cb;

            var epsR = 8.9f;
            var air = dt / dx;
            var dielectric = dt / dx * 1.0f / epsR;

            var a = 30;
            var r = a * 0.2f;

            var sourcePosition = new int2(512, 512);
            Debug.Log($"Using source position {sourcePosition}");

            var offset = 512 % a - a / 2;

            var domainWidth = _simulation.domainSize.x;
            var domainHeight = _simulation.domainSize.y;

            Simulation.Helpers.ClearArray(ref cb, air);

            // offset to center this around the source

            for (var i = offset; i <= domainWidth; i += a)
            {
                for (var j = offset; j <= domainHeight; j += a)
                {
                    var center = new int2(i, j);
                    Cylinder(center, r, r, domainWidth, domainHeight, dielectric, cb);
                }
            }

        }



        public void calculateParameters()
        {
            nmPerYeeCell = rodDiameterNm / cellDivisor;

            rodSpacingCells = Mathf.CeilToInt(rodSpacingNm / nmPerYeeCell);
            rodDiameterCells = Mathf.CeilToInt(rodDiameterNm / nmPerYeeCell);

            lambdaScalar = 10 * nmPerYeeCell;
            lambdaNorm = lambdaNm / lambdaScalar;

            var batchRun = GameObject.FindObjectOfType<BatchRun>();
            if (null != batchRun)
                batchRun.lambdaScalar = lambdaScalar;

            // dx = _simulation.parameters.dx;
            // dt = _simulation.parameters.dt;
        }

        public void generateRodArray2(float[] cb)
        {
            calculateParameters();

            /*
            rod spacing  A= 1200nm
            rod diameter D= 480nm

            dx = D/10 = 48nm

            So, in array coordinates:
            diameter = 10
            Spacing = 1200/48nm = 25

            Set source wavelength to lambda nm / dx
            For lambda = 1500nm, wavelength = 31.25

            if dx = 1200 * 0.4 / 10 = 48um,
            and default lambda = 10 * dx = 480um, 
            then lambda_rel = 1500 / 480nm = 3.125

            // */
            // float nmPerCell = rodDiameterNm / cellDivisor;
            // // float dx_nm = rodDiameterNm / 10;
            // int rodSpacingCells = Mathf.CeilToInt(rodSpacingNm / nmPerCell);
            // int rodDiameterCells = Mathf.CeilToInt(rodDiameterNm / nmPerCell);
            // float lambdaNm = 1500;

            // /// FDTD condition: dx = lambda / 10. But, we're defining dx
            // /// in terms of rod diameter. 
            // float lambdaNorm = cellDivisor * nmPerCell;
            // float lambdaRel = lambdaNm / (cellDivisor * nmPerCell);


            _simulation.modelRequestedLambda = lambdaNorm;

            if (!enableDielectric)
            {
                return;
            }

            var dt = _simulation.parameters.dt;
            var dx = _simulation.parameters.dx;
            var cbDefault = _simulation.parameters.cb;

            var epsR = 8.9f;
            var air = dt / dx;
            var dielectric = dt / dx * 1.0f / epsR;

            var sourcePosition = new int2(512, 512);
            Debug.Log($"Using source position {sourcePosition}");


            var domainWidth = _simulation.domainSize.x;
            var domainHeight = _simulation.domainSize.y;

            var radius = rodDiameterCells / 2;

            Simulation.Helpers.ClearArray(ref cb, air);
            var offset = 0;
            var cellCount = new int2(domainWidth / rodSpacingCells + 1, domainHeight / rodSpacingCells + 1);
            var excludeCell = new bool[cellCount.x, cellCount.y];

            if (includeDefect)
            {
                // offset to center this around the source
                offset = radius * 5 / 2;                                        
                excludeCell[cellCount.x / 2, cellCount.y / 2] = true;
            }
            else
            {
                offset = 512 % rodSpacingCells - rodSpacingCells / 2;
            }

            offset = 512 % rodSpacingCells - rodSpacingCells / 2;


            for (var j = 0; j < cellCount.y; ++j)
            {
                for (var i = 0; i < cellCount.x; ++i)
                {
                    if (excludeCell[i, j])
                        continue;

                    // if (i >= 6 && i < 17 && j >= 6 && j < 17)
                    //     continue;

                    var x = offset + i * rodSpacingCells;
                    var y = offset + j * rodSpacingCells;
                    var center = new int2(x, y);
                    Cylinder(center, radius, radius, domainWidth, domainHeight, dielectric, cb);
                }
            }

        }

    }

    [CustomEditor(typeof(ModelProvider))]
    public class ModelProviderInspector : Editor
    {
        ModelProvider provider => target as ModelProvider;

        public override void OnInspectorGUI()
        {
            if (GUILayout.Button("Recalculate grid parameters"))
            {
                provider.calculateParameters();
            }
            base.OnInspectorGUI();
        }

    }
}