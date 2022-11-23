using System;
using System.Collections;
using System.Collections.Generic;
// using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Events;
using Unity.Mathematics;
using UnityEditor;

/*
/// <summary>
/// initializes resources such as field and boundary arrays and sets material properties
/// </summary>
/// <remarks>
/// ez---hy---ez---hy---ez---hy---ez---hy---ez
/// |         |         |         |         |         
/// hx        hx        hx        hx        hx
/// |         |         |         |         |         
/// ez---hy---EZ---hy---EZ---hy---EZ---hy---ez
/// |         |         |         |         |         
/// hx        hx        hx        hx        hx
/// |         |         |         |         |         
/// ez---hy---EZ---hy---EZ---hy---EZ---hy---ez
/// |         |         |         |         |         
/// hx        hx        hx        hx        hx
/// |         |         |         |         |         
/// ez---hy---EZ---hy---EZ---hy---EZ---hy---ez
/// |         |         |         |         |         
/// hx        hx        hx        hx        hx
/// |         |         |         |         |         
/// ez---hy---ez---hy---ez---hy---ez---hy---ez
/// </remarks>
*/
namespace GoLightly
{
    [DisallowMultipleComponent]
    public partial class Simulation : MonoBehaviour
    {
        [Header("Simulation parameters")]
        public Vector2Int outputTextureSize = new Vector2Int(2048, 1024);
        public RenderTexture outputTexture;
        public ComputeShader computeShader;
        public Vector2Int domainSize = new Vector2Int(2048, 1024);

        public SimulationParameters parameters = SimulationParameters.Create(1.0f);

        public UnityAction<float[]> onGenerateModels;
        public UnityAction<List<Source>> onGenerateSources;
        public UnityAction<List<Monitor>> onUpdateInstruments;

        /// <summary>
        /// Select to clear all field buffers to zero on the next update.
        /// </summary>
        internal bool resetRequested;
        internal bool isPaused;

        /// <summary>
        /// When running batch mode (sweeping wavelengths), the delta between
        /// wavelengths on each subsequent run. 
        /// </summary>
        public float sweepWavelengthDelta = 0.1f;

        /// <summary>
        /// Contrast multiplier to use when drawing field values to 
        /// the visualizer texture.
        /// </summary>
        [Range(1, 200)]
        public float contrast = 80;

        /// <summary>
        /// Current simulation time step (t). Don't modify this in code. It's just here
        /// so we can see the simulation progress in the Unity inspector.
        /// </summary>
        public int timeStep = 0;

        /// <summary>
        /// For the visualizer, the contrast multiplier used to render PML values
        /// </summary>
        [Range(0, 100)]
        public float psiContrast = 0;

        /// <summary>
        /// Number of field updates (simulation steps) to calculate between visualizer updates. 
        /// Updating the visualizer takes non-trivial GPU resources, so we don't want to do it every frame.
        /// </summary>
        [Range(1, 200)]    
        public uint simulationTimeStepsPerFrame = 1;

        public float modelRequestedLambda = 1.0f;

        /// <summary>
        /// PML decay values for E
        /// </summary>
        readonly float[] e_decay = new float[] {
                    1,
                    0.999798477f,
                    0.996780813f,
                    0.983808935f,
                    0.9497177f,
                    0.881655931f,
                    0.770144641f,
                    0.616397917f,
                    0.438038647f,
                    0.266546071f,
                    0.133286506f,
                    0,
                    0.133286506f,
                    0.266546071f,
                    0.438038647f,
                    0.616397917f,
                    0.770144641f,
                    0.881655931f,
                    0.9497177f,
                    0.983808935f,
                    0.996780813f,
                    0.999798477f,
                    1
                };

        /// <summary>
        /// PML decay values for H
        /// </summary>
        readonly float[] h_decay = new float[]  {
                    1,
                    0.999987423f,
                    0.998980284f,
                    0.99215883f,
                    0.970211327f,
                    0.920684338f,
                    0.831596196f,
                    0.697860897f,
                    0.528538823f,
                    0.349247128f,
                    0.193701461f,
                    0,
                    0.193701461f,
                    0.349247128f,
                    0.528538823f,
                    0.697860897f,
                    0.831596196f,
                    0.920684338f,
                    0.970211327f,
                    0.99215883f,
                    0.998980284f,
                    0.999987423f,
                    1,
                       };

        /// <summary>
        /// Compute shader buffers, referenced by key = buffer name ("ez", "hx", etc as defined in the compute shader source file). 
        /// </summary>
        private readonly Dictionary<string, ComputeBuffer> _buffers = new Dictionary<string, ComputeBuffer>(StringComparer.OrdinalIgnoreCase);
        /// <summary>
        /// Compute shader kernels referenced by key = kernel name.
        /// </summary>
        private readonly Dictionary<string, int> _kernels = new Dictionary<string, int>();

        /// <summary>
        /// Parameters for an FDTD source. These should not be modified at runtime. 
        /// To see the source in the Unity scene view or game view, be sure to enable
        /// Gizmos on the Unity toolbar.
        /// </summary>
        [System.Serializable]
        public struct Source
        {
            public Vector3 position;
            public float amplitude;
            /// <summary>
            /// Normalized wavelength = dx / 10. So this should be lambdaRelative = lambda / (10 * dx).
            /// </summary>
            public float wavelength;
            public float maxLife;
            public uint _enabled;

            public static int GetSize()
            {
                return (3 * sizeof(float)) + (3 * sizeof(float)) + sizeof(uint);
            }
        }
        /// <summary>
        /// All of the sources that we have.
        /// </summary>
        public List<Source> sources = new List<Source>();

        /// <summary>
        /// FDTD field monitors 
        /// </summary>
        private List<Monitor> monitors = new List<Monitor>();

        /// <summary>
        /// linear offsets into the field arrays for each monitor.
        /// </summary>
        private List<int> monitorAddresses = new List<int>();

        // Start is called before the first frame update
        void Start()
        {
            if (null == outputTexture)
            {
                Debug.Log("Creating render texture");
                // outputTexture = new RenderTexture(domainSize.x, domainSize.y, 24);
                outputTexture = new RenderTexture(outputTextureSize.x, outputTextureSize.y, 24);
                outputTexture.enableRandomWrite = true;

                var textureWasCreated = outputTexture.Create();
                Assert.IsTrue(textureWasCreated, "Could not create visualizer texture.");
            }

            InitComputeResources();
        }

        private bool _isInitialized = false;

        /// <summary>
        /// For batch mode, increment the source wavelength to the next
        /// value of interest and restart the simulation.
        /// </summary>
        internal void runNextWavelength()
        {
            // isPaused = true;
            resetRequested = true;
            var source = sources[0];
            source.wavelength += sweepWavelengthDelta;
            sources[0] = source;

            uploadSources();
        }

        /// <summary>
        /// For batch mode, clear the monitor values.
        /// </summary>
        private void resetMonitors()
        {
            var chartManager = GameObject.FindObjectOfType<ChartManager>();
            chartManager?.ResetCharts();
        }

        internal void SetSingleSourceWavelengthAndReset(float lambda)
        {
            resetRequested = true;
            var source = sources[0];
            source.wavelength = lambda;
            sources[0] = source;

            resetMonitors();

            uploadSources();
        }

        /// <summary>
        /// Copy configured source parameters to the GPU
        /// </summary>
        void uploadSources()
        {
            if(!_buffers.TryGetValue("sources", out var sourcesBuffer))
                sourcesBuffer = new ComputeBuffer(sources.Count, Source.GetSize());
            sourcesBuffer.SetData(sources);
            computeShader.SetInt("numSources", sources.Count);
            _buffers["sources"] = sourcesBuffer;
        }


        private void InitComputeResources()
        {
            if (_isInitialized)
                return;

            if (null == outputTexture)
            {
                Debug.Log("Creating render texture");
                outputTexture = new RenderTexture(domainSize.x, domainSize.y, 24);
                outputTexture.enableRandomWrite = true;
                outputTexture.Create();
            }

            {
                onGenerateSources?.Invoke(sources);
                uploadSources();
            }

            {
                var parameterBuffer = new ComputeBuffer(1, SimulationParameters.GetSize());
                var pdata = new SimulationParameters[] { parameters };
                parameterBuffer.SetData(pdata);
                _buffers["Parameters"] = parameterBuffer;
            }

            var fieldBufferSize = domainSize.x * domainSize.y;

            {
                var fieldBufferNames = new string[] { "ez", "hx", "hy" };

                foreach (var name in fieldBufferNames)
                {
                    var buffer = new ComputeBuffer(fieldBufferSize, sizeof(float));
                    Helpers.ClearBuffer(buffer);
                    _buffers[name] = buffer;
                }
            }

            {
                var cbData = new float[fieldBufferSize];
                Helpers.ClearArray(ref cbData, parameters.cb);
                var cb = new ComputeBuffer(fieldBufferSize, sizeof(float));
                onGenerateModels?.Invoke(cbData);

                cb.SetData(cbData);
                _buffers["cb"] = cb;
            }

            /// Kernel names that we're using as defined in the compute shader source file "FDTDComputeShaders.compute"
            var kernelNames = new string[] {
                "CSUpdateVisualizerTexture"
                ,"CSUpdateEz"
                ,"CSUpdateHFields"
                ,"CSUpdateSources"
                ,"CSUpdateMonitors"
            };

            /// Get a reference to each compiled kernel in the compute shader
            foreach (var name in kernelNames)
            {
                var kernelIndex = computeShader.FindKernel(name);
                Assert.IsTrue(kernelIndex >= 0, $"Compute shader kernel `{name}` not found!");

                _kernels[name] = kernelIndex;
            }

            InitializeBoundaries(parameters.pmlLayers);

            InitializeMonitors();

            _isInitialized = true;
        }

        /// <summary>
        /// Clean up unmanaged resources
        /// </summary>
        private void OnDestroy()
        {
            Debug.Log($"Disposing {_buffers.Count} compute buffers");

            foreach (var kvp in _buffers)
            {
                kvp.Value?.Dispose();
            }
            _buffers.Clear();

            outputTexture.Release();
            outputTexture = null;

            _isInitialized = false;
        }

        private void RunKernel(int kernelIndex)
        {
            /// Use a grid block size of domainSize / 32, or 32*32 = 1024 threads per block.
            var threadGroupsX = domainSize.x / 32;
            var threadGroupsY = domainSize.y / 32;

            RunKernel(kernelIndex, threadGroupsX, threadGroupsY);
        }

        private void RunKernel(int kernelIndex, int threadGroupsX, int threadGroupsY)
        {
            computeShader.SetBuffer(kernelIndex, "ez", _buffers["ez"]);
            computeShader.SetBuffer(kernelIndex, "hx", _buffers["hx"]);
            computeShader.SetBuffer(kernelIndex, "hy", _buffers["hy"]);
            if (monitorAddresses.Count > 0)
            {
                computeShader.SetBuffer(kernelIndex, "monitorAddresses", _buffers["monitorAddresses"]);
                computeShader.SetBuffer(kernelIndex, "monitorValues", _buffers["monitorValues"]);
            }
            computeShader.SetBuffer(kernelIndex, "sources", _buffers["sources"]);
            computeShader.SetBuffer(kernelIndex, "decay_all", _buffers["decay_all"]);
            computeShader.SetBuffer(kernelIndex, "cb", _buffers["cb"]);
            computeShader.SetConstantBuffer("Parameters", _buffers["Parameters"], 0, SimulationParameters.GetSize());
            computeShader.SetBuffer(kernelIndex, "psi_all", _buffers["psi_all"]);

            computeShader.Dispatch(kernelIndex, threadGroupsX, threadGroupsY, 1);
        }

        /// <summary>
        /// Run one or more simulation steps. These are done in a tight loop to avoid
        /// engine overhead per frame, or to get multiple simulation time step updates
        /// for each visualizer update.
        /// </summary>
        /// <param name="steps">Number of simulation time steps to run</param>
        private void RunSimulationSteps(uint steps = 1)
        {
            computeShader.SetFloat("time", Time.fixedTime);
            computeShader.SetVector("domainSize", new Vector4(domainSize.x, domainSize.y, 0, 0));
            computeShader.SetTexture(0, "VisualizerTexture", outputTexture);

            computeShader.SetVector("domainSize", new Vector2(domainSize.x, domainSize.y));
            computeShader.SetInt("numSources", sources.Count);
            computeShader.SetInt("numMonitorAddresses", monitorAddresses.Count);
            computeShader.SetBool("resetRequested", resetRequested);

            var kernelNames = new string[] { "CSUpdateHFields", "CSUpdateEz" };

            if (steps < 1)
                steps = 1;

            if (resetRequested)
            {
                timeStep = 0;
                steps = 1;
            }


            for (var j = 0; j < steps; ++j)
            {
                //computeShader.SetFloat("time", Time.fixedTime);
                computeShader.SetInt("TimeStep", timeStep);

                {
                    var sourceThreads = (int)Mathf.CeilToInt(sources.Count / 64.0f);
                    RunKernel(_kernels["CSUpdateSources"], sourceThreads, 1);
                }

                for (var i = 0; i < kernelNames.Length; ++i)
                {
                    RunKernel(_kernels[kernelNames[i]]);
                }

                if (monitorAddresses.Count > 0)
                {
                    var numThreads = (int)Mathf.CeilToInt(monitorAddresses.Count / 64.0f);
                    RunKernel(_kernels["CSUpdateMonitors"], numThreads, 1);
                    ReadMonitors();
                }

                ++timeStep;
            }

            resetRequested = false;

            UpdateVisualizerTexture(outputTexture);
        }

        /// <summary>
        /// Draw a visualization of the field of interest to the associated render texture.
        /// </summary>
        /// <param name="gameView"></param>
        public void UpdateVisualizerTexture(RenderTexture gameView)
        {
            var kernelName = "CSUpdateVisualizerTexture";
            if (_kernels.TryGetValue(kernelName, out var kernelIndex))
            {
                computeShader.SetTexture(kernelIndex, "GameViewTexture", gameView);
                computeShader.SetTexture(kernelIndex, "VisualizerTexture", outputTexture);
                computeShader.SetFloat("contrast", contrast);
                computeShader.SetFloat("psiContrast", psiContrast);
                computeShader.SetVector("GameViewTextureSize", new Vector4(gameView.width, gameView.height, 0, 0));
                RunKernel(kernelIndex);
            }
            else
                Debug.LogError($"Compute kernel `{kernelName}` not found.");
        }

        // Update is called once per frame
        public void Update()
        {
            if (!isPaused)
                RunSimulationSteps(simulationTimeStepsPerFrame);
        }


        /// <summary>
        /// Clear the E and H fields and reset the source state.        /// 
        /// </summary>
        public void Reset()
        {

        }

        /// <summary>
        /// Generate PML values for the outside of the domain.
        /// </summary>
        /// <param name="decayAll"></param>
        /// <param name="minCoord"></param>
        /// <param name="maxCoord"></param>
        private void CreateBoundaryOutside(float4[] decayAll, int2 minCoord, int2 maxCoord)
        {
            /*
            x -> ezx decay
            y -> ezy decay
            z -> hyx decay
            w -> hxy decay
            */
            var layers = e_decay.Length;
            for (var k = 1; k < layers - 1; ++k)
            {
                /// draw left and right layers
                for (var j = minCoord.y; j < maxCoord.y; ++j)
                {
                    /// left 
                    {
                        var o = offsetOf(minCoord.x + k - 1, j);
                        var v = decayAll[o];
                        v.x = e_decay[k];
                        v.z = h_decay[k];

                        decayAll[o] = v;
                    }

                    /// right
                    {

                        var o = offsetOf(maxCoord.x - k, j);
                        var v = decayAll[o];
                        v.x = e_decay[k];
                        v.z = h_decay[k - 1];

                        decayAll[o] = v;
                    }

                }

                /// draw top and bottom PML layers
                for (var i = minCoord.x; i < maxCoord.x; ++i)
                {
                    /// top
                    {
                        var o = offsetOf(i, minCoord.y + k - 1);
                        var v = decayAll[o];
                        v.y = e_decay[k];
                        v.w = h_decay[k];
                        decayAll[o] = v;
                    }
                    /// bottom
                    {
                        var o = offsetOf(i, maxCoord.y - k);
                        var v = decayAll[o];
                        v.y = e_decay[k];
                        v.w = h_decay[k - 1];
                        decayAll[o] = v;
                    }

                }


            }
        }

        /// <summary>
        /// Helper method to convert x,y coordinate to a linear array offset.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        int offsetOf(int x, int y)
        {
            return y * domainSize.x + x;
        }

        /// <summary>
        /// Clip the coordinate to a valid field coordinate
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        int2 clip(int2 v)
        {
            v.x = v.x < 0 ? 0 : (v.x >= domainSize.x) ? domainSize.x - 1 : v.x;
            v.y = v.y < 0 ? 0 : (v.y >= domainSize.y) ? domainSize.y - 1 : v.y;

            return v;
        }

        /// <summary>
        /// Generate PML coefficients for a circular area with the given CENTER
        /// and outer RADIUS. RADIUS should be at least > 12. 
        /// </summary>
        /// <param name="decay"></param>
        /// <param name="center"></param>
        /// <param name="radius"></param>
        private void CreateSink(float4[] decay, int2 center, int radius)
        {
            Assert.IsTrue(radius >= e_decay.Length);
            for (var i = -radius; i <= radius; ++i)
            {
                for (var j = -radius; j <= radius; ++j)
                {
                    // calculate layer
                    var d = radius - (int)Mathf.Sqrt(i * i + j * j);
                    var o = offsetOf(i + center.x, j + center.y);
                    // var v = decay[o];

                    if (d > 0 && d < e_decay.Length - 1)
                    {
                        /*
                        x -> ezx decay
                        y -> ezy decay
                        z -> hyx decay
                        w -> hxy decay
                        */
                        var v = new float4();
                        v.x = e_decay[d];
                        v.y = e_decay[d];
                        v.z = h_decay[d + 1];
                        v.w = h_decay[d + 1];
                        decay[o] = v;
                    }
                }
            }
        }

        private bool fixCoordinateOrder(ref int2 mn, ref int2 mx)
        {
            var didSomething = false;

            if (mn.x > mx.x)
            {
                var t = mn.x;
                mn.x = mx.x;
                mx.x = t;

                didSomething = true;
            }

            if (mn.y > mx.y)
            {
                var t = mn.y;
                mn.y = mx.y;
                mx.y = mn.y;

                didSomething = true;
            }

            return didSomething;

        }


        /// <summary>
        /// For PML sinks, create a rectangluar sink with corners at the given location.s
        /// </summary>
        /// <param name="decayAll"></param>
        /// <param name="minCoord"></param>
        /// <param name="maxCoord"></param>
        private void CreateSink(float4[] decayAll, int2 minCoord, int2 maxCoord)
        {
            minCoord = clip(minCoord);
            maxCoord = clip(maxCoord);

            /*
            x -> ezx decay
            y -> ezy decay
            z -> hyx decay
            w -> hxy decay
            */
            var layers = e_decay.Length;
            for (var k = 1; k < layers - 1; ++k)
            {
                /// draw left and right layers
                for (var j = minCoord.y; j < maxCoord.y; ++j)
                {
                    /// left 
                    {
                        var o = offsetOf(minCoord.x + k - 1, j);
                        var v = decayAll[o];
                        v.x = e_decay[k - 1];
                        v.z = h_decay[k];

                        decayAll[o] = v;
                    }
                    /// right
                    {
                        var o = offsetOf(maxCoord.x - k, j);
                        var v = decayAll[o];
                        v.x = e_decay[k];
                        v.z = h_decay[k];

                        decayAll[o] = v;
                    }
                }

                /// Populate top and bottom (relative to the domain Y coordinates) layers
                for (var i = minCoord.x; i < maxCoord.x; ++i)
                {
                    /// top
                    {
                        var o = offsetOf(i, minCoord.y + k - 1);
                        var v = decayAll[o];
                        v.y = e_decay[k - 1];
                        v.w = h_decay[k];
                        decayAll[o] = v;
                    }
                    /// bottom
                    {
                        var o = offsetOf(i, maxCoord.y - k);
                        var v = decayAll[o];
                        v.y = e_decay[k];
                        v.w = h_decay[k];
                        decayAll[o] = v;
                    }

                }


            }
        }

        /// <summary>
        /// Draw a linear waveguide at the given location. The guide spans the entire domain and
        /// assumes a core epsilon_r of 9 and a clas epsilon_r of 3. 
        /// </summary>
        /// <param name="cbData"></param>
        /// <param name="centerY"></param>
        void lineGuide(float[] cbData, int centerY)
        {
            var scalar = parameters.dt / parameters.dx;
            var cladMaterial = scalar * 1.0f / 9;//parameters.dt / parameters.dx * 1.0f / 9;
            var coreMaterial = scalar * 1.0f / 3;
            // var coreMaterial = cladMaterial;

            //var width = 20;
            //var top = middleY - width / 2;

            var coreLayers = 4;
            var cladLayers = 12;

            var top = centerY - (coreLayers / 2 + cladLayers);
            var bottom = top + coreLayers + 2 * cladLayers;

            for (var i = 0; i < domainSize.x; ++i)
            {
                var y = top;
                for (var j = 0; j < cladLayers; ++j)
                {
                    cbData[y * domainSize.x + i] = cladMaterial;
                    ++y;
                }
                for (var j = 0; j < coreLayers; ++j)
                {
                    cbData[y * domainSize.x + i] = coreMaterial;
                    ++y;
                }
                for (var j = 0; j < cladLayers; ++j)
                {
                    cbData[y * domainSize.x + i] = cladMaterial;
                    ++y;
                }
            }

        }

        void demoGuide2(float[] cbData)
        {
            var scalar = parameters.dt / parameters.dx;
            var core = scalar * 1.0f / 9;
            var clad = scalar * 1.0f / 1000;

            var coreLayers = 3;
            var cladLayers = 3;

            var top = domainSize.y / 2 - coreLayers / 2 - cladLayers;
            for (var j = 0; j < coreLayers + 2 * cladLayers; ++j)
            {
                var cb = scalar;
                if (j < cladLayers)
                    cb = clad;
                else if (j < coreLayers + cladLayers)
                    cb = core;
                else
                    cb = clad;

                for (var i = 0; i < domainSize.x; ++i)
                {
                    cbData[(j + top) * domainSize.x + i] = cb;
                }
            }
        }

        /// <summary>
        /// Draw a Whispering Gallery Mode circle waveguide.
        /// </summary>
        /// <param name="cbData"></param>
        /// <param name="radius"></param>
        /// <param name="width"></param>
        void wgm(float[] cbData, float radius, float width)
        {
            var center = new int2(domainSize.x / 2, domainSize.y / 2);
            var m = (parameters.dt / parameters.dx * 1 / 9.0f);
            ModelProvider.Cylinder(center, radius, width, domainSize.x, domainSize.y, m, cbData);
        }

        void SetMaterials(float[] cbData)
        {
            lineGuide(cbData, domainSize.y / 2 - 300);
            wgm(cbData, 282, 20);
        }

        /// <summary>
        /// Initialize PML boundaries
        /// </summary>
        /// <param name="_"></param>
        private void InitializeBoundaries(int _)
        {
            var layers = e_decay.Length;

            var domainWidth = domainSize.x;
            var domainHeight = domainSize.y;

            /// unified PML psi buffer
            var psiBuffer = new ComputeBuffer(domainWidth * domainHeight, sizeof(float) * 4);
            _buffers["psi_all"] = psiBuffer;

            /// unified PML decay buffer.
            var decayAll = new float4[domainWidth * domainHeight];

            Helpers.SetArray(ref decayAll, 1);
            CreateBoundaryOutside(decayAll, 0, new int2(domainSize.x - 2, domainSize.y - 2));


#if false
            var coords = new int2[] {
                new int2(30, 300), new int2(700, 1000)
                ,new int2(2048 - 700, 300), new int2(2048 - 30, 1000)
                ,new int2(12, 12), new int2(2048 - 12, 150)
            };

            var pmlArea = 0;

            for (var i = 0; i < coords.Length; i += 2)
            {
                var mn = coords[i];
                var mx = coords[i + 1];
                CreateSink(decayAll, mn, mx);

                var size = mx - mn;

                var area = size.x * size.y;
                pmlArea += area;
            }

            // CreateSink(decayAll, new int2(30, 300), new int2(700, 1000));
            // CreateSink(decayAll, new int2(2048 - 700, 300), new int2(2048 - 30, 1000));
            // CreateSink(decayAll, new int2(12, 12), new int2(2048 - 12, 150));

            var center = new int2(domainSize.x / 2, domainSize.y / 2);
            var radius = 200;
            CreateSink(decayAll, center, radius);
            pmlArea += (int)(Mathf.PI * radius * radius);


            var domainArea = domainSize.x * domainSize.y;
            Debug.Log($"Total area:\ndomain {domainArea}\nPML {pmlArea}\n ratio {pmlArea * 1.0f / domainArea} Saved {domainArea-pmlArea} cells");
#endif

            var decayBuffer = new ComputeBuffer(decayAll.Length, sizeof(float) * 4);
            decayBuffer.SetData(decayAll);
            _buffers["decay_all"] = decayBuffer;
        }

        private void InitializeMonitors()
        {
            monitors = new List<Monitor>(GameObject.FindObjectsOfType<Monitor>());

            Debug.Log($"Found {monitors.Count} monitors. Domain size is {domainSize}");

            if (0 == monitors.Count)
            {
                Debug.LogWarning($"No monitors found. No results will be saved.");
                return;
            }

            var numMonitorAddresses = 0;
            monitorAddresses = new List<int>(domainSize.x * domainSize.y);

            // var offset = 0;
            for (var i = 0; i < monitors.Count; ++i)
            {
                var monitor = monitors[i];
                monitor.offset = numMonitorAddresses;

                if (!monitor.isInitialized)
                    monitor.Initialize();
                monitorAddresses.AddRange(monitor.indices);
                Assert.IsTrue(monitor.isInitialized, $"Monitor {monitor.id} is not yet initialized!");

                numMonitorAddresses += monitor.indices.Count;
            }

            Debug.Log($"{nameof(monitorAddresses)}.{nameof(monitorAddresses.Count)} = {monitorAddresses.Count}");

            computeShader.SetInt("numMonitorAddresses", monitorAddresses.Count);

            var buffer = new ComputeBuffer(monitorAddresses.Count, sizeof(int));
            buffer.SetData(monitorAddresses);
            _buffers["monitorAddresses"] = buffer;

            var monitorValues = new ComputeBuffer(monitorAddresses.Count, sizeof(float));
            Helpers.ClearBuffer(monitorValues);
            _buffers["monitorValues"] = monitorValues;
        }


        [HideInInspector]
        public float monitorMin = float.MaxValue;
        [HideInInspector]
        public float monitorMax = float.MinValue;
        [HideInInspector]
        public float[] monitorValues;


        /// <summary>
        /// Read the monitorValues array back from the GPU and save the data, then clear the array
        /// </summary>
        private void ReadMonitors()
        {
            if (monitorValues.Length != monitorAddresses.Count)
                monitorValues = new float[monitorAddresses.Count];

            var buffer = _buffers["monitorValues"];
            buffer.GetData(monitorValues);

            foreach (var monitor in monitors)
                monitor.UpdateFromBuffer(monitorValues);

            onUpdateInstruments?.Invoke(monitors);
        }


        void OnDrawGizmos()
        {
            foreach (var source in sources)
            {
                var px = UI.Helpers.pixelToWorld((int)source.position.x, domainSize.y - (int)source.position.y);
                Gizmos.color = Color.white;
                Gizmos.DrawSphere(px, 0.2f);
            }
            {
                var lambda = sources[0].wavelength;
                var p = UI.Helpers.pixelToWorld(512, 0);
                Handles.Label(p, $"Lambda {lambda}");
            }

        }

    }

    /// <summary>
    /// Custom editor to provide some convenience buttons when
    /// running in the Unity editor.
    /// </summary>
    [CustomEditor(typeof(Simulation))]
    public class SimulationInspector : Editor
    {
        private Simulation sim => target as Simulation;
        public override void OnInspectorGUI()
        {
            if (GUILayout.Button("Reset simulation"))
            {
                sim.resetRequested = true;
                Debug.Log("Requesting simulation reset.");
            }

            if (sim.isPaused && GUILayout.Button("Resume"))
            {
                sim.isPaused = false;
            }
            else if (!sim.isPaused && GUILayout.Button("Pause"))
            {
                sim.isPaused = true;
            }

            if (GUILayout.Button("Next wavelength"))
            {
                sim.runNextWavelength();
            }

            base.OnInspectorGUI();
        }

    }
}