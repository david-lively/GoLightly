using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.Mathematics;

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
    public partial class Simulation : MonoBehaviour
    {
        private RenderTexture _renderTexture;
        public ComputeShader computeShader;
        public Vector2Int domainSize = new Vector2Int(2048, 1024);

        public SimulationParameters parameters = SimulationParameters.Create(1.0f);

        [Range(1, 200)]
        public float contrast = 80;
        public int timeStep = 0;

        [Range(0, 100)]
        public float psiContrast = 0;

        public uint simulationTimeStepsPerFrame = 1;

        readonly float[] e_decay = new float[] {
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

        readonly float[] h_decay = new float[]  {
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
                1
                };

        private readonly Dictionary<string, ComputeBuffer> _buffers = new Dictionary<string, ComputeBuffer>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, int> _kernels = new Dictionary<string, int>();
        private readonly Dictionary<string, Boundary> _boundaries = new Dictionary<string, Boundary>();

        [System.Serializable]
        public struct Source
        {
            public Vector3 position;
            public float amplitude;
            public float wavelength;
            public float maxLife;
            public uint _enabled;

            public static int GetSize()
            {
                return (3 * sizeof(float)) + (3 * sizeof(float)) + sizeof(uint);
            }
        }


        public List<Source> sources = new List<Source>();

        // Start is called before the first frame update
        void Start()
        {
            if (null == _renderTexture)
            {
                Debug.Log("Creating render texture");
                _renderTexture = new RenderTexture(domainSize.x, domainSize.y, 24);
                _renderTexture.enableRandomWrite = true;

                var textureWasCreated = _renderTexture.Create();
                Assert.IsTrue(textureWasCreated, "Could not create visualizer texture.");
            }
            InitComputeResources();
        }

        private bool _isInitialized = false;

        private void InitComputeResources()
        {
            if (_isInitialized)
                return;

            if (null == _renderTexture)
            {
                Debug.Log("Creating render texture");
                _renderTexture = new RenderTexture(domainSize.x, domainSize.y, 24);
                _renderTexture.enableRandomWrite = true;
                _renderTexture.Create();
            }

            {
                var sourcesBuffer = new ComputeBuffer(sources.Count, Source.GetSize());
                sourcesBuffer.SetData(sources);
                computeShader.SetInt("numSources", sources.Count);
                _buffers["sources"] = sourcesBuffer;
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
                SetMaterials(cbData);
                var cb = new ComputeBuffer(fieldBufferSize, sizeof(float));
                cb.SetData(cbData);
                //Helpers.ClearBuffer(cb, parameters.cb);
                _buffers["cb"] = cb;
            }

            var kernelNames = new string[] {
                "CSUpdateVisualizerTexture"
                ,"CSUpdateEz"
                ,"CSUpdateHFields"
                ,"CSUpdateSources"
            };

            foreach (var name in kernelNames)
            {
                var kernelIndex = computeShader.FindKernel(name);
                Assert.IsTrue(kernelIndex >= 0, $"Compute shader kernel `{name}` not found!");

                _kernels[name] = kernelIndex;
            }

            InitializeBoundaries(parameters.pmlLayers);

            _isInitialized = true;
        }

        private void OnDestroy()
        {
            Debug.Log($"Disposing {_buffers.Count} compute buffers");

            foreach (var kvp in _buffers)
            {
                kvp.Value?.Dispose();
            }
            _buffers.Clear();

            Debug.Log($"Disposing {_boundaries.Count} boundary compute buffers");

            foreach (var kvp in _boundaries)
            {
                kvp.Value?.Dispose();
            }

            _boundaries.Clear();
            _renderTexture.Release();
            _renderTexture = null;

            _isInitialized = false;
        }

        private void RunKernel(int kernelIndex)
        {
            var launchX = _renderTexture.width / 32;
            var launchY = _renderTexture.height / 32;

            RunKernel(kernelIndex, launchX, launchY);
        }

        private void RunKernel(int kernelIndex, int launchX, int launchY)
        {
            computeShader.SetBuffer(kernelIndex, "ez", _buffers["ez"]);
            computeShader.SetBuffer(kernelIndex, "hx", _buffers["hx"]);
            computeShader.SetBuffer(kernelIndex, "hy", _buffers["hy"]);
            computeShader.SetBuffer(kernelIndex, "sources", _buffers["sources"]);
            computeShader.SetBuffer(kernelIndex, "decay_all", _buffers["decay_all"]);
            computeShader.SetBuffer(kernelIndex, "cb", _buffers["cb"]);
            computeShader.SetConstantBuffer("Parameters", _buffers["Parameters"], 0, SimulationParameters.GetSize());
            computeShader.SetBuffer(kernelIndex, "psi_all", _buffers["psi_all"]);

            computeShader.Dispatch(kernelIndex, launchX, launchY, 1);
        }

        private void RunSimulationStep(uint steps = 1)
        {
            computeShader.SetFloat("time", Time.fixedTime);
            computeShader.SetVector("domainSize", new Vector4(domainSize.x, domainSize.y, 0, 0));
            computeShader.SetTexture(0, "VisualizerTexture", _renderTexture);

            computeShader.SetVector("domainSize", new Vector2(domainSize.x, domainSize.y));
            computeShader.SetInt("numSources", sources.Count);

            var kernelNames = new string[] { "CSUpdateHFields", "CSUpdateEz" };

            if (steps < 1)
                steps = 1;

            for (var j = 0; j < steps; ++j)
            {
                //computeShader.SetFloat("time", Time.fixedTime);
                computeShader.SetInt("TimeStep", timeStep);

                RunKernel(_kernels["CSUpdateSources"], 1, 1);

                for (var i = 0; i < kernelNames.Length; ++i)
                {
                    RunKernel(_kernels[kernelNames[i]]);
                }

                ++timeStep;
            }
        }

        private void UpdateVisualizerTexture(RenderTexture gameView)
        {
            var kernelName = "CSUpdateVisualizerTexture";
            if (_kernels.TryGetValue(kernelName, out var kernelIndex))
            {
                computeShader.SetTexture(kernelIndex, "GameViewTexture", gameView);
                computeShader.SetTexture(kernelIndex, "VisualizerTexture", _renderTexture);
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
            RunSimulationStep(simulationTimeStepsPerFrame);
        }

        public void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            UpdateVisualizerTexture(source);
            Graphics.Blit(_renderTexture, destination);
        }

        private void drawPmlBox(uint2 topLeft, uint2 dimensions, float4[] decayAll)
        {
            const int layers = 12;

            var width = dimensions.x;
            var height = dimensions.y;

            var ed = new float[layers];
            var hd = new float[layers];

            // reverse the e and h decay arrays
            for (var i = 0; i < e_decay.Length; ++i)
            {
                ed[i] = e_decay[e_decay.Length - 1 - i];
                hd[i] = h_decay[h_decay.Length - 1 - i];
                //ed[i] = e_decay[i];
                //hd[i] = h_decay[i];
            }

            /*
            float4 results:
            x -> ezx decay
            y -> ezy decay
            z -> hyx decay
            w -> hxy decay
            */
            for (var j = 0; j < height - 1; ++j)
            {
                for (var i = 0; i < width - 1; ++i)
                {
                    //var v = new float4(1, 1, 1, 1);
                    var v = new float4(0, 0, 0, 0);

                    if (i < layers)
                    {
                        v.x = ed[i];
                        v.z = hd[i];
                    }
                    else if (i >= width - layers)
                    {
                        v.x = ed[width - i - 1];
                        v.z = hd[width - i - 1];
                    }

                    // ezy & hxy
                    if (j < layers)
                    {
                        v.y = ed[j];
                        v.w = hd[j];
                    }
                    else if (j >= height - layers)
                    {
                        v.y = ed[height - j - 1];
                        v.w = hd[height - j - 1];
                    }

                    {
                        var x = i + topLeft.x;
                        var y = j + topLeft.y;
                        decayAll[y * domainSize.x + x] = v;
                    }

                }
            }

        }

        private void InitializeBoundaries(int _)
        {
            const int layers = 12;
            
            foreach (var boundary in _boundaries)
                boundary.Value?.Dispose();
            _boundaries.Clear();

            var domainWidth = domainSize.x;
            var domainHeight = domainSize.y;

            /// unified PML psi buffer
            var psiBuffer = new ComputeBuffer(domainWidth * domainHeight, sizeof(float) * 4);
            _buffers["psi_all"] = psiBuffer;

            /// unified PML decay buffer.
            var decayAll = new float4[domainWidth * domainHeight];
            /*
            float4 results:
            x -> ezx decay
            y -> ezy decay
            z -> hyx decay
            w -> hxy decay
            */
            for (var j = 0; j < domainHeight - 1; ++j)
            {
                for (var i = 0; i < domainWidth - 1; ++i)
                {
                    var v = new float4(1, 1, 1, 1);

                    /// ezx & hyx
                    if (i < layers)
                    {
                        v.x = e_decay[i];
                        v.z = h_decay[i];
                    }
                    else if (i >= domainWidth - layers)
                    {
                        v.x = e_decay[domainWidth - i - 1];
                        v.z = h_decay[domainWidth - i - 2];
                    }

                    // ezy & hxy
                    if (j < layers)
                    {
                        v.y = e_decay[j];
                        v.w = h_decay[j];
                    }
                    else if (j >= domainHeight - layers)
                    {
                        v.y = e_decay[domainHeight - j - 1];
                        v.w = h_decay[domainHeight - j - 2];
                    }

                    decayAll[j * domainWidth + i] = v;
                }
            }

            drawPmlBox(new uint2(1024-190, 512-190), new uint2(380, 380), decayAll);
            drawPmlBox(new uint2(20, 240), new uint2(800, 700), decayAll);
            //drawPmlBox(new uint2(100, 100), new uint2(1900, 100), decayAll);

            /*
            x = ezxDecay
            y = ezyDecay
            z = hxyDecay
            w = hyxDecay
            */
            var decayBuffer = new ComputeBuffer(decayAll.Length, sizeof(float) * 4);
            decayBuffer.SetData(decayAll);
            _buffers["decay_all"] = decayBuffer;
        }

        void lineGuide(float[] cbData, int top)
        {
            var scalar = parameters.dt / parameters.dx;
            var coreMaterial = scalar * 1.0f / 3;
            var cladMaterial = parameters.dt / parameters.dx * 1.0f / 9;

            var middleY = domainSize.y / 2;
            //var width = 20;
            //var top = middleY - width / 2;

            var coreLayers = 3;
            var cladLayers = 6;
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

        void wgm(float[] cbData, float radius, float width)
        {
            var center = new uint2((uint)(domainSize.x / 2), (uint)domainSize.y / 2);
            var m = (parameters.dt / parameters.dx * 1 / 9.0f);
            ModelProvider.Cylinder(center, radius, width, domainSize.x, m, cbData);
        }

        void SetMaterials(float[] cbData)
        {
            //return;
            //demoLineGuide(cbData);
            //demoGuide2(cbData);

            // set source position to 256,219
            lineGuide(cbData, domainSize.y / 2 - 300);

            wgm(cbData, 282, 5);
        }
    }
}