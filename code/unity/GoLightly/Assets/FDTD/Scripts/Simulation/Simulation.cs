using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;


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
        private Dictionary<string, ComputeBuffer> _buffers = new Dictionary<string, ComputeBuffer>(StringComparer.OrdinalIgnoreCase);
        private Dictionary<string, int> _kernels = new Dictionary<string, int>();
        private Dictionary<string, Boundary> _boundaries = new Dictionary<string, Boundary>();

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
                _renderTexture.Create();
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
                var buffer = new ComputeBuffer(fieldBufferSize, sizeof(float));
                Helpers.ClearBuffer(buffer);
                _buffers["ez"] = buffer;
            }

            {
                var hx = new ComputeBuffer(fieldBufferSize, sizeof(float));
                Helpers.ClearBuffer(hx);
                _buffers["hx"] = hx;
            }

            {
                var hy = new ComputeBuffer(fieldBufferSize, sizeof(float));
                Helpers.ClearBuffer(hy);
                _buffers["hy"] = hy;
            }

            {
                var cb = new ComputeBuffer(fieldBufferSize, sizeof(float));
                Helpers.ClearBuffer(cb, parameters.cb);
                _buffers["cb"] = cb;
            }

            var kernelNames = new string[] {
                "CSUpdateVisualizerTexture"
                ,"CSUpdateEz"
                ,"CSUpdateHx"
                ,"CSUpdateHy"
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

            _renderTexture?.Release();
            _renderTexture = null;

            _isInitialized = false;
        }

        private void RunKernel(int kernelIndex)
        {
            var launchX = _renderTexture.width / 8;
            var launchY = _renderTexture.height / 8;

            RunKernel(kernelIndex, launchX, launchY);
        }

        private void RunKernel(int kernelIndex, int launchX, int launchY)
        {
            computeShader.SetBuffer(kernelIndex, "ez", _buffers["ez"]);
            computeShader.SetBuffer(kernelIndex, "hx", _buffers["hx"]);
            computeShader.SetBuffer(kernelIndex, "hy", _buffers["hy"]);
            computeShader.SetBuffer(kernelIndex, "sources", _buffers["sources"]);
            computeShader.SetConstantBuffer("Parameters", _buffers["Parameters"], 0, SimulationParameters.GetSize());
            foreach (var kvp in _boundaries)
            {
                kvp.Value.SetBuffers(computeShader, kernelIndex);
            }

            computeShader.Dispatch(kernelIndex, launchX, launchY, 1);
        }

        private void RunSimulationStep(uint steps = 1)
        {
            computeShader.SetVector("domainSize", new Vector2(domainSize.x, domainSize.y));
            computeShader.SetInt("numSources", sources.Count);

            var kernelNames = new string[] { "CSUpdateEz", "CSUpdateHx", "CSUpdateHy" };

            if (steps < 1)
                steps = 1;

            for (var j = 0; j < steps; ++j)
            {
                computeShader.SetFloat("time", Time.fixedTime);
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
            if (_kernels.TryGetValue(kernelName, out int kernelIndex))
            {                
                var kernel = _kernels["CSUpdateVisualizerTexture"];
                computeShader.SetTexture(kernel, "GameViewTexture", gameView);
                computeShader.SetTexture(kernel, "VisualizerTexture", _renderTexture);
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
            computeShader.SetFloat("time", Time.fixedTime);
            computeShader.SetVector("domainSize", new Vector4(domainSize.x, domainSize.y, 0, 0));
            computeShader.SetTexture(0, "VisualizerTexture", _renderTexture);

            RunSimulationStep(simulationTimeStepsPerFrame);
        }

        public void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            UpdateVisualizerTexture(source);
            Graphics.Blit(_renderTexture, destination);
            Debug.Log($"Texture sizes: {_renderTexture.width}x{_renderTexture.height}, {source.width}x{source.height}");
        }
        private void InitializeBoundaries(int layers = 10)
        {
            foreach (var boundary in _boundaries)
                boundary.Value?.Dispose();
            _boundaries.Clear();

            float sigmaMax = 1.0f;
            float sigmaOrder = 4.0f;
            float epsR = 1.0f;

            // PmlSigmaMax = 0.75f * (0.8f * (PmlSigmaOrder + 1) / (Dx * (float)pow(mu0 / (eps0 * epsR), 0.5f)));
            sigmaMax = 0.75f * (0.8f * (sigmaOrder + 1) / (parameters.dx * Mathf.Pow(parameters.mu0 / (parameters.eps0 * epsR), 0.5f)));
            Debug.Log($"PML layers {layers} sigmaMax {sigmaMax} order {sigmaOrder} dx {parameters.dx} dt {parameters.dt}");

            float xmin = layers * parameters.dx;
            float invLayersDx = 1.0f / xmin;

            var domainWidth = domainSize.x;
            var domainHeight = domainSize.y;

            var ezxDecay = new float[domainWidth];
            var ezyDecay = new float[domainHeight];
            var hyxDecay = new float[domainWidth];
            var hxyDecay = new float[domainHeight];


            for (var i = 0; i < layers; ++i)
            {
                var elength = i * parameters.dx;
                var hlength = (i + 0.5f) * parameters.dx;

                var esigma = sigmaMax * Mathf.Pow(Mathf.Abs(elength - xmin) * invLayersDx, sigmaOrder);
                var hsigma = sigmaMax * Mathf.Pow(Mathf.Abs(hlength - xmin) * invLayersDx, sigmaOrder);

                var edecay = Mathf.Exp(-1 * (parameters.dt * esigma) / parameters.eps0);
                // var eAmp = edecay - 1;
                var hdecay = Mathf.Exp(-1 * (parameters.dt * hsigma) / parameters.mu0);
                // var hAmp = hdecay - 1;

                ezxDecay[i] = edecay;
                ezxDecay[ezxDecay.Length - i - 1] = edecay;

                ezyDecay[i] = edecay;
                ezyDecay[ezyDecay.Length - i - 1] = edecay;

                hxyDecay[i] = hdecay;
                hxyDecay[hxyDecay.Length - i - 1] = hdecay;

                hyxDecay[i] = hdecay;
                hyxDecay[hyxDecay.Length - i - 1] = hdecay;
            }

            var ezx = new Boundary { name = "ezx" };
            // ezx.name = "ezx";
            ezx.SetData(domainWidth, domainHeight, ezxDecay);
            _boundaries.Add("ezx", ezx);

            var ezy = new Boundary { name = "ezy" };
            ezy.SetData(domainWidth, domainHeight, ezyDecay);
            _boundaries.Add("ezy", ezy);

            var ezA = "Emma Stone";

            var hxy = new Boundary { name = "hxy" };
            hxy.SetData(domainWidth, domainHeight, hxyDecay);
            _boundaries.Add("hxy", hxy);

            var hyx = new Boundary { name = "hyx" };
            hyx.SetData(domainWidth, domainHeight, hyxDecay);
            _boundaries.Add("hyx", hyx);
        }
    }
}