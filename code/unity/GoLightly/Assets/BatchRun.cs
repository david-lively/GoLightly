using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEditor;

namespace GoLightly
{
    public class BatchRun : MonoBehaviour
    {
        public float startLambda;
        public float endLambda;
        public float lambdaSteps = 10;
        private float _lambdaDelta = 0.1f;
        public int timeStepsPerRun = 2000;
        public float lambdaScalar = 1;

        public Simulation simulation;

        private Monitor[] _monitors;

        private StringBuilder _results = new StringBuilder();
        private float _currentLambda;


        // Start is called before the first frame update
        void Start()
        {
            if (null == simulation)
            {
                simulation = GameObject.FindObjectOfType<Simulation>();
                Assert.IsNotNull(simulation, "No Simulation component found!");
                simulation.sweepWavelengthDelta = _lambdaDelta;
            }

            Debug.Log($"Found {simulation.sources.Count} sources.");
            Assert.IsTrue(simulation.sources.Count == 1, "Exactly one Source is required");
            Assert.IsTrue(_lambdaDelta > 0, "Lambda delta must be positive.");

            var source = simulation.sources[0];

            _lambdaDelta = (endLambda - startLambda) * 1.0f / lambdaSteps;
            Debug.Log($"Lambda Delta {_lambdaDelta}");

            _monitors = GameObject.FindObjectsOfType<Monitor>();
            _results.Clear();

            _currentLambda = startLambda;

            /// generate the CSV header string
            var header = "Lambda,LambdaNM,timeSteps,maxRMS,maxValue";
            _results.AppendLine(header);
        }

        // Update is called once per frame
        void Update()
        {
            if (simulation.timeStep >= timeStepsPerRun && !simulation.isPaused)
            {
                captureResult();
                _currentLambda += _lambdaDelta;
                if (_currentLambda <= endLambda)
                {
                    simulation.SetSingleSourceWavelengthAndReset(_currentLambda);
                    Debug.Log($"Starting run with lambda {_currentLambda}");                    
                }
                else
                {
                    Debug.Log($"Batch run complete.");
                    simulation.isPaused = true;
                    saveResultsToFile();
                }
            }

        }

        private void captureResult()
        {
            var rmsSum = 0f;
            var valSum = 0f;
            foreach(var monitor in _monitors)
            {
                rmsSum += monitor.rmsMaxValue;
                valSum += monitor.maxValue;
            }
            var line = $"{_currentLambda},{_currentLambda * lambdaScalar},{timeStepsPerRun},{rmsSum},{valSum}";

            _results.AppendLine(line);
        }

        void saveResultsToFile()
        {
            var content = _results.ToString();
            // The target file path e.g.
            var folder = Application.streamingAssetsPath;

            if (!Directory.Exists(folder))
                Directory.CreateDirectory(folder);

            var filePath = Path.Combine(folder, "export.csv");

            // using (var writer = new StreamWriter(filePath, false))
            // {
            //     writer.Write(content);
            // }

            File.WriteAllText(filePath, content);

            // Or just
            //File.WriteAllText(content);

            Debug.Log($"CSV file written to \"{filePath}\"");

            AssetDatabase.Refresh();
        }



    }
}