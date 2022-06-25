using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace GoLightly
{
    struct BatchRunMonitorResult
    {
        public float minValue;
        public float maxValue;
        public float minRMS;
        public float maxRMS;
    }
    
    class BatchRunResult
    {
        public Dictionary<string, BatchRunMonitorResult> monitorResults = new Dictionary<string, BatchRunMonitorResult>();
        public float lambda;
        public int numTimeSteps;
    }

    public class BatchRun : MonoBehaviour    
    {
        private float _startLambda = 3.125f * 2;
        public float lambda = 3.125f * 2;
        public float lastLambda;
        public float lambdaSteps = 10;
        public float lambdaDelta = 0.1f;
        public int timeStepsPerRun = 2000;

        public Simulation simulation;

        private Monitor[] _monitors;

        // Start is called before the first frame update
        void Start()
        {
            if (null == simulation)
            {
                simulation = GameObject.FindObjectOfType<Simulation>();
                Assert.IsNotNull(simulation, "No Simulation component found!");
                simulation.sweepWavelengthDelta = lambdaDelta;
            }
            Debug.Log($"Found {simulation.sources.Count} sources.");
            Assert.IsTrue(simulation.sources.Count == 1, "Exactly one Source is required");
            Assert.IsTrue(lambdaDelta > 0, "Lambda delta must be positive.");
            var source = simulation.sources[0];
            lambda = source.wavelength;
            _startLambda = lambda;
            lastLambda = _startLambda + lambdaDelta * lambdaSteps;
            Debug.Log($"Last lambda = {lastLambda}");

            _monitors = GameObject.FindObjectsOfType<Monitor>();
        }

        // Update is called once per frame
        void Update()
        {
            if (simulation.timeStep >= timeStepsPerRun && !simulation.isPaused)
            {
                saveCurrentResults();
                lambda += lambdaDelta;
                if (lambda <= lastLambda)
                {
                    simulation.setSingleSourceWavelengthAndReset(lambda);
                    Debug.Log($"Starting run with lambda {lambda}");
                }
                else
                {
                    Debug.Log($"Batch run complete.");
                    simulation.isPaused = true;
                }
            }

        }

        private void saveCurrentResults()
        {
            var result = new BatchRunResult();
            result.lambda = this.lambda;
            result.numTimeSteps = timeStepsPerRun;
            foreach(var monitor in _monitors)
            {
                var monitorResult = new BatchRunMonitorResult();
                monitorResult.maxRMS = monitor.rmsMaxValue;
                monitorResult.minRMS = monitor.rmsMinValue;
                monitorResult.maxValue = monitor.maxValue;
                monitorResult.minValue = monitor.minValue;
                result.monitorResults[monitor.friendlyName] = monitorResult;
            }

        }
    }
}