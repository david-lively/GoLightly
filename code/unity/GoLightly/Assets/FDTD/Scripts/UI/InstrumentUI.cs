using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace GoLightly
{
    /// <summary>
    /// This blits the simulation visualization texture to the camera on this
    /// GameObject. 
    /// </summary>
    public class InstrumentUI : MonoBehaviour
    {
        public Simulation simulation;
        public Graph graph;
        // Start is called before the first frame update
        void Start()
        {
            if (null == simulation)
                simulation = FindObjectOfType<Simulation>();
            Assert.IsNotNull(simulation, $"No {nameof(Simulation)} found. Output visualization will be disabled.");

            simulation.onUpdateInstruments = updateGraphData;


            if (null == graph)
                graph = GameObject.FindObjectOfType<Graph>();
            Assert.IsNotNull(graph, $"No {nameof(Graph)} component found.");
        }

        // Update is called once per frame
        void Update()
        {


        }

        void updateGraphData(List<Monitor> monitors)
        {
            if (0 >= monitors.Count)
                return;

            graph?.values.Add(monitors[0].currentValue);
        }

        public void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            Graphics.Blit(simulation.outputTexture, destination);
        }


    }
}
