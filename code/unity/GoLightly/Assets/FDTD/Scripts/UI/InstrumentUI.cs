using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using GoLightly;

namespace GoLightly.UI
{
    /// <summary>
    /// This blits the simulation visualization texture to the camera on this
    /// GameObject. 
    /// </summary>
    public class InstrumentUI : MonoBehaviour
    {
        public Simulation simulation;
        // Start is called before the first frame update
        void Start()
        {
            if (null == simulation)
                simulation = FindObjectOfType<Simulation>();
            Assert.IsNotNull(simulation, $"No {nameof(Simulation)} found. Output visualization will be disabled.");

            simulation.onUpdateInstruments = updateGraphData;
        }

        // Update is called once per frame
        void Update()
        {


        }

        void updateGraphData(List<Monitor> monitors)
        {
            if (0 >= monitors.Count)
                return;
        }

        public void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            Graphics.Blit(simulation.outputTexture, destination);
        }


    }
}
