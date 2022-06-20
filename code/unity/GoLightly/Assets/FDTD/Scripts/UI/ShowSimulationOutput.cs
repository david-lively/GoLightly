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
    public class ShowSimulationOutput : MonoBehaviour
    {
        public Simulation simulation;
        // Start is called before the first frame update
        void Start()
        {
            simulation = FindObjectOfType<Simulation>();
            Assert.IsNotNull(simulation, $"No {nameof(Simulation)} found. Output visualization will be disabled.");
        }

        // Update is called once per frame
        void Update()
        {

        }

        public void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            Graphics.Blit(simulation.outputTexture, destination);
        }        
    }
}
