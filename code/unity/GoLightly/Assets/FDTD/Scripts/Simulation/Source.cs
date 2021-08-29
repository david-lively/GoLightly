using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace GoLightly
{
    /// <summary>
    /// Source in the simulation. Note that wavelength must match the lambda used
    /// to calculate dx/dt in the simulator. 
    /// </summary>
    public class Source : MonoBehaviour
    {
        public float amplitude = 1.0f;
        public float wavelength = 1.0f;

        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {

        }
    }
}
