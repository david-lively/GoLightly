using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace GoLightly
{
    public struct SimulationParameters
    {
        #region FDTD parameters
        public float dt;
        public float dx;
        public float eps0;
        public float mu0;

        public float lambda;

        public float ca;
        public float cb;

        public float da;
        public float db;
        public int pmlLayers;

        #endregion

        public static int GetSize()
        {            
            unsafe
            {
                return sizeof(SimulationParameters);
            }
        }

        public static SimulationParameters Create(float lambda = 1.0f)
        {
            var p = new SimulationParameters();
            p.eps0 = 1.0f;
            p.mu0 = 1.0f;
            p.lambda = lambda;

            p.dx = p.lambda / 10.0f;
            // nyquist 
            p.dt = p.dx / Mathf.Sqrt(2.0f) * 0.95f;

            p.ca = 1.0f;
            p.cb = p.dt / (p.eps0 * p.dx);

            p.da = 1.0f;
            p.db = p.dt / (p.mu0 * p.dx);

            p.pmlLayers = 10;

            return p;
        }

    }
}