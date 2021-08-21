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
        public float frequency;

        public readonly float ca;
        public float cb;

        public readonly float da;
        public float db;

        #endregion

        public static int GetSize()
        {
            return sizeof(float) * 10;
        }

        public static SimulationParameters Create()
        {
            var p = new SimulationParameters();
            p.eps0 = 1.0f;
            p.mu0 = 1.0f;
            p.lambda = 1.0f;

            p.dx = p.lambda / 10.0f;
            // nyquist 
            p.dt = p.dx / Mathf.Sqrt(2.0f) * 0.95f;

            p.cb = p.dt / (p.eps0 * p.dx);
            p.db = p.dt / (p.mu0 * p.dx);

            p.frequency = 1.0f / p.lambda;

            return p;

        }

    }
}