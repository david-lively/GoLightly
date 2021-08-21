using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace GoLightly
{
    public class Boundary : IDisposable
    {
        public string name;
        public ComputeBuffer psi;
        public ComputeBuffer decay;

        public Boundary()
        {

        }

        public Boundary(string name, int size)
        {
            this.name = name;
            decay = new ComputeBuffer(size, sizeof(float));
            psi = new ComputeBuffer(size, sizeof(float));
        }

        public void SetBuffers(ComputeShader shader, int kernelIndex)
        {
            shader.SetBuffer(kernelIndex, name + "_decay", decay);
            shader.SetBuffer(kernelIndex, name + "_psi", psi);
        }

        public void SetData(int domainWidth, int domainHeight, float[] decayData)
        {
            if (null == decay || decay.count != decayData.Length)
            {
                decay?.Dispose();
                decay = new ComputeBuffer(decayData.Length, sizeof(float));
            }
            decay.SetData(decayData);

            var psiLength = domainWidth * domainHeight;
            if (null == psi || psi.count != psiLength)
            {
                psi?.Dispose();
                psi = new ComputeBuffer(psiLength, sizeof(float));
            }
            Simulation.Helpers.ClearBuffer(psi);
        }

        public void Clear()
        {
            if (null != psi)
                Simulation.Helpers.ClearBuffer(psi);
        }

        #region IDisposable

        private bool _disposed = false;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    psi?.Dispose();
                    psi = null;
                    decay?.Dispose();
                    decay = null;
                }
                _disposed = true;
            }

        }

        #endregion

    }

}