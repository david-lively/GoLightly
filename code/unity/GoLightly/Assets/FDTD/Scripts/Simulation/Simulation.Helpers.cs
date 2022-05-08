using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace GoLightly
{
    public partial class Simulation
    {
        public static class Helpers
        {

            public static void ClearArray(ref float[] arr, float value)
            {
                for (var i = 0; i < arr.Length; ++i)
                    arr[i] = value;
            }

            public static void ClearBuffer(ComputeBuffer buffer, float value = 0.0f)
            {
                var data = new float[buffer.count];
                if (0.0f != value)
                    ClearArray(ref data, value);
                buffer.SetData(data);
            }

            public static void SetArray<T>(ref T[] arr, T value)
            {
                for (var i = 0; i < arr.Length; ++i)
                    arr[i] = value;
            }

        }
    }
}