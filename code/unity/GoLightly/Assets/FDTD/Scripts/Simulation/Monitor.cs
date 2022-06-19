using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace GoLightly
{
    public class Monitor : MonoBehaviour
    {
        public Vector2Int topLeft;
        public Vector2Int bottomRight;
        public int id;
        public int offset;
        public int count;

        public List<float> history = new List<float>();
        public int maxHistoryLength = 128;

        /// <summary>
        /// Cell offset for each monitor "pixel" location. These correspond to y * width + x values in the domain.
        /// </summary> 
        int frameCount;
        int currentFrame;

        public float maxValue = float.MinValue;
        public float minValue = float.MaxValue;
        List<float> magnitudeHistory = new List<float>();

        float rmsMinValue = float.MaxValue;
        float rmsMaxValue = float.MinValue;
        List<float> rmsHistory = new List<float>();

        public bool isInitialized { get; private set; }

        /// <summary>
        /// list of domain coordinates that this monitor occupies
        /// </summary>
        [HideInInspector]
        public List<int> indices;

        public void Start()
        {
            var simulation = FindObjectOfType<Simulation>();

            var domainSize = simulation.domainSize;
            var domainMax = domainSize - Vector2Int.one;

            var min = Vector2Int.Max(Vector2Int.Min(topLeft, bottomRight), Vector2Int.zero);
            var max = Vector2Int.Min(Vector2Int.Max(topLeft, bottomRight), domainMax);

            var numElements = (max.x - min.x + 1) * (max.y - min.y + 1);

            indices = new List<int>(numElements);

            for (var j = min.y; j <= max.y; ++j)
            {
                for (var i = min.x; i <= max.x; ++i)
                {
                    var offset = j * domainSize.y + i;

                    indices.Add(offset);
                }
            }

            Debug.Log($"Monitor {id} has {indices.Count} elements");
            isInitialized = true;

            history = new List<float>(maxHistoryLength);
        }

        private int _historyIndex = 0;
        public void UpdateFromBuffer(float[] buffer)
        {
            if (null == indices || indices.Count == 0)
                return;

            var sum = 0f;

            for (var i = offset; i < offset + indices.Count; ++i)
            {
                sum += buffer[i];
            }

            var average = sum / indices.Count;

            /// history is a ring buffer. 
            if (history.Count < maxHistoryLength)
                history.Add(average);
            else
            {
                _historyIndex %= history.Count;
                history[_historyIndex] = average;
            }

            minValue = Mathf.Min(minValue, average);
            maxValue = Mathf.Max(maxValue, average);
            ++_historyIndex;
        }


        public static Vector3 pixelToWorld(int x, int y)
        {
            var v = new Vector3(x, y, 0);
            var s = Camera.main.ScreenToWorldPoint(v);
            s.z = 0;
            return s;
        }

        public static void GizmoRect(Vector2Int min, Vector2Int max)
        {
            var va = Vector2Int.Min(min, max);
            var vb = Vector2Int.Max(min, max);

            var nw = pixelToWorld(va.x, va.y);
            var se = pixelToWorld(vb.x, vb.y);
            var ne = pixelToWorld(vb.x, va.y);
            var sw = pixelToWorld(va.x, vb.y);

            // var saveColor = Gizmos.color;

            Gizmos.DrawLine(nw, ne);
            Gizmos.DrawLine(ne, se);
            Gizmos.DrawLine(se, sw);
            Gizmos.DrawLine(sw, nw);

            // Gizmos.color = saveColor;
        }

        public void OnDrawGizmos()
        {
            Gizmos.color = Color.red;
            GizmoRect(topLeft, bottomRight);
            // var cam = Camera.main;

            // var tl = pixelToWorld(10, 10);
            // var br = pixelToWorld(2048 - 10, 1024 - 10);
            // Gizmos.color = Color.red;
            // Gizmos.DrawLine(tl, br);
        }
    }
}
