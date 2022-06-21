using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEditor;

namespace GoLightly
{

    public class MonitorProvider : MonoBehaviour
    {
        public bool removeExistingMonitors;
        public Simulation simulation;
        public readonly string msg = "Hello world";
        public int padding = 16;
        public int boxWidth = 512;
        public int thickness = 8;       

        // Start is called before the first frame update
        void Start()
        {
            if (null == simulation)
                simulation = GameObject.FindObjectOfType<Simulation>();

            Assert.IsNotNull(simulation, "Could not find Simulation object.");
        }

        // Update is called once per frame
        void Update()
        {

        }

        public void DestroyMonitors()
        {
            var monitors = GameObject.FindObjectsOfType<Monitor>();
            Debug.Log($"Found {monitors.Length} monitors.");

            for (var i = 0; i < monitors.Length; ++i)
            {
                Undo.DestroyObjectImmediate(monitors[i]);
            }

            Debug.Log($"Destroyed {monitors.Length} monitors.");

            Monitor.nextId = 0;
        }

        /// <summary>
        /// Generate a "box" of monitors around the source.
        /// </summary>
        internal void GenerateSquareArray()
        {
            Assert.IsNotNull(simulation, "No simulation object specified.");

            if (removeExistingMonitors)
                DestroyMonitors();

            var center = simulation.domainSize / 2;
            var halfWidth = boxWidth / 2;
            var west = -halfWidth;
            var east = halfWidth;
            var north = halfWidth;
            var south = -halfWidth;

            var rects = new int[] {
                //top
                west+padding,north+thickness,east-padding,north,
                // right
                east,north-padding,east+thickness,south+padding,
                // bottom
                west+padding, south-thickness, east-padding, south,
                // left
                west-thickness,north-padding, west, south+padding, 
            };

            var monitors = new List<Monitor>();
            for (var i = 0; i < rects.Length; i += 4)
            {
                // var monitor = simulation.gameObject.AddComponent<Monitor>();
                var monitor = this.gameObject.AddComponent<Monitor>();
                monitor.topLeft = center + new Vector2Int(rects[i], rects[i + 1]);
                monitor.bottomRight = center + new Vector2Int(rects[i + 2], rects[i + 3]);
                monitor.friendlyName = $"monitor_{monitor.id}";
                monitors.Add(monitor);
            }

#if false
            {
                /// test code - shifts the eastern monitor 100 cells to the right.
                var offset = 100;
                var m = monitors[1];
                
                var v = m.topLeft;
                v.x += offset;
                m.topLeft = v;
                
                v = m.bottomRight;
                v.x += offset;
                m.bottomRight = v;
            }
#endif
        }


    }

    [CustomEditor(typeof(MonitorProvider))]
    public class MonitorProviderEditor : Editor
    {
        public Simulation simulation;

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var comp = target as MonitorProvider;

            if (GUILayout.Button("Generate monitors"))
            {
                comp.GenerateSquareArray();
            }

            if (GUILayout.Button("Destroy all monitors"))
            {
                comp.DestroyMonitors();
            }


        }

    }

}