using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using GoLightly.UI;

namespace GoLightly
{

    [DisallowMultipleComponent]
    public class ChartManager : MonoBehaviour
    {
        public float chartScale = 200;
        public float chartOffset = 50;
        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {

        }
    }

    [CustomEditor(typeof(ChartManager))]
    public class ChartManagerInspector : Editor
    {
        private ChartManager manager => target as ChartManager;

        public override void OnInspectorGUI()
        {
            if (GUILayout.Button("Set chart properties"))
            {
                var charts = GameObject.FindObjectsOfType<Chart>();

                foreach (var chart in charts)
                {
                    chart.yScale = manager.chartScale;
                    chart.offset = manager.chartOffset;
                }

            }

            base.OnInspectorGUI();
        }

    }
}