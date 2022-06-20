using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace GoLightly
{

    public class MonitorProvider : MonoBehaviour
    {
        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {

        }

        void OnInspectorGUI()
        {
            if (GUILayout.Button("Hello World"))
            {
                Debug.Log("Hello world!");
            }

        }



    }

}