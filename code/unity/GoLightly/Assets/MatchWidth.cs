using System.Collections;
using System.Collections.Generic;
using UnityEngine;


// [ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class MatchWidth : MonoBehaviour {

    // Set this to the in-world distance between the left & right edges of your scene.
    public float sceneWidth = 10;
    void Start() {
    }

    // Adjust the camera's height so the desired scene width fits in view
    // even if the screen/window size changes dynamically.
    void Update() {
        var sim = GetComponent<GoLightly.Simulation>();
        float unitsPerPixel = sim.domainSize.x / Screen.width;

        float desiredHalfHeight = 0.5f * unitsPerPixel * Screen.height;

        GetComponent<Camera>().orthographicSize = desiredHalfHeight;
    }
}