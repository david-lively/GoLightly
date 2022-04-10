using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace GoLightly
{


    [ExecuteInEditMode]
    public class Domain : MonoBehaviour
    {
        // Start is called before the first frame update
        private Simulation _simulation;
        private LineRenderer _renderer;

        public string SimulationObjectName;
        private Vector2Int _prevDomainSize = Vector2Int.zero;
        void Start()
        {
            _simulation = FindObjectOfType<GoLightly.Simulation>(true);
            Assert.IsNotNull(_simulation, "GoLightly.Simulation component not found in the scene.");
            SimulationObjectName = _simulation.gameObject.name;

            _renderer = GetComponent<LineRenderer>();
            Assert.IsNotNull(_renderer, "No line renderer found on this gameObject.");

            _prevDomainSize = _simulation.domainSize;

        }

        private void UpdateGeometry()
        {
            // if (_prevDomainSize != _simulation.domainSize)
            {
                _prevDomainSize = _simulation.domainSize;
                var size = new Vector2(_simulation.domainSize.x, _simulation.domainSize.y) / 2;

                if (_simulation.domainSize.x > _simulation.domainSize.y)
                {
                    size.y /= size.x;
                    size.x = 1;
                }
                else
                {
                    size.x /= size.y;
                    size.y = 1;
                }

                var cameraRect = Camera.main.rect;

                // if (aspect > 1)
                // {
                //     size.x *= aspect;
                // }
                // else
                //     size.y *= aspect;

                var coords = new int[]
                {
                    -1, -1,
                    1,-1,
                    1,1,
                    -1,1
                };

                throw new System.Exception();

                /*
                instead of fuzzing about with this, just set the camera orthographic size to whichever the larger dimension is (normalized)
                ?? for 2048x1024 camera, orthoSize would be 2. 
                ... still have to scale the rect coordinates to fit in a unit rectangle!!! 
                Note that ortho size is *vertical*, so for a width > height situation, it would be < 1. 
                */


                if (size.y < size.x)
                {
                    var s = size.x * 1.0f / size.y;
                    size *= s;
                }

                if (Camera.main.pixelWidth > Camera.main.pixelHeight)
                {
                    var cam = Camera.main;

                    size.y *= cameraRect.width * 1.0f / cameraRect.height;
                    Debug.Log($"Camera: [ {cam.pixelWidth}, {cam.pixelHeight}, size {size.x}x{size.y}\n");
                }

                var positions = new Vector3[coords.Length];

                for (var i = 0; i < coords.Length; i+=2)
                {
                    var x = coords[i] * size.x;
                    var y = coords[i + 1] * size.y;


                    var v = new Vector3(x, y, 0);
                    v.z = 0;
                    positions[i / 2] = v;
                }

                _renderer.SetPositions(positions);

                {
                    // var cam = Camera.main;
                    // cam.rect = new Rect(0, 0, 1, 1);
                    // cam.orthographicSize = 1;//halfSize.x;
                    // // Adjust the camera's height so the desired scene width fits in view
                    // // even if the screen/window size changes dynamically.
                    // void Update() {
                    //     var sim = GetComponent<GoLightly.Simulation>();
                    //     float unitsPerPixel = sim.domainSize.x / Screen.width;

                    //     float desiredHalfHeight = 0.5f * unitsPerPixel * Screen.height;

                    //     GetComponent<Camera>().orthographicSize = desiredHalfHeight;
                    // }

                }

            }
        }

        // Update is called once per frame
        void Update()
        {
            UpdateGeometry();
        }
    }
}
