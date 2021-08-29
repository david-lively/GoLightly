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
                var halfSize = _prevDomainSize/2;
                var cam = Camera.main;


                var coords = new int[]
                {
                    -1, -1,
                    1,-1,
                    1,1,
                    -1,1
                };

                var positions = new Vector3[coords.Length];

                for (var i = 0; i < coords.Length; i+=2)
                {
                    var x = coords[i] * halfSize.x + 2*halfSize.x;
                    var y = coords[i + 1] * halfSize.y + halfSize.y;

                    var v = new Vector3(x, y, 0);
                    v = cam.ScreenToWorldPoint(v);
                    v.z = 0;
                    positions[i / 2] = v;
                }

                _renderer.SetPositions(positions);

            }
        }

        // Update is called once per frame
        void Update()
        {
            UpdateGeometry();
        }
    }
}
