using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

/// <summary>
/// https://stackoverflow.com/questions/37137110/creating-graphs-in-unity
/// Draws a basic oscilloscope type graph in a GUI.Window()
/// Michael Hutton May 2020
/// This is just a basic 'as is' do as you wish...
/// Let me know if you use it as I'd be interested if people find it useful.
/// I'm going to keep experimenting wih the GL calls...eg GL.LINES etc 
/// </summary>
public class Graph : MonoBehaviour
{
    Material mat;
    public Rect windowRect = new Rect(1024, 0, 1000, 256);

    private static int _nextId = 0;
    public int windowId = _nextId++;

    public float yScale = 300;
    public float offset = 20;
    // A list of random values to draw  
    public List<float> values;

    // Start is called before the first frame update
    void Start()
    {
        mat = new Material(Shader.Find("Hidden/Internal-Colored"));
        // Should check for material but I'll leave that to you..
        Assert.IsNotNull(mat, "Graph: could not load material.");

        // Fill a list with ten random values
        values = new List<float>();
        values.Add(0);
    }

    // Update is called once per frame
    void Update()
    {
        // Keep adding values
        // values.Add(Random.value * 200);
    }

    void OnGUI()
    {
        windowRect = GUI.Window(windowId, windowRect, DrawGraph, "Monitor 0");
    }

    void drawSeries(List<float> values)
    {
        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.green);

        var valueIndex = values.Count - 1;
        for (var i = (int)windowRect.width - 4; i >= 3 && valueIndex >= 0; --i)
        {
            var y = values[valueIndex] * yScale + offset;
            GL.Vertex3(i, windowRect.height - 4 - y, 0);
            --valueIndex;
        }

        GL.End();
    }

    void drawAxis()
    {
        var y = windowRect.height - offset;
        /// draw X axis
        GL.Begin(GL.LINES);
        GL.Color(Color.red);

        GL.Vertex3(0, y, 0);
        GL.Vertex3(windowRect.width, y, 0);
        GL.End();
    }

    void DrawGraph(int windowID)
    {
        // Make Window Draggable
        GUI.DragWindow();

        // Draw the graph in the repaint cycle
        if (Event.current.type == EventType.Repaint)
        {
            GL.PushMatrix();

            GL.Clear(true, false, Color.black);
            mat.SetPass(0);

            // Draw a black back ground Quad 
            GL.Begin(GL.QUADS);
            GL.Color(Color.black);
            GL.Vertex3(4, 4, 0);
            GL.Vertex3(windowRect.width - 4, 4, 0);
            GL.Vertex3(windowRect.width - 4, windowRect.height - 4, 0);
            GL.Vertex3(4, windowRect.height - 4, 0);
            GL.End();

            // Draw the lines of the graph

            drawSeries(values);

            GL.PopMatrix();
        }

        drawAxis();
    }
}

