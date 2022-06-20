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
    public Rect windowRect = new Rect(20, 20, 1000, 256);

    public float yScale = 2000;
    public float offset = -1000;
    // A list of random values to draw  
    public List<float> values;

    // The list the drawing function uses...
    // private List<float> drawValues = new List<float>();
    public List<GoLightly.Monitor> monitors;

    // List of Windows
    private bool showWindow = true;

    // Start is called before the first frame update
    void Start()
    {
        mat = new Material(Shader.Find("Hidden/Internal-Colored"));
        // Should check for material but I'll leave that to you..
        Assert.IsNotNull(mat, "Graph: could not load material.");

        // Fill a list with ten random values
        values = new List<float>();
        values.Add(0);
        // for (int i = 0; i < 10; i++)
        // {
        //     values.Add(Random.value * 200);
        // }
    }

    // Update is called once per frame
    void Update()
    {
        // Keep adding values
        // values.Add(Random.value * 200);
    }

    private void OnGUI()
    {
        // Create a GUI.toggle to show graph window
        showWindow = GUI.Toggle(new Rect(10, 10, 100, 20), showWindow, "Show Graph");

        if (showWindow)
        {
            // Set out drawValue list equal to the values list 
            // drawValues = values;
            windowRect = GUI.Window(0, windowRect, DrawGraph, "");
        }
    }


    void drawSeries(List<float> values)
    {
            GL.Begin(GL.LINES);
            GL.Color(Color.green);

            int valueIndex = values.Count - 1;
            for (int i = (int)windowRect.width - 4; i > 3; i--)
            {
                float y1 = 0;
                float y2 = 0;
                if (valueIndex > 0)
                {
                    y2 = values[valueIndex] * yScale + offset;
                    y1 = values[valueIndex - 1] * yScale + offset;
                }
                GL.Vertex3(i, windowRect.height - 4 - y2, 0);
                GL.Vertex3((i - 1), windowRect.height - 4 - y1, 0);
                valueIndex -= 1;
            }
            GL.End();
    }

    void DrawGraph(int windowID)
    {
        // Make Window Draggable
        GUI.DragWindow(new Rect(0, 0, 10000, 10000));

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
    }
}

