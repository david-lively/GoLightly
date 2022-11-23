using UnityEngine;
using UnityEditor;

/// <summary>
/// Convenience class to add some in-editor notes to a GameObject. 
/// </summary>
public class Note : MonoBehaviour
{
    public string summary;
    public string text;
}

[CustomEditor(typeof(Note))]
class NotesEditor : Editor
{
    public override void OnInspectorGUI()
    {
        var note = target as Note;

        // DrawDefaultInspector();
        GUILayout.Label("Summary");
        note.summary = GUILayout.TextField(note.summary);

        GUILayout.Label("Enter your notes below");
        note.text = GUILayout.TextArea(note.text);
    }
}
