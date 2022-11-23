using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEditor;

namespace GoLightly.UI
{
    public static class Helpers
    {
        public static Vector3 pixelToWorld(int x, int y)
        {
            var v = new Vector3(x, 1024-y, 0);
            var s = Camera.main.ScreenToWorldPoint(v);
            s.z = 0;
            return s;
        }

        public static Vector3 pixelToWorld(Vector2 v)
        {
            return pixelToWorld((int)v.x, (int)v.y);
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
        }

        public static void GizmoRect(Rect r)
        {
            GizmoRect(new Vector2Int((int)r.xMin, (int)r.yMin), new Vector2Int((int)r.xMax, (int)r.yMax));
        }

    }
}
