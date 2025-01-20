using UnityEngine;
using System.IO;
using System.Collections.Generic;
using UnityEngine.UI;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using TMPro;
using System;


public class PointCloudAligner : MonoBehaviour
{

    [Header("File Paths")]
    public string filePathP = "Assets/file5.txt";

    public string filePathQ = "Assets/file6.txt";

    [Header("Materials")]
    public Material materialP;
    public Material materialQ;
    public Material materialTransformedQ;
    public Material materialLine;

    [Header("UI Elements")]
    public Button alignButton;
    public Button toggleVisualizationButton;
    public Button toggleMethodButton;
    public TextMeshProUGUI transformationInfoText;

    private List<Vector3> pointsP = new List<Vector3>();
    private List<Vector3> pointsQ = new List<Vector3>();
    private List<Vector3> transformedQ = new List<Vector3>();

    private Matrix<double> bestRotation; // Rotation matrix
    private Vector<double> bestTranslation; // Translation vector

    private bool showOriginalAndAligned = true; 

    // Threshold for RANSAC
    private float inlierThreshold = 0.5f; 

    void Start()
    {
        // Assign button listeners
        if (alignButton != null)
            alignButton.onClick.AddListener(OnTransformationButtonPressed);

        if (toggleVisualizationButton != null)
            toggleVisualizationButton.onClick.AddListener(OnMovementLineButtonPressed);

        if (toggleMethodButton != null)
            toggleMethodButton.onClick.AddListener(OnParametersButtonPressed);

        // Load point clouds
        if (!string.IsNullOrEmpty(filePathP) && File.Exists(filePathP))
            pointsP = LoadPointsFromFile(filePathP);
        else
            Debug.LogError("File path P is invalid or does not exist.");

        if (!string.IsNullOrEmpty(filePathQ) && File.Exists(filePathQ))
            pointsQ = LoadPointsFromFile(filePathQ);
        else
            Debug.LogError("File path Q is invalid or does not exist.");

        // Initial visualization of point clouds
        VisualizePoints(pointsP, materialP, "P_");
        VisualizePoints(pointsQ, materialQ, "Q_");
    }

    public void OnTransformationButtonPressed()
    {
        if (pointsP.Count < 3 || pointsQ.Count < 3)
        {
            Debug.LogError("Less then 3 points.");
            return;
        }

        Debug.Log("Starting Transformation...");

        // Perform Rigid Transformation using RANSAC and Kabsch
        ComputeRigidTransformationRANSAC();

        // Apply Transformation
        ApplyTransformation();

        // Update Visualization
        UpdateVisualization();
    }

    public void OnMovementLineButtonPressed()
    {
        showOriginalAndAligned = !showOriginalAndAligned;
        UpdateVisualization();
    }

    public void OnParametersButtonPressed()
    {
        // Show the transfortmation info
        // DisplayTransformationInfo();
        DisplayTransformationInfo(pointsP);
    }

    private List<Vector3> LoadPointsFromFile(string filePath)
    {
        List<Vector3> points = new List<Vector3>(); // List to store the points

        try
        {
            string[] lines = File.ReadAllLines(filePath);
            if (lines.Length < 1)
            {
                Debug.LogError("File is empty: " + filePath);
                return points;
            }

            int numPts = int.Parse(lines[0].Trim());
            if (lines.Length < numPts + 1)
            {
                Debug.LogError($"File {filePath} has fewer lines than expected. Expected {numPts} points.");
                return points;
            }

            for (int i = 1; i <= numPts; i++)
            {
                // Split line by space or tab
                string[] vals = lines[i].Trim().Split(new char[] { ' ', '\t' }, System.StringSplitOptions.RemoveEmptyEntries);
                if (vals.Length < 3)
                {
                    Debug.LogWarning($"Line {i + 1} in {filePath} does not have enough coordinates.");
                    continue;
                }

                float x = float.Parse(vals[0]);
                float y = float.Parse(vals[1]);
                float z = float.Parse(vals[2]);

                points.Add(new Vector3(x, y, z));
            }

            Debug.Log($"Loaded {points.Count} points from {filePath}.");
        }
        catch (System.Exception e)
        {
            Debug.LogError("Error reading file " + filePath + ": " + e.Message);
        }

        return points;
    }

    private void VisualizePoints(List<Vector3> points, Material mat, string prefix)
    {
        foreach (Vector3 point in points)
        {
            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.transform.position = point;
            sphere.transform.localScale = Vector3.one * 0.3f; 
            sphere.GetComponent<Renderer>().material = mat;
            sphere.name = prefix + System.Guid.NewGuid().ToString();
        }
    }

    // Called when updating the visualization
    private void ClearVisualization()
    {
        // Find and destroy all child spheres and lines
        GameObject[] allObjects = FindObjectsByType<GameObject>(FindObjectsSortMode.None);
        foreach (GameObject obj in allObjects)
        {
            if (obj.name.StartsWith("P_") || obj.name.StartsWith("Q_") || obj.name.StartsWith("Q_trans_") || obj.name.StartsWith("Line_"))
            {
                Destroy(obj);
            }
        }
    }

    private void ComputeRigidTransformationRANSAC()
    {
        int iterations = 1000;
        int maxInliers = 0;
        Matrix<double> bestR = null;
        Vector<double> bestT = null;

        Vector3[] P_arr = pointsP.ToArray();
        Vector3[] Q_arr = pointsQ.ToArray();

        System.Random rand = new System.Random(); // Random number generator

        // Loop through RANSAC iterations
        for (int i = 0; i < iterations; i++)
        {
            // Randomly select 3 unique indices from P and Q for 3-point alignment
            int[] indicesP = GetUniqueRandomIndices(rand, pointsP.Count, 3);
            int[] indicesQ = GetUniqueRandomIndices(rand, pointsQ.Count, 3);

            Vector3[] P_sample = new Vector3[3];
            Vector3[] Q_sample = new Vector3[3];

            // Get the 3 points from P and Q
            for (int j = 0; j < 3; j++)
            {
                P_sample[j] = P_arr[indicesP[j]];
                Q_sample[j] = Q_arr[indicesQ[j]];
            }

            // Compute transformation using Kabsch algorithm (SVD)
            bool success = ComputeKabsch(P_sample, Q_sample, out Matrix<double> R, out Vector<double> T);
            if (!success)
                continue;

            // Count inliers using the computed rotation and translation
            int inliers = CountInliers(P_arr, Q_arr, R, T);

            // Update best transformation if more inliers found
            if (inliers > maxInliers)
            {
                maxInliers = inliers;
                bestR = R;
                bestT = T;
            }
        }

        // Set the best transformation
        if (bestR != null && bestT != null)
        {
            bestRotation = bestR;
            bestTranslation = bestT;
            Debug.Log($"Best Transformation Found with {maxInliers} inliers out of {pointsQ.Count} Q points.");
        }
        else
        {
            Debug.LogError("Failed to find a valid transformation.");
        }
    }

    // Helper function to get unique random indices
    private int[] GetUniqueRandomIndices(System.Random rand, int max, int count)
    {
        HashSet<int> indices = new HashSet<int>();
        while (indices.Count < count)
        {
            indices.Add(rand.Next(max));
        }
        int[] result = new int[count];
        indices.CopyTo(result);
        return result;
    }

    // Kabsh Algorithm to compute the rotation and translation
    private bool ComputeKabsch(Vector3[] P, Vector3[] Q, out Matrix<double> R, out Vector<double> T)
    {
        R = null;
        T = null;

        if (P.Length != Q.Length || P.Length < 3)
            return false;

        // Convert to Math.NET matrices
        Matrix<double> matP = Matrix<double>.Build.Dense(3, P.Length);
        Matrix<double> matQ = Matrix<double>.Build.Dense(3, Q.Length);

        // Fill matrices with point coordinates
        for (int i = 0; i < P.Length; i++)
        {
            matP[0, i] = P[i].x;
            matP[1, i] = P[i].y;
            matP[2, i] = P[i].z;

            matQ[0, i] = Q[i].x;
            matQ[1, i] = Q[i].y;
            matQ[2, i] = Q[i].z;
        }

        // Compute centroids
        Vector<double> centroidP = matP.RowSums() / P.Length;
        Vector<double> centroidQ = matQ.RowSums() / Q.Length;

        // Subtract centroids from each point 
        Matrix<double> P_centered = SubtractCentroid(matP, centroidP);
        Matrix<double> Q_centered = SubtractCentroid(matQ, centroidQ);

        // Compute covariance matrix
        Matrix<double> H = P_centered * Q_centered.Transpose();

        // Perform SVD on covariance matrix
        var svd = H.Svd();

        // Find U and V matrices to compute rotation
        Matrix<double> U = svd.U;
        Matrix<double> V = svd.VT.Transpose();

        // Compute rotation
        R = V * U.Transpose();

        // If the determinant of R is -1, then we need to flip the last column of V
        if (R.Determinant() < 0)
        {
            // Flip the last column of V
            Matrix<double> V_flipped = V.Clone();
            V_flipped.SetColumn(2, V_flipped.Column(2).Multiply(-1));
            R = V_flipped * U.Transpose();
        }

        // Compute translation using the centroids
        T = centroidP - (R * centroidQ);

        Debug.Log("Kabsch Transformation Computed.");

        return true;
    }

    // Subtract the centroid from each point in the matrix
    // Used for centering the points around the origin
    private Matrix<double> SubtractCentroid(Matrix<double> matrix, Vector<double> centroid)
    {
        Matrix<double> centered = matrix.Clone();

        for (int i = 0; i < matrix.ColumnCount; i++)
        {
            centered.SetColumn(i, centered.Column(i) - centroid);
        }

        return centered;
    }

    // Count the number of inliers based on the threshold
    private int CountInliers(Vector3[] P, Vector3[] Q, Matrix<double> R, Vector<double> T)
    {
        int inliers = 0;

        List<Vector3> P_list = new List<Vector3>(P);

        // Loop through all points in Q 
        foreach (Vector3 q in Q)
        {
            // Apply transformation
            Vector<double> qVec = Vector<double>.Build.Dense(new double[] { q.x, q.y, q.z });
            Vector<double> qTransformed = R * qVec + T; // We found the rotation and translation from Kabsch

            // Get the transformed point from index 0, 1, 2 of the vector qTransformed
            Vector3 qPoint = new Vector3((float)qTransformed[0], (float)qTransformed[1], (float)qTransformed[2]);

            // Find the closest P point to the transformed Q point
            float minDist = float.MaxValue;
            foreach (Vector3 p in P_list)
            {
                float dist = Vector3.Distance(qPoint, p);
                if (dist < minDist)
                    minDist = dist; // Update the minimum distance
            }

            // If the distance is less than the threshold, it is an inlier
            if (minDist < inlierThreshold)
                inliers++;
        }

        return inliers;
    }

    // Used after the transformation is computed with RANSAC
    private void ApplyTransformation()
    {
        transformedQ.Clear();

        // Apply the transformation to all points in Q
        for (int i = 0; i < pointsQ.Count; i++)
        {
            Vector<double> qVec = Vector<double>.Build.Dense(new double[] { pointsQ[i].x, pointsQ[i].y, pointsQ[i].z });
            Vector<double> qTransformed = bestRotation * qVec + bestTranslation;
            Vector3 qPoint = new Vector3((float)qTransformed[0], (float)qTransformed[1], (float)qTransformed[2]);
            transformedQ.Add(qPoint);
        }

        Debug.Log("Transformation Applied to Q points.");
    }

    // Show movement lines or just points
    private void UpdateVisualization()
    {
        ClearVisualization();

        // Visualize the points only
        if (showOriginalAndAligned)
        {
            // Visualize original P and Q
            VisualizePoints(pointsP, materialP, "P_");
            VisualizePoints(pointsQ, materialQ, "Q_");

            // Visualize transformed Q
            VisualizePoints(transformedQ, materialTransformedQ, "Q_trans_");
        }
        else
        {
            // Visualize movement lines
            for (int i = 0; i < pointsQ.Count; i++)
            {
                GameObject lineObj = new GameObject("Line_" + i);
                LineRenderer lr = lineObj.AddComponent<LineRenderer>();
                lr.material = materialLine;
                lr.startWidth = 0.05f; 
                lr.endWidth = 0.05f;
                lr.positionCount = 2;
                lr.SetPosition(0, pointsQ[i]);
                lr.SetPosition(1, transformedQ[i]);

                GameObject sphereStart = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphereStart.transform.position = pointsQ[i];
                sphereStart.transform.localScale = Vector3.one * 0.02f;
                sphereStart.GetComponent<Renderer>().material = materialQ;
                sphereStart.name = "Q_Start_" + i;

                GameObject sphereEnd = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphereEnd.transform.position = transformedQ[i];
                sphereEnd.transform.localScale = Vector3.one * 0.02f;
                sphereEnd.GetComponent<Renderer>().material = materialTransformedQ;
                sphereEnd.name = "Q_End_" + i;
            }

            // Visualize original P and Q and transformed Q
            VisualizePoints(pointsP, materialP, "P_");
            VisualizePoints(pointsQ, materialQ, "Q_");
            VisualizePoints(transformedQ, materialTransformedQ, "Q_trans_");
        }
    }

    private void DisplayTransformationInfo(List<Vector3> pointsP)
    {
        if (transformationInfoText == null)
            return;

        // Extract rotation matrix and translation vector
        Matrix<double> R = bestRotation;
        Vector<double> T = bestTranslation;

        // Create a Unity Matrix4x4 for easy conversion to Euler angles
        Matrix4x4 unityR = new Matrix4x4();
        unityR.SetRow(0, new Vector4((float)R[0, 0], (float)R[0, 1], (float)R[0, 2], 0));
        unityR.SetRow(1, new Vector4((float)R[1, 0], (float)R[1, 1], (float)R[1, 2], 0));
        unityR.SetRow(2, new Vector4((float)R[2, 0], (float)R[2, 1], (float)R[2, 2], 0));
        unityR.SetRow(3, new Vector4(0, 0, 0, 1));

        Quaternion rotation = unityR.rotation;
        Vector3 euler = rotation.eulerAngles;
        Vector3 translation = new Vector3((float)T[0], (float)T[1], (float)T[2]);

        // Debug transformed points
        foreach (var point in pointsP)
        {
            Vector<double> p = Vector<double>.Build.DenseOfArray(new double[] { point.x, point.y, point.z });
            Vector<double> q = R * p + T;

            Debug.Log($"Original: ({point.x:F2}, {point.y:F2}, {point.z:F2}) -> " +
                      $"Transformed: ({q[0]:F2}, {q[1]:F2}, {q[2]:F2})");
        }

        // Display transformation info
        string info = $"Translation:\n" +
                    $"X: {translation.x:F2}, Y: {translation.y:F2}, Z: {translation.z:F2}\n\n" +
                    $"Rotation Matrix:\n" +
                    $"[{R[0, 0]:F2} {R[0, 1]:F2} {R[0, 2]:F2}]\n" +
                    $"[{R[1, 0]:F2} {R[1, 1]:F2} {R[1, 2]:F2}]\n" +
                    $"[{R[2, 0]:F2} {R[2, 1]:F2} {R[2, 2]:F2}]\n\n";

        transformationInfoText.text = info;
        Debug.Log(info);
    }


}


