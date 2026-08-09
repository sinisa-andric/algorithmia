package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	ellipsoidMethodMaxSteps     = 1000
	ellipsoidMethodTolerance    = 1e-6
	ellipsoidMethodRadiusFactor = 2.0
)

// EllipsoidMethod minimizuje sphere funkciju koristeći metod elipsoida za konveksnu optimizaciju: u svakom koraku
// ažurira centar i oblik elipsoida duž pravca gradijenta, garantujući smanjenje zapremine
// problem.Point je početna tačka pretrage
func EllipsoidMethod(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	maxSteps := ellipsoidMethodMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := ellipsoidMethodTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	center := append([]float64(nil), problem.Point...)
	n := len(center)

	radius := norm(problem.Point) * ellipsoidMethodRadiusFactor
	shape := make([][]float64, n)
	for i := range shape {
		shape[i] = make([]float64, n)
		shape[i][i] = radius * radius
	}

	best := append([]float64(nil), center...)
	bestValue := fn.Evaluate(center)

	steps := 0
	for ; steps < maxSteps; steps++ {

		gradient := fn.Gradient(center)
		shapeGradient := matVec(shape, gradient)

		normG := math.Sqrt(dot(gradient, shapeGradient))
		if normG < tolerance {
			break
		}

		for i := range shapeGradient {
			shapeGradient[i] /= normG
		}

		for i := range center {
			center[i] -= shapeGradient[i] / float64(n+1)
		}

		scale := float64(n*n) / float64(n*n-1)
		outerProduct := outer(shapeGradient, shapeGradient)

		for i := range shape {
			for j := range shape[i] {
				shape[i][j] = scale * (shape[i][j] - (2.0/float64(n+1))*outerProduct[i][j])
			}
		}

		if value := fn.Evaluate(center); value < bestValue {
			bestValue = value
			best = append([]float64(nil), center...)
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "ellipsoid_method",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

func outer(a, b []float64) [][]float64 {

	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(b))
		for j := range b {
			result[i][j] = a[i] * b[j]
		}
	}

	return result
}
