package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	quasiNewtonMaxSteps  = 1000
	quasiNewtonTolerance = 1e-6
	quasiNewtonStepSize  = 0.1
)

// QuasiNewton minimizuje konfigurisanu benchmark funkciju koristeći BFGS kvazi-Newton metod: inverzni Hesijan se
// ažurira BFGS formulom uz fiksnu veličinu koraka, bez line search-a
// problem.Point je početna tačka pretrage
func QuasiNewton(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := quasiNewtonMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := quasiNewtonTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	dimensions := len(point)
	inverseHessian := identityMatrix(dimensions)

	steps := 0
	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}

		direction := negate(matVec(inverseHessian, gradient))

		nextPoint := make([]float64, dimensions)
		for i := range point {
			nextPoint[i] = point[i] + quasiNewtonStepSize*direction[i]
		}

		s := make([]float64, dimensions)
		for i := range point {
			s[i] = nextPoint[i] - point[i]
		}

		nextGradient := fn.Gradient(nextPoint)
		y := make([]float64, dimensions)
		for i := range gradient {
			y[i] = nextGradient[i] - gradient[i]
		}

		ys := dot(y, s)
		if ys != 0 {
			inverseHessian = bfgsUpdate(inverseHessian, s, y, 1/ys)
		}

		point = nextPoint

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "quasi_newton",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

// bfgsUpdate primenjuje BFGS ažuriranje
// H' = (I - rho*s*y^T) H (I - rho*y*s^T) + rho*s*s^T.
func bfgsUpdate(inverseHessian [][]float64, s, y []float64, rho float64) [][]float64 {

	n := len(s)

	left := make([][]float64, n)
	right := make([][]float64, n)
	for i := 0; i < n; i++ {
		left[i] = make([]float64, n)
		right[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			identity := 0.0
			if i == j {
				identity = 1.0
			}
			left[i][j] = identity - rho*s[i]*y[j]
			right[i][j] = identity - rho*y[i]*s[j]
		}
	}

	updated := matMul(matMul(left, inverseHessian), right)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			updated[i][j] += rho * s[i] * s[j]
		}
	}

	return updated
}

func identityMatrix(n int) [][]float64 {

	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
		matrix[i][i] = 1
	}

	return matrix
}

func matVec(matrix [][]float64, vector []float64) []float64 {

	result := make([]float64, len(matrix))
	for i, row := range matrix {
		sum := 0.0
		for j, v := range vector {
			sum += row[j] * v
		}
		result[i] = sum
	}

	return result
}

func matMul(a, b [][]float64) [][]float64 {

	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(b[0]))
		for j := range b[0] {
			sum := 0.0
			for k := range b {
				sum += a[i][k] * b[k][j]
			}
			result[i][j] = sum
		}
	}

	return result
}
