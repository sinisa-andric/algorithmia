package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	levenbergMarquardtLambda      = 0.01
	levenbergMarquardtLambdaUp    = 10.0
	levenbergMarquardtLambdaDown  = 0.1
	levenbergMarquardtMaxSteps    = 1000
	levenbergMarquardtTolerance   = 1e-6
	levenbergMarquardtHessianStep = 1e-5
)

// LevenbergMarquardt minimizuje konfigurisanu benchmark funkciju koristeći Levenberg-Marquardt metod: dijagonalna
// aproksimacija Hesijana se prigušuje parametrom lambda koji raste posle neuspešnog koraka i opada posle uspešnog,
// interpolirajući između Gauss-Newton-a i gradijentnog spusta
// problem.Point je početna tačka pretrage
func LevenbergMarquardt(problem models.Problem) (result models.Result, err error) {

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

	lambda := levenbergMarquardtLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	lambdaUp := levenbergMarquardtLambdaUp
	if v, ok := problem.Payload["lambda_up"].(float64); ok {
		lambdaUp = v
	}

	lambdaDown := levenbergMarquardtLambdaDown
	if v, ok := problem.Payload["lambda_down"].(float64); ok {
		lambdaDown = v
	}

	maxSteps := levenbergMarquardtMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := levenbergMarquardtTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}

		hessianDiag := diagonalHessian(fn.Evaluate, point, levenbergMarquardtHessianStep)

		direction := make([]float64, len(point))
		for i := range point {
			direction[i] = -gradient[i] / (hessianDiag[i] + lambda)
		}

		newPoint := make([]float64, len(point))
		for i := range point {
			newPoint[i] = point[i] + direction[i]
		}

		if fn.Evaluate(newPoint) < fn.Evaluate(point) {
			point = newPoint
			lambda *= lambdaDown
		} else {
			lambda *= lambdaUp
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "levenberg_marquardt",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

// diagonalHessian aproksimira dijagonalu Hesijana funkcije f u tački point
// koristeći centralne razlike sa korakom h.
func diagonalHessian(f func([]float64) float64, point []float64, h float64) []float64 {

	center := f(point)
	diag := make([]float64, len(point))

	for i := range point {
		plus := append([]float64(nil), point...)
		minus := append([]float64(nil), point...)
		plus[i] += h
		minus[i] -= h
		diag[i] = (f(plus) - 2*center + f(minus)) / (h * h)
	}

	return diag
}
