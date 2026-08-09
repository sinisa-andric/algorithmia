package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	gaussNewtonMaxSteps    = 1000
	gaussNewtonTolerance   = 1e-6
	gaussNewtonHessianStep = 1e-5
	gaussNewtonMinHessian  = 1e-8
)

// GaussNewton minimizuje konfigurisanu benchmark funkciju koristeći ažuriranje u stilu Gauss-Newton-a: Hesijan se
// aproksimira preko dijagonalnog Jakobijana izvedenog iz gradijenta
// problem.Point je početna tačka pretrage
func GaussNewton(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := gaussNewtonMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := gaussNewtonTolerance
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

		hessianDiag := diagonalHessian(fn.Evaluate, point, gaussNewtonHessianStep)

		for i := range point {
			h := hessianDiag[i]
			if h < gaussNewtonMinHessian {
				h = gaussNewtonMinHessian
			}
			point[i] -= gradient[i] / h
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
		Method:     "gauss_newton",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
