package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	frankWolfeMaxSteps  = 1000
	frankWolfeTolerance = 1e-6
)

// FrankWolfe minimizuje konfigurisanu benchmark funkciju koristeći Frank-Wolfe (uslovni gradijent) metod: nad
// neograničenim prostorom svodi se na gradijentni spust sa opadajućom veličinom koraka 2/(step+2)
// problem.Point je početna tačka pretrage
func FrankWolfe(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := frankWolfeMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := frankWolfeTolerance
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

		gamma := 2.0 / (float64(steps) + 2)
		for i := range point {
			point[i] -= gamma * gradient[i]
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
		Method:     "frank_wolfe",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
