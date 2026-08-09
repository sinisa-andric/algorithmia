package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	polakRibiereCgMaxSteps  = 1000
	polakRibiereCgTolerance = 1e-6
)

// PolakRibiereCg minimizuje konfigurisanu benchmark funkciju koristeći Polak-Ribière konjugovani gradijent:
// koeficijent beta se računa iz projekcije promene gradijenta i automatski se restartuje kada postane negativan
// problem.Point je početna tačka pretrage
func PolakRibiereCg(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := polakRibiereCgMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := polakRibiereCgTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	gradient := fn.Gradient(point)
	direction := negate(gradient)
	steps := 0

	for ; steps < maxSteps; steps++ {
		if norm(gradient) < tolerance {
			break
		}

		alpha := 0.1 / (1 + norm(gradient))
		for i := range point {
			point[i] += alpha * direction[i]
		}

		newGradient := fn.Gradient(point)

		diff := make([]float64, len(gradient))
		for i := range diff {
			diff[i] = newGradient[i] - gradient[i]
		}
		beta := math.Max(0, dot(newGradient, diff)/dot(gradient, gradient))

		for i := range direction {
			direction[i] = -newGradient[i] + beta*direction[i]
		}

		gradient = newGradient

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "polak_ribiere_cg",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
