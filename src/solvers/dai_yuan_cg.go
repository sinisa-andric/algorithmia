package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	daiYuanCgMaxSteps  = 1000
	daiYuanCgTolerance = 1e-6
)

// DaiYuanCg minimizuje konfigurisanu benchmark funkciju koristeći Dai-Yuan nelinearni metod konjugovanog gradijenta:
// pravac pretrage kombinuje trenutni gradijent sa prethodnim pravcem preko odnosa koji garantuje descent svojstvo
// problem.Point je početna tačka pretrage
func DaiYuanCg(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := daiYuanCgMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := daiYuanCgTolerance
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

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}

		newGradient := fn.Gradient(point)

		y := make([]float64, len(gradient))
		for i := range y {
			y[i] = newGradient[i] - gradient[i]
		}

		dy := dot(direction, y)
		beta := 0.0
		if math.Abs(dy) >= 1e-10 {
			beta = dot(newGradient, newGradient) / dy
		}

		for i := range direction {
			direction[i] = -newGradient[i] + beta*direction[i]
		}

		gradient = newGradient
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "dai_yuan_cg",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
