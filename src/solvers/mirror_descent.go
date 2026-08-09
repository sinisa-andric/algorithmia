package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	mirrorDescentLearningRate = 0.1
	mirrorDescentMaxSteps     = 1000
	mirrorDescentTolerance    = 1e-6
)

// MirrorDescent minimizuje konfigurisanu benchmark funkciju koristeći mirror descent sa Euklidskim mirror mapiranjem:
// nad neograničenim prostorom svodi se na gradijentni spust sa opadajućom veličinom koraka learning_rate/sqrt(step+1)
// problem.Point je početna tačka pretrage
func MirrorDescent(problem models.Problem) (result models.Result, err error) {

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

	learningRate := mirrorDescentLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	maxSteps := mirrorDescentMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := mirrorDescentTolerance
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

		lr := learningRate / math.Sqrt(float64(steps+1))

		// y = x - lr*gradient(x); projekcija na neograničen prostor je
		// identitet, pa je x_new = y.
		for i := range point {
			point[i] -= lr * gradient[i]
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
		Method:     "mirror_descent",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
