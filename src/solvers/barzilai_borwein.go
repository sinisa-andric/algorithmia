package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	barzilaiBorweinMaxSteps    = 1000
	barzilaiBorweinTolerance   = 1e-6
	barzilaiBorweinInitialStep = 0.01
)

// BarzilaiBorwein minimizuje konfigurisanu benchmark funkciju koristeći Barzilai-Borwein spektralni gradijentni
// metod: veličina koraka se u svakoj iteraciji procenjuje iz odnosa uzastopnih razlika pozicije i gradijenta
// problem.Point je početna tačka pretrage
func BarzilaiBorwein(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := barzilaiBorweinMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := barzilaiBorweinTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	gradient := fn.Gradient(point)
	alpha := barzilaiBorweinInitialStep
	steps := 0

	for ; steps < maxSteps; steps++ {
		if norm(gradient) < tolerance {
			break
		}

		newPoint := make([]float64, len(point))
		for i := range point {
			newPoint[i] = point[i] - alpha*gradient[i]
		}
		newGradient := fn.Gradient(newPoint)

		s := make([]float64, len(point))
		y := make([]float64, len(point))
		for i := range point {
			s[i] = newPoint[i] - point[i]
			y[i] = newGradient[i] - gradient[i]
		}

		sy := dot(s, y)
		ss := dot(s, s)
		if math.Abs(sy) > 1e-10 {
			alpha = ss / math.Abs(sy)
		}
		alpha = math.Max(1e-10, math.Min(1.0, alpha))

		point = newPoint
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
		Method:     "barzilai_borwein",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
