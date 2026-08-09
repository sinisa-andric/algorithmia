package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	secantMethodMaxSteps  = 1000
	secantMethodTolerance = 1e-6
	secantMethodPerturb   = 0.01
)

// SecantMethod minimizuje konfigurisanu benchmark funkciju koristeći sečice metod: drugi izvod po svakoj dimenziji se
// aproksimira iz uzastopnih razlika gradijenta umesto da se računa direktno
// problem.Point je početna tačka pretrage
func SecantMethod(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := secantMethodMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := secantMethodTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	gradient := fn.Gradient(point)

	pointPrev := make([]float64, len(point))
	for i := range pointPrev {
		pointPrev[i] = point[i] + secantMethodPerturb
	}
	gradientPrev := fn.Gradient(pointPrev)

	steps := 0
	for ; steps < maxSteps; steps++ {
		if norm(gradient) < tolerance {
			break
		}

		newPoint := make([]float64, len(point))
		for i := range point {
			dg := gradient[i] - gradientPrev[i]
			dx := point[i] - pointPrev[i]
			if math.Abs(dg) > 1e-10 {
				newPoint[i] = point[i] - gradient[i]*dx/dg
			} else {
				newPoint[i] = point[i] - secantMethodPerturb*gradient[i]
			}
		}

		pointPrev = point
		gradientPrev = gradient
		point = newPoint
		gradient = fn.Gradient(point)

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "secant_method",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
