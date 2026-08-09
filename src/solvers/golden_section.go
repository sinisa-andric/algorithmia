package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	goldenSectionMaxSteps  = 1000
	goldenSectionTolerance = 1e-6
)

// GoldenSection minimizuje konfigurisanu benchmark funkciju koristeći pretragu zlatnim presekom: u svakoj iteraciji
// poredi dve unutrašnje tačke intervala [a, b] iz payload-a i odbacuje deo koji ne može sadržati minimum
// problem.Point se ne koristi, minimizacija se izvodi isključivo nad intervalom [a, b]
func GoldenSection(problem models.Problem) (result models.Result, err error) {

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(1); err != nil {
		return result, err
	}

	a, ok := problem.Payload["a"].(float64)
	if !ok {
		err = fmt.Errorf("interval bound a is required")
		return result, err
	}

	b, ok := problem.Payload["b"].(float64)
	if !ok {
		err = fmt.Errorf("interval bound b is required")
		return result, err
	}

	maxSteps := goldenSectionMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := goldenSectionTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	phi := (math.Sqrt(5) - 1) / 2

	c := b - phi*(b-a)
	d := a + phi*(b-a)

	steps := 0
	for ; steps < maxSteps && math.Abs(b-a) > tolerance; steps++ {
		if fn.Evaluate([]float64{c}) < fn.Evaluate([]float64{d}) {
			b = d
		} else {
			a = c
		}
		c = b - phi*(b-a)
		d = a + phi*(b-a)

		if includeTrajectory {
			midX := (a + b) / 2
			trajectory = recordTrajectory(trajectory, steps, []float64{midX}, fn.Evaluate([]float64{midX}), false)
		}
	}

	x := (a + b) / 2

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, []float64{x}, fn.Evaluate([]float64{x}), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "golden_section",
		Point:      []float64{x},
		Value:      fn.Evaluate([]float64{x}),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
