package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	spsaA         = 0.1
	spsaC         = 0.1
	spsaAlpha     = 0.602
	spsaGamma     = 0.101
	spsaMaxSteps  = 1000
	spsaTolerance = 1e-6
)

// SPSA minimizuje konfigurisanu benchmark funkciju koristeći Simultaneous Perturbation Stochastic Approximation: grad
// ijent se procenjuje samo iz dva merenja funkcije duž nasumičnog pravca perturbacije, nezavisno od broja dimenzija
// problem.Point je početna tačka pretrage
func SPSA(problem models.Problem) (result models.Result, err error) {

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

	a := spsaA
	if v, ok := problem.Payload["a"].(float64); ok {
		a = v
	}

	c := spsaC
	if v, ok := problem.Payload["c"].(float64); ok {
		c = v
	}

	alpha := spsaAlpha
	if v, ok := problem.Payload["alpha"].(float64); ok {
		alpha = v
	}

	gamma := spsaGamma
	if v, ok := problem.Payload["gamma"].(float64); ok {
		gamma = v
	}

	maxSteps := spsaMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := spsaTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	dimensions := len(point)

	steps := 0
	for ; steps < maxSteps; steps++ {

		ak := a / math.Pow(float64(steps+1), alpha)
		ck := c / math.Pow(float64(steps+1), gamma)

		delta := make([]float64, dimensions)
		for i := range delta {
			if rand.Float64() < 0.5 {
				delta[i] = -1
			} else {
				delta[i] = 1
			}
		}

		plus := make([]float64, dimensions)
		minus := make([]float64, dimensions)
		for i := range point {
			plus[i] = point[i] + ck*delta[i]
			minus[i] = point[i] - ck*delta[i]
		}

		gPlus := fn.Evaluate(plus)
		gMinus := fn.Evaluate(minus)

		gradApprox := make([]float64, dimensions)
		for i := range gradApprox {
			gradApprox[i] = (gPlus - gMinus) / (2 * ck * delta[i])
		}

		if norm(gradApprox) < tolerance {
			break
		}

		for i := range point {
			point[i] -= ak * gradApprox[i]
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
		Method:     "spsa",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
