package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	spsaBundleA         = 0.1
	spsaBundleC         = 0.1
	spsaBundleSize      = 5
	spsaBundleMaxSteps  = 1000
	spsaBundleTolerance = 1e-6
)

// SPSABundle minimizuje konfigurisanu benchmark funkciju koristeći SPSA sa bundle usrednjavanjem gradijenta: svaka
// SPSA aproksimacija gradijenta se čuva u kliznom prozoru nedavnih procena i usrednjava pre svakog ažuriranja
// problem.Point je početna tačka pretrage
func SPSABundle(problem models.Problem) (result models.Result, err error) {

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

	a := spsaBundleA
	if v, ok := problem.Payload["a"].(float64); ok {
		a = v
	}

	c := spsaBundleC
	if v, ok := problem.Payload["c"].(float64); ok {
		c = v
	}

	bundleSize := spsaBundleSize
	if v, ok := problem.Payload["bundle_size"].(float64); ok {
		bundleSize = int(v)
	}

	maxSteps := spsaBundleMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := spsaBundleTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	dimensions := len(point)
	bundle := make([][]float64, 0, bundleSize)

	steps := 0
	for ; steps < maxSteps; steps++ {

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
			plus[i] = point[i] + c*delta[i]
			minus[i] = point[i] - c*delta[i]
		}

		gPlus := fn.Evaluate(plus)
		gMinus := fn.Evaluate(minus)

		gradApprox := make([]float64, dimensions)
		for i := range gradApprox {
			gradApprox[i] = (gPlus - gMinus) / (2 * c * delta[i])
		}

		bundle = append(bundle, gradApprox)
		if len(bundle) > bundleSize {
			bundle = bundle[1:]
		}

		aggregate := make([]float64, dimensions)
		for _, g := range bundle {
			for i := range aggregate {
				aggregate[i] += g[i]
			}
		}
		for i := range aggregate {
			aggregate[i] /= float64(len(bundle))
		}

		if norm(aggregate) < tolerance {
			break
		}

		aK := a / math.Sqrt(float64(steps+1))
		for i := range point {
			point[i] -= aK * aggregate[i]
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
		Method:     "spsa_bundle",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
