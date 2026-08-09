package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	duckPopulation = 20
	duckMaxSteps   = 1000
	duckTolerance  = 1e-6
	duckRangeLow   = -5.0
	duckRangeHigh  = 5.0
)

// Duck minimizuje konfigurisanu benchmark funkciju koristeći Duck Swarm Algorithm: patka nasumično bira između
// pretrage (široko nasumično istraživanje), foraginga (kretanje ka najboljem rešenju) ili budnosti (mala korekcija
// oko trenutne pozicije)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Duck(problem models.Problem) (result models.Result, err error) {

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

	population := duckPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := duckMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := duckTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	ducks := make([][]float64, population)
	values := make([]float64, population)
	for i := range ducks {
		duck := make([]float64, dimensions)
		for d := range duck {
			duck[d] = duckRangeLow + rand.Float64()*(duckRangeHigh-duckRangeLow)
		}
		ducks[i] = duck
		values[i] = fn.Evaluate(duck)
	}

	best := append([]float64(nil), ducks[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), ducks[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range ducks {
			p := rand.Float64()
			switch {
			case p < 0.3:
				for d := range ducks[i] {
					ducks[i][d] = ducks[i][d] + (rand.Float64()*2-1)*0.8
				}
			case p < 0.7:
				for d := range ducks[i] {
					ducks[i][d] = ducks[i][d] + rand.Float64()*(best[d]-ducks[i][d])
				}
			default:
				for d := range ducks[i] {
					ducks[i][d] = ducks[i][d] + (rand.Float64()*2-1)*0.1
				}
			}

			for d := range ducks[i] {
				ducks[i][d] = clamp(ducks[i][d], duckRangeLow, duckRangeHigh)
			}
			values[i] = fn.Evaluate(ducks[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), ducks[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "duck",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
