package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	woodpeckerPopulation = 20
	woodpeckerMaxSteps   = 1000
	woodpeckerTolerance  = 1e-6
	woodpeckerRangeLow   = -5.0
	woodpeckerRangeHigh  = 5.0
)

// Woodpecker minimizuje konfigurisanu benchmark funkciju koristeći Woodpecker Mating Algorithm: detlić ili kuca
// fino istražujući oko trenutne pozicije amplitudom koja opada tokom izvršavanja, ili odleti direktno ka najboljem
// pronađenom stablu
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Woodpecker(problem models.Problem) (result models.Result, err error) {

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

	population := woodpeckerPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := woodpeckerMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := woodpeckerTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	woodpeckers := make([][]float64, population)
	values := make([]float64, population)
	for i := range woodpeckers {
		woodpecker := make([]float64, dimensions)
		for d := range woodpecker {
			woodpecker[d] = woodpeckerRangeLow + rand.Float64()*(woodpeckerRangeHigh-woodpeckerRangeLow)
		}
		woodpeckers[i] = woodpecker
		values[i] = fn.Evaluate(woodpecker)
	}

	best := append([]float64(nil), woodpeckers[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), woodpeckers[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range woodpeckers {
			if rand.Float64() < 0.6 {
				for d := range woodpeckers[i] {
					woodpeckers[i][d] = woodpeckers[i][d] + (rand.Float64()*2-1)*0.15*(1-float64(steps)/float64(maxSteps))
				}
			} else {
				for d := range woodpeckers[i] {
					woodpeckers[i][d] = woodpeckers[i][d] + rand.Float64()*(best[d]-woodpeckers[i][d])
				}
			}

			for d := range woodpeckers[i] {
				woodpeckers[i][d] = clamp(woodpeckers[i][d], woodpeckerRangeLow, woodpeckerRangeHigh)
			}
			values[i] = fn.Evaluate(woodpeckers[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), woodpeckers[i]...)
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
		Method:     "woodpecker",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
