package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	honeyBadgerPopulation = 20
	honeyBadgerMaxSteps   = 1000
	honeyBadgerTolerance  = 1e-6
	honeyBadgerRangeLow   = -5.0
	honeyBadgerRangeHigh  = 5.0
)

// HoneyBadger minimizuje konfigurisanu benchmark funkciju koristeći Honey Badger Algorithm:
// svaki jazavac kopa oko najboljeg rešenja u smeru koji nasumično menja predznak, ili prati miris hrane ka razlici
// sopstvene i nasumične pozicije, skalirano faktorom koji eksponencijalno opada
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func HoneyBadger(problem models.Problem) (result models.Result, err error) {

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

	population := honeyBadgerPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := honeyBadgerMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := honeyBadgerTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	badgers := make([][]float64, population)
	values := make([]float64, population)
	for i := range badgers {
		badger := make([]float64, dimensions)
		for d := range badger {
			badger[d] = honeyBadgerRangeLow + rand.Float64()*(honeyBadgerRangeHigh-honeyBadgerRangeLow)
		}
		badgers[i] = badger
		values[i] = fn.Evaluate(badger)
	}

	best := append([]float64(nil), badgers[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), badgers[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		alpha := rand.Float64() * math.Exp(-float64(steps)/float64(maxSteps))

		for i := range badgers {
			di := make([]float64, dimensions)
			for d := range di {
				di[d] = math.Abs(best[d] - badgers[i][d])
			}

			if rand.Float64() < 0.5 {
				f := 1.0
				if rand.Float64() < 0.5 {
					f = -1
				}
				for d := range badgers[i] {
					badgers[i][d] = best[d] + f*alpha*di[d]*rand.Float64()
				}
			} else {
				r := badgers[rand.IntN(population)]
				for d := range badgers[i] {
					badgers[i][d] = best[d] + alpha*rand.Float64()*(badgers[i][d]-r[d])
				}
			}

			for d := range badgers[i] {
				badgers[i][d] = clamp(badgers[i][d], honeyBadgerRangeLow, honeyBadgerRangeHigh)
			}

			values[i] = fn.Evaluate(badgers[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), badgers[i]...)
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
		Method:     "honey_badger",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
