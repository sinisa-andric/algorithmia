package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	pelicanPopulation = 20
	pelicanMaxSteps   = 1000
	pelicanTolerance  = 1e-6
	pelicanRangeLow   = -5.0
	pelicanRangeHigh  = 5.0
)

// Pelican minimizuje konfigurisanu benchmark funkciju koristeći Pelican Optimization Algorithm:
// u prvoj fazi svaki pelikan hoda ka nasumičnom boljem pelikanu ili se udaljava od lošijeg (eksploracija), a u drugoj
// fazi roni ka najboljem rešenju sa promenljivim intenzitetom, prihvatajući ronjenje samo ako poboljša rezultat
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Pelican(problem models.Problem) (result models.Result, err error) {

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

	population := pelicanPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := pelicanMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := pelicanTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	pelicans := make([][]float64, population)
	values := make([]float64, population)
	for i := range pelicans {
		pelican := make([]float64, dimensions)
		for d := range pelican {
			pelican[d] = pelicanRangeLow + rand.Float64()*(pelicanRangeHigh-pelicanRangeLow)
		}
		pelicans[i] = pelican
		values[i] = fn.Evaluate(pelican)
	}

	best := append([]float64(nil), pelicans[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), pelicans[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range pelicans {
			r := pelicans[rand.IntN(population)]
			if fn.Evaluate(r) < fn.Evaluate(pelicans[i]) {
				for d := range pelicans[i] {
					pelicans[i][d] += rand.Float64() * (r[d] - pelicans[i][d])
				}
			} else {
				for d := range pelicans[i] {
					pelicans[i][d] += rand.Float64() * (pelicans[i][d] - r[d])
				}
			}
			for d := range pelicans[i] {
				pelicans[i][d] = clamp(pelicans[i][d], pelicanRangeLow, pelicanRangeHigh)
			}

			intensity := math.Round(1 + rand.Float64())
			dive := make([]float64, dimensions)
			for d := range dive {
				dive[d] = clamp(pelicans[i][d]+rand.Float64()*(best[d]-intensity*pelicans[i][d]), pelicanRangeLow, pelicanRangeHigh)
			}
			if fn.Evaluate(dive) < fn.Evaluate(pelicans[i]) {
				pelicans[i] = dive
			}

			values[i] = fn.Evaluate(pelicans[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), pelicans[i]...)
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
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
		Method:     "pelican",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
