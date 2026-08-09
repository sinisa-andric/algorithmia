package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	lobsterPopulation = 20
	lobsterMaxSteps   = 1000
	lobsterTolerance  = 1e-6
	lobsterRangeLow   = -5.0
	lobsterRangeHigh  = 5.0
)

// Lobster minimizuje konfigurisanu benchmark funkciju koristeći Lobster Optimization Algorithm: jastog ili
// postepeno prati hemijski trag ka najboljem rešenju sa koeficijentom koji blago opada tokom izvršavanja, ili
// izvodi odbrambeni skok repom — nagli veliki nasumičan pomeraj
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Lobster(problem models.Problem) (result models.Result, err error) {

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

	population := lobsterPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := lobsterMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := lobsterTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	lobsters := make([][]float64, population)
	values := make([]float64, population)
	for i := range lobsters {
		lobster := make([]float64, dimensions)
		for d := range lobster {
			lobster[d] = lobsterRangeLow + rand.Float64()*(lobsterRangeHigh-lobsterRangeLow)
		}
		lobsters[i] = lobster
		values[i] = fn.Evaluate(lobster)
	}

	best := append([]float64(nil), lobsters[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), lobsters[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range lobsters {
			if rand.Float64() < 0.7 {
				for d := range lobsters[i] {
					lobsters[i][d] = lobsters[i][d] + rand.Float64()*(best[d]-lobsters[i][d])*(1-float64(steps)/float64(maxSteps)*0.5)
				}
			} else {
				for d := range lobsters[i] {
					lobsters[i][d] = lobsters[i][d] + (rand.Float64()*2-1)*1.2
				}
			}

			for d := range lobsters[i] {
				lobsters[i][d] = clamp(lobsters[i][d], lobsterRangeLow, lobsterRangeHigh)
			}
			values[i] = fn.Evaluate(lobsters[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), lobsters[i]...)
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
		Method:     "lobster",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
