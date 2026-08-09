package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	riverFormationPopulation = 20
	riverFormationMaxSteps   = 500
	riverFormationTolerance  = 1e-6
	riverFormationRangeLow   = -5.0
	riverFormationRangeHigh  = 5.0
	riverFormationCandidates = 3
)

// RiverFormation minimizuje konfigurisanu benchmark funkciju koristeći River Formation Dynamics: svaka kap vode
// uzorkuje nekoliko nasumičnih kandidata u sopstvenoj okolini i erodira teren u pravcu najvećeg pada, uz dodatni
// blagi pomak ka best-u koji predstavlja nizvodni tok
// problem.Point inicijalizuje populaciju kapi i određuje njenu dimenzionalnost
func RiverFormation(problem models.Problem) (result models.Result, err error) {

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

	population := riverFormationPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := riverFormationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := riverFormationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = riverFormationRangeLow + rand.Float64()*(riverFormationRangeHigh-riverFormationRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

	best := append([]float64(nil), individuals[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), individuals[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range individuals {
			bestCandidate := individuals[i]
			bestCandidateValue := values[i]
			for range riverFormationCandidates {
				candidate := make([]float64, dimensions)
				for d := range candidate {
					candidate[d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*0.5, riverFormationRangeLow, riverFormationRangeHigh)
				}
				candidateValue := fn.Evaluate(candidate)
				if candidateValue < bestCandidateValue {
					bestCandidateValue = candidateValue
					bestCandidate = candidate
				}
			}

			for d := range individuals[i] {
				individuals[i][d] = clamp(bestCandidate[d]+rand.Float64()*(best[d]-bestCandidate[d])*0.2, riverFormationRangeLow, riverFormationRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
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
		Method:     "river_formation",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
