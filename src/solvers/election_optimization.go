package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	electionOptimizationPopulation = 20
	electionOptimizationMaxSteps   = 500
	electionOptimizationTolerance  = 1e-6
	electionOptimizationRangeLow   = -5.0
	electionOptimizationRangeHigh  = 5.0
)

// ElectionOptimization minimizuje konfigurisanu benchmark funkciju koristeći Election-Based Optimization
// Algorithm: kandidat sa više normalizovanih glasova jače teži ka pobedniku izbora (best-u), dok kandidat sa manje
// glasova zadržava veći udeo nezavisne kampanje
// problem.Point inicijalizuje populaciju kandidata i određuje njenu dimenzionalnost
func ElectionOptimization(problem models.Problem) (result models.Result, err error) {

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

	population := electionOptimizationPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := electionOptimizationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := electionOptimizationTolerance
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
			individual[d] = electionOptimizationRangeLow + rand.Float64()*(electionOptimizationRangeHigh-electionOptimizationRangeLow)
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

		worstValue := values[0]
		for _, v := range values {
			if v > worstValue {
				worstValue = v
			}
		}

		for i := range individuals {
			votesNorm := (worstValue - values[i]) / (worstValue - bestValue + 1e-10)

			for d := range individuals[i] {
				pull := votesNorm * rand.Float64() * (best[d] - individuals[i][d])
				// votesNorm teži 1 za pobednika (best), a (1-votesNorm) tada teži 0 zajedno sa pull članom —
				// bez donje granice bi pobednik potpuno prestao da se pomera; mala nezavisna kampanja to sprečava
				noise := (1-votesNorm)*(rand.Float64()*2-1)*0.5 + (rand.Float64()*2-1)*0.02
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, electionOptimizationRangeLow, electionOptimizationRangeHigh)
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
		Method:     "election_optimization",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
