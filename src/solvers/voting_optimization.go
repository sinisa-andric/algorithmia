package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	votingOptimizationPopulation = 20
	votingOptimizationMaxSteps   = 500
	votingOptimizationTolerance  = 1e-6
	votingOptimizationRangeLow   = -5.0
	votingOptimizationRangeHigh  = 5.0
)

// VotingOptimization minimizuje konfigurisanu benchmark funkciju koristeći Voting-Based Optimization: jedinke
// glasaju unutar nasumičnog podskupa od tri kandidata, pomeraju se ka pobedniku podskupa i ka globalnom
// konsenzusu (best-u), uz nezavisan šum koji sprečava kolaps na trenutno najbolju jedinku
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func VotingOptimization(problem models.Problem) (result models.Result, err error) {

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

	population := votingOptimizationPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := votingOptimizationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := votingOptimizationTolerance
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
			individual[d] = votingOptimizationRangeLow + rand.Float64()*(votingOptimizationRangeHigh-votingOptimizationRangeLow)
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

		n := len(individuals)
		for i := range individuals {
			winner := rand.IntN(n)
			for range 2 {
				candidate := rand.IntN(n)
				if values[candidate] < values[winner] {
					winner = candidate
				}
			}

			for d := range individuals[i] {
				pullWinner := rand.Float64() * (individuals[winner][d] - individuals[i][d]) * 0.5
				pullBest := rand.Float64() * (best[d] - individuals[i][d]) * 0.3
				noise := (rand.Float64()*2 - 1) * 0.05
				individuals[i][d] = clamp(individuals[i][d]+pullWinner+pullBest+noise, votingOptimizationRangeLow, votingOptimizationRangeHigh)
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
		Method:     "voting_optimization",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
