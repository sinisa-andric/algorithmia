package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	rootGrowthPopulation = 20
	rootGrowthMaxSteps   = 500
	rootGrowthBranchProb = 0.3
	rootGrowthTolerance  = 1e-6
	rootGrowthRangeLow   = -5.0
	rootGrowthRangeHigh  = 5.0
)

// RootGrowth minimizuje konfigurisanu benchmark funkciju koristeći Root Growth Optimization: glavni koren raste
// ka najboljem rešenju uz gravitropizam koji jača tokom izvršavanja, dok bočni koreni povremeno granaju ka
// nasumičnom peer-u radi istraživanja prostora
// problem.Point inicijalizuje populaciju korenova i određuje njenu dimenzionalnost
func RootGrowth(problem models.Problem) (result models.Result, err error) {

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

	population := rootGrowthPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := rootGrowthMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	branchProb := rootGrowthBranchProb
	if v, ok := problem.Payload["branch_prob"].(float64); ok {
		branchProb = v
	}

	tolerance := rootGrowthTolerance
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
			individual[d] = rootGrowthRangeLow + rand.Float64()*(rootGrowthRangeHigh-rootGrowthRangeLow)
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
			if rand.Float64() < branchProb {
				peerIdx := i
				if population > 1 {
					peerIdx = rand.IntN(population)
					for peerIdx == i {
						peerIdx = rand.IntN(population)
					}
				}
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*(individuals[peerIdx][d]-individuals[i][d])*0.4, rootGrowthRangeLow, rootGrowthRangeHigh)
				}
			} else {
				gravitropism := 0.3 + 0.5*float64(steps)/float64(maxSteps)
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+rand.Float64()*(best[d]-individuals[i][d])*gravitropism, rootGrowthRangeLow, rootGrowthRangeHigh)
				}
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
		Method:     "root_growth",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
