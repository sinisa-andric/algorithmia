package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	brainstormingOptimizationPopulation  = 20
	brainstormingOptimizationMaxSteps    = 500
	brainstormingOptimizationNumClusters = 3
	brainstormingOptimizationTolerance   = 1e-6
	brainstormingOptimizationRangeLow    = -5.0
	brainstormingOptimizationRangeHigh   = 5.0
)

// BrainstormingOptimization minimizuje konfigurisanu benchmark funkciju koristeći Brainstorming Optimization
// Algorithm: ideje se kombinuju unutar sopstvenog klastera ili sa nasumičnom idejom iz drugog klastera, uz uvek
// prisutnu malu mutaciju
// problem.Point inicijalizuje populaciju ideja i određuje njenu dimenzionalnost
func BrainstormingOptimization(problem models.Problem) (result models.Result, err error) {

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

	population := brainstormingOptimizationPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := brainstormingOptimizationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	numClusters := brainstormingOptimizationNumClusters
	if v, ok := problem.Payload["num_clusters"].(float64); ok {
		numClusters = int(v)
	}
	if numClusters < 1 {
		numClusters = 1
	}

	tolerance := brainstormingOptimizationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	cluster := make([]int, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = brainstormingOptimizationRangeLow + rand.Float64()*(brainstormingOptimizationRangeHigh-brainstormingOptimizationRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
		cluster[i] = i % numClusters
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

		clusterCenter := make([][]float64, numClusters)
		clusterCount := make([]int, numClusters)
		for c := range clusterCenter {
			clusterCenter[c] = make([]float64, dimensions)
		}
		for i, individual := range individuals {
			c := cluster[i]
			clusterCount[c]++
			for d := range individual {
				clusterCenter[c][d] += individual[d]
			}
		}
		for c := range clusterCenter {
			if clusterCount[c] == 0 {
				continue
			}
			for d := range clusterCenter[c] {
				clusterCenter[c][d] /= float64(clusterCount[c])
			}
		}

		for i := range individuals {
			c := cluster[i]

			if rand.Float64() < 0.8 {
				for d := range individuals[i] {
					pull := rand.Float64() * (clusterCenter[c][d] - individuals[i][d]) * 0.4
					pullBest := rand.Float64() * (best[d] - individuals[i][d]) * 0.2
					individuals[i][d] = clamp(individuals[i][d]+pull+pullBest, brainstormingOptimizationRangeLow, brainstormingOptimizationRangeHigh)
				}
			} else {
				otherIdx := i
				for range 10 {
					candidate := rand.IntN(len(individuals))
					if cluster[candidate] != c {
						otherIdx = candidate
						break
					}
				}
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+rand.Float64()*(individuals[otherIdx][d]-individuals[i][d])*0.5, brainstormingOptimizationRangeLow, brainstormingOptimizationRangeHigh)
				}
			}

			for d := range individuals[i] {
				individuals[i][d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*0.1, brainstormingOptimizationRangeLow, brainstormingOptimizationRangeHigh)
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
		Method:     "brainstorming_optimization",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
