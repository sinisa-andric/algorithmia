package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

const (
	henryGasPopulation  = 20
	henryGasMaxSteps    = 500
	henryGasNumClusters = 4
	henryGasTolerance   = 1e-6
	henryGasRangeLow    = -5.0
	henryGasRangeHigh   = 5.0
	henryGasCj          = 0.1
	henryGasGamma       = 1.0
	henryGasAlpha       = 1.0
)

// HenryGas minimizuje konfigurisanu benchmark funkciju koristeći Henry Gas Solubility Optimization: čestice su
// podeljene u klastere čija se Henrijeva konstanta i rastvorljivost menjaju sa opadajućom temperaturom, a svaka
// čestica se kreće ka globalnom best-u ponderisanom rastvorljivošću i ka najboljoj čestici sopstvenog klastera
// problem.Point inicijalizuje populaciju gasnih čestica i određuje njenu dimenzionalnost
func HenryGas(problem models.Problem) (result models.Result, err error) {

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

	population := henryGasPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := henryGasMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	numClusters := henryGasNumClusters
	if v, ok := problem.Payload["num_clusters"].(float64); ok {
		numClusters = int(v)
	}
	if numClusters < 1 {
		numClusters = 1
	}

	tolerance := henryGasTolerance
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
			individual[d] = henryGasRangeLow + rand.Float64()*(henryGasRangeHigh-henryGasRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
		cluster[i] = i % numClusters
	}

	h := make([]float64, numClusters)
	for c := range h {
		h[c] = 1.0
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

		temperature := math.Exp(-float64(steps) / float64(maxSteps))
		solubility := make([]float64, numClusters)
		for c := range h {
			h[c] *= math.Exp(-henryGasCj * (1/temperature - 1/298.15))
			solubility[c] = h[c]
		}

		bestInCluster := make([][]float64, numClusters)
		bestInClusterValue := make([]float64, numClusters)
		for c := range bestInClusterValue {
			bestInClusterValue[c] = math.Inf(1)
		}
		for i := range individuals {
			c := cluster[i]
			if values[i] < bestInClusterValue[c] {
				bestInClusterValue[c] = values[i]
				bestInCluster[c] = individuals[i]
			}
		}

		for i := range individuals {
			c := cluster[i]
			clusterBest := bestInCluster[c]
			if clusterBest == nil {
				clusterBest = individuals[i]
			}
			for d := range individuals[i] {
				toGlobal := rand.Float64() * henryGasGamma * (solubility[c]*best[d] - individuals[i][d])
				toCluster := rand.Float64() * henryGasAlpha * (clusterBest[d] - individuals[i][d])
				noise := (rand.Float64()*2 - 1) * 0.05
				individuals[i][d] = clamp(individuals[i][d]+toGlobal+toCluster+noise, henryGasRangeLow, henryGasRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		numReset := population / 10
		if numReset > 0 {
			order := make([]int, population)
			for i := range order {
				order[i] = i
			}
			sort.Slice(order, func(a, b int) bool {
				return values[order[a]] > values[order[b]]
			})
			for i := 0; i < numReset; i++ {
				idx := order[i]
				individual := make([]float64, dimensions)
				for d := range individual {
					individual[d] = henryGasRangeLow + rand.Float64()*(henryGasRangeHigh-henryGasRangeLow)
				}
				individuals[idx] = individual
				values[idx] = fn.Evaluate(individual)
			}
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
		Method:     "henry_gas",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
