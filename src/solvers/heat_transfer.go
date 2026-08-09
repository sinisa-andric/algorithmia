package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	heatTransferPopulation = 20
	heatTransferMaxSteps   = 500
	heatTransferTolerance  = 1e-6
	heatTransferRangeLow   = -5.0
	heatTransferRangeHigh  = 5.0
)

// HeatTransfer minimizuje konfigurisanu benchmark funkciju koristeći Heat Transfer Relation-based Optimization:
// svako telo svakog koraka nasumično prolazi kroz konvekciju (brz prenos ka best-u), kondukciju (sporiji prenos ka
// nasumičnom peer-u) ili zračenje (nezavisna nasumična emisija)
// problem.Point inicijalizuje populaciju tela i određuje njenu dimenzionalnost
func HeatTransfer(problem models.Problem) (result models.Result, err error) {

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

	population := heatTransferPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := heatTransferMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := heatTransferTolerance
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
			individual[d] = heatTransferRangeLow + rand.Float64()*(heatTransferRangeHigh-heatTransferRangeLow)
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

		tempDiff := 1 - float64(steps)/float64(maxSteps)

		for i := range individuals {
			mode := rand.Float64()

			if mode < 0.33 {
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+rand.Float64()*tempDiff*(best[d]-individuals[i][d]), heatTransferRangeLow, heatTransferRangeHigh)
				}
			} else if mode < 0.66 {
				peerIdx := i
				if population > 1 {
					peerIdx = rand.IntN(population)
					for peerIdx == i {
						peerIdx = rand.IntN(population)
					}
				}
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+0.3*rand.Float64()*tempDiff*(individuals[peerIdx][d]-individuals[i][d]), heatTransferRangeLow, heatTransferRangeHigh)
				}
			} else {
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*0.3*tempDiff, heatTransferRangeLow, heatTransferRangeHigh)
				}
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
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
		Method:     "heat_transfer",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
