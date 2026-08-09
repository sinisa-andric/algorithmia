package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	monarchButterflyCount     = 20
	monarchButterflyMaxSteps  = 500
	monarchButterflyTolerance = 1e-6
	monarchButterflySpread    = 10.0
	monarchButterflyP         = 5.0 / 12.0
	monarchButterflyAdjust    = 0.05
	monarchButterflyAttract   = 0.1
	monarchButterflyBound     = 15.0
)

// MonarchButterfly minimizuje sphere funkciju koristeći Monarch Butterfly Optimization: bolja polovina populacije
// (Land1) se rekombinuje kroz migraciju između dve podpopulacije, dok se slabija polovina (Land2) privlači ka
// trenutno najboljem
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func MonarchButterfly(problem models.Problem) (result models.Result, err error) {

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

	count := monarchButterflyCount
	if v, ok := problem.Payload["butterflies"].(float64); ok {
		count = int(v)
	}
	if count < 2 {
		count = 2
	}

	maxSteps := monarchButterflyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := monarchButterflyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	population := make([][]float64, count)
	for i := range population {
		butterfly := make([]float64, dimensions)
		for d := range butterfly {
			butterfly[d] = problem.Point[d] + (rand.Float64()*2-1)*monarchButterflySpread
		}
		population[i] = butterfly
	}
	sortByFitness(fn, population)

	best := append([]float64(nil), population[0]...)
	bestValue := fn.Evaluate(best)

	globalBest := append([]float64(nil), best...)
	globalBestValue := bestValue

	land1Size := count / 2

	steps := 0
	for ; steps < maxSteps && globalBestValue >= tolerance; steps++ {

		next := make([][]float64, count)

		for i := 0; i < land1Size; i++ {
			butterfly := make([]float64, dimensions)
			for d := range butterfly {
				var source []float64
				if rand.Float64() <= monarchButterflyP {
					source = population[rand.IntN(land1Size)]
				} else {
					source = population[land1Size+rand.IntN(count-land1Size)]
				}
				butterfly[d] = source[d] + monarchButterflyAttract*(best[d]-source[d])

				if butterfly[d] > monarchButterflyBound {
					butterfly[d] = monarchButterflyBound
				} else if butterfly[d] < -monarchButterflyBound {
					butterfly[d] = -monarchButterflyBound
				}
			}
			next[i] = butterfly
		}

		for i := land1Size; i < count; i++ {
			butterfly := make([]float64, dimensions)
			for d := range butterfly {
				butterfly[d] = best[d] + monarchButterflyAdjust*(rand.Float64()*2-1)

				if butterfly[d] > monarchButterflyBound {
					butterfly[d] = monarchButterflyBound
				} else if butterfly[d] < -monarchButterflyBound {
					butterfly[d] = -monarchButterflyBound
				}
			}
			next[i] = butterfly
		}

		population = next
		sortByFitness(fn, population)

		best = append([]float64(nil), population[0]...)
		bestValue = fn.Evaluate(best)

		if bestValue < globalBestValue {
			globalBestValue = bestValue
			globalBest = append([]float64(nil), best...)
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "monarch_butterfly",
		Point:      globalBest,
		Value:      globalBestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
