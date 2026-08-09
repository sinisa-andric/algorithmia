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
	equilibriumOptimizerPopulation = 20
	equilibriumOptimizerMaxSteps   = 500
	equilibriumOptimizerA1         = 2.0
	equilibriumOptimizerA2         = 1.0
	equilibriumOptimizerGP         = 0.5
	equilibriumOptimizerTolerance  = 1e-6
	equilibriumOptimizerRangeLow   = -5.0
	equilibriumOptimizerRangeHigh  = 5.0
	equilibriumOptimizerPoolBest   = 4
)

// EquilibriumOptimizer minimizuje konfigurisanu benchmark funkciju koristeći Equilibrium Optimizer: svaka jedinka
// teži ravnotežnoj koncentraciji nasumično izabranoj iz bazena četiri najbolje jedinke i njihovog proseka, uz
// eksponencijalno opadajući faktor F i stohastičku brzinu generisanja koja ubrizgava dodatnu perturbaciju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func EquilibriumOptimizer(problem models.Problem) (result models.Result, err error) {

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

	population := equilibriumOptimizerPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := equilibriumOptimizerMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	a1 := equilibriumOptimizerA1
	if v, ok := problem.Payload["a1"].(float64); ok {
		a1 = v
	}

	a2 := equilibriumOptimizerA2
	if v, ok := problem.Payload["a2"].(float64); ok {
		a2 = v
	}

	gp := equilibriumOptimizerGP
	if v, ok := problem.Payload["gp"].(float64); ok {
		gp = v
	}

	tolerance := equilibriumOptimizerTolerance
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
			individual[d] = equilibriumOptimizerRangeLow + rand.Float64()*(equilibriumOptimizerRangeHigh-equilibriumOptimizerRangeLow)
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

		order := make([]int, len(individuals))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool {
			return values[order[a]] < values[order[b]]
		})

		poolSize := equilibriumOptimizerPoolBest
		if poolSize > len(order) {
			poolSize = len(order)
		}
		pool := make([][]float64, 0, poolSize+1)
		avg := make([]float64, dimensions)
		for k := 0; k < poolSize; k++ {
			candidate := individuals[order[k]]
			pool = append(pool, candidate)
			for d := range avg {
				avg[d] += candidate[d]
			}
		}
		for d := range avg {
			avg[d] /= float64(poolSize)
		}
		pool = append(pool, avg)

		t := math.Pow(1-float64(steps)/float64(maxSteps), a2*float64(steps)/float64(maxSteps))

		for i := range individuals {
			cEq := pool[rand.IntN(len(pool))]

			for d := range individuals[i] {
				// lambda se drži van [0, malo] da bi se izbeglo eksplozivno deljenje u generation_rate/lambda
				// članu — pri vrednosti blizu nule taj termin bi bez ovog ograničenja divergirao
				lambda := 0.01 + rand.Float64()*0.98

				sign := 1.0
				if rand.Float64() < 0.5 {
					sign = -1.0
				}
				f := a1 * sign * (math.Exp(-lambda*t) - 1)

				generationRate := 0.0
				if rand.Float64() >= gp {
					generationRate = gp * rand.Float64() * (cEq[d] - lambda*individuals[i][d])
				}

				individuals[i][d] = clamp(cEq[d]+(individuals[i][d]-cEq[d])*f+generationRate*(1-f)/lambda, equilibriumOptimizerRangeLow, equilibriumOptimizerRangeHigh)
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
		Method:     "equilibrium_optimizer",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
