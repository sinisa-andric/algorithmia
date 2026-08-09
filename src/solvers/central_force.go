package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	centralForcePopulation = 20
	centralForceMaxSteps   = 500
	centralForceAlphaCFO   = 2.0
	centralForceBetaCFO    = 2.0
	centralForceTolerance  = 1e-6
	centralForceRangeLow   = -5.0
	centralForceRangeHigh  = 5.0
	centralForceAccelBound = 3.0
)

// CentralForce minimizuje konfigurisanu benchmark funkciju koristeći Central Force Optimization: svaka probna masa
// ubrzava isključivo ka masama koje su trenutno bolje od nje (za razliku od gravitational_search.go, gde privlače
// SVE jedinke ponderisano normalizovanom masom), silom koja opada sa kubom rastojanja umesto linearno
// problem.Point inicijalizuje populaciju probnih masa i određuje njenu dimenzionalnost
func CentralForce(problem models.Problem) (result models.Result, err error) {

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

	population := centralForcePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := centralForceMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	alphaCfo := centralForceAlphaCFO
	if v, ok := problem.Payload["alpha_cfo"].(float64); ok {
		alphaCfo = v
	}

	betaCfo := centralForceBetaCFO
	if v, ok := problem.Payload["beta_cfo"].(float64); ok {
		betaCfo = v
	}

	tolerance := centralForceTolerance
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
			individual[d] = centralForceRangeLow + rand.Float64()*(centralForceRangeHigh-centralForceRangeLow)
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
			accel := make([]float64, dimensions)
			for j := range individuals {
				if j == i || values[j] >= values[i] {
					continue
				}
				diff := make([]float64, dimensions)
				for d := range diff {
					diff[d] = individuals[j][d] - individuals[i][d]
				}
				r := norm(diff) + 1e-10
				massDiff := values[i] - values[j]
				coeff := alphaCfo * math.Pow(massDiff, betaCfo) / (r * r * r)
				for d := range accel {
					accel[d] += coeff * diff[d]
				}
			}

			for d := range accel {
				accel[d] = clamp(accel[d], -centralForceAccelBound, centralForceAccelBound)
			}

			for d := range individuals[i] {
				individuals[i][d] = clamp(individuals[i][d]+0.5*accel[d]+(rand.Float64()*2-1)*0.05, centralForceRangeLow, centralForceRangeHigh)
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
		Method:     "central_force",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
