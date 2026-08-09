package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	runnerRootPopulation = 20
	runnerRootMaxSteps   = 500
	runnerRootTolerance  = 1e-6
	runnerRootRangeLow   = -5.0
	runnerRootRangeHigh  = 5.0
)

// RunnerRoot minimizuje konfigurisanu benchmark funkciju koristeći Runner Root Algorithm: svaka biljka svakog
// koraka nasumično bira između naglog skoka runner-om u nasumičnu okolinu najboljeg rešenja (opadajućeg dometa) i
// fine lokalne pretrage sopstvenim korenčićima uz blagu vezu ka najboljem rešenju
// problem.Point inicijalizuje populaciju biljaka i određuje njenu dimenzionalnost
func RunnerRoot(problem models.Problem) (result models.Result, err error) {

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

	population := runnerRootPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := runnerRootMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := runnerRootTolerance
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
			individual[d] = runnerRootRangeLow + rand.Float64()*(runnerRootRangeHigh-runnerRootRangeLow)
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
			if rand.Float64() < 0.4 {
				for d := range individuals[i] {
					individuals[i][d] = clamp(best[d]+(rand.Float64()*2-1)*1.5*(1-float64(steps)/float64(maxSteps)), runnerRootRangeLow, runnerRootRangeHigh)
				}
			} else {
				for d := range individuals[i] {
					pull := rand.Float64() * (best[d] - individuals[i][d]) * 0.3
					individuals[i][d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*0.2+pull, runnerRootRangeLow, runnerRootRangeHigh)
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
		Method:     "runner_root",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
