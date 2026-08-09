package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	opticsInspiredPopulation = 20
	opticsInspiredMaxSteps   = 500
	opticsInspiredTolerance  = 1e-6
	opticsInspiredRangeLow   = -5.0
	opticsInspiredRangeHigh  = 5.0
)

// OpticsInspired minimizuje konfigurisanu benchmark funkciju koristeći Optics Inspired Optimization: konveksno
// sočivo preslikava jedinku bliže fokusu (best-u) dok konkavno sočivo divergentno rasipa jedinku dalje od fokusa,
// oba uz nezavisan šum koji sprečava potpuni kolaps na fokus
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func OpticsInspired(problem models.Problem) (result models.Result, err error) {

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

	population := opticsInspiredPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := opticsInspiredMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := opticsInspiredTolerance
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
			individual[d] = opticsInspiredRangeLow + rand.Float64()*(opticsInspiredRangeHigh-opticsInspiredRangeLow)
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

		focalLength := 1.0*(1-float64(steps)/float64(maxSteps)) + 0.1

		for i := range individuals {
			// čisto multiplikativno preslikavanje oko fokusa (bez nezavisnog aditivnog šuma) bi kolabiralo na
			// fokus čim se jedinka dovoljno približi best-u, jer bi razlika (x-best) sama teoretski konvergirala
			// ka nuli — mali nezavisan šum obezbeđuje da jedinke zadrže finu pretragu i blizu fokusa
			if rand.Float64() < 0.5 {
				for d := range individuals[i] {
					individuals[i][d] = clamp(best[d]+(individuals[i][d]-best[d])*focalLength*rand.Float64()+(rand.Float64()*2-1)*0.05, opticsInspiredRangeLow, opticsInspiredRangeHigh)
				}
			} else {
				for d := range individuals[i] {
					individuals[i][d] = clamp(best[d]-(individuals[i][d]-best[d])*focalLength*rand.Float64()*0.5+(rand.Float64()*2-1)*0.05, opticsInspiredRangeLow, opticsInspiredRangeHigh)
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
		Method:     "optics_inspired",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
