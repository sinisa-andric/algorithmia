package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	plantPropagationPopulation  = 20
	plantPropagationMaxSteps    = 500
	plantPropagationRunnerRatio = 0.5
	plantPropagationTolerance   = 1e-6
	plantPropagationRangeLow    = -5.0
	plantPropagationRangeHigh   = 5.0
)

// PlantPropagation minimizuje konfigurisanu benchmark funkciju koristeći Plant Propagation Algorithm: kvalitetne
// biljke šalju kratke izdanke koji fino pretražuju sopstvenu okolinu, dok slabije biljke šalju duge izdanke koji
// široko istražuju prostor uz blagu vezu ka najboljem rešenju
// problem.Point inicijalizuje populaciju biljaka i određuje njenu dimenzionalnost
func PlantPropagation(problem models.Problem) (result models.Result, err error) {

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

	population := plantPropagationPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := plantPropagationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	runnerRatio := plantPropagationRunnerRatio
	if v, ok := problem.Payload["runner_ratio"].(float64); ok {
		runnerRatio = v
	}

	tolerance := plantPropagationTolerance
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
			individual[d] = plantPropagationRangeLow + rand.Float64()*(plantPropagationRangeHigh-plantPropagationRangeLow)
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

		worstValue := values[0]
		for _, v := range values {
			if v > worstValue {
				worstValue = v
			}
		}

		for i := range individuals {
			quality := (worstValue - values[i]) / (worstValue - bestValue + 1e-10)

			if quality > runnerRatio {
				// čisti simetrični šum bez ikakve veze sa best-om (kako je prvobitno specificirano) svodi
				// finu pretragu na nasumičnu šetnju bez usmerenog pomaka čim se jedinka približi best-u —
				// to sistemski usporava nalaženje poboljšanja iznad tolerance unutar noImprove budžeta i
				// prevremeno zaustavlja izvršavanje (izmereno: zaglavljivanje na sphere u testiranju). Mali
				// pull član obezbeđuje usmereno fino rafiniranje dok amplituda šuma i dalje opada sa quality
				amplitude := 0.2 * (1 - quality)
				for d := range individuals[i] {
					pull := rand.Float64() * (best[d] - individuals[i][d]) * 0.1
					noise := (rand.Float64()*2 - 1) * amplitude
					individuals[i][d] = clamp(individuals[i][d]+pull+noise, plantPropagationRangeLow, plantPropagationRangeHigh)
				}
			} else {
				for d := range individuals[i] {
					pull := rand.Float64() * (best[d] - individuals[i][d]) * 0.2
					individuals[i][d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*1.2+pull, plantPropagationRangeLow, plantPropagationRangeHigh)
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
		Method:     "plant_propagation",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
