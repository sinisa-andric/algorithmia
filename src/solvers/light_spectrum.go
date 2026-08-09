package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	lightSpectrumPopulation = 20
	lightSpectrumMaxSteps   = 500
	lightSpectrumTolerance  = 1e-6
	lightSpectrumRangeLow   = -5.0
	lightSpectrumRangeHigh  = 5.0
)

// LightSpectrum minimizuje konfigurisanu benchmark funkciju koristeći Light Spectrum Optimizer: zraci sa dužom
// talasnom dužinom (lošijim fitnesom) prolaze kroz veću dispreziju prizme dok zraci bliski best-u fino konvergiraju
// ka njemu
// problem.Point inicijalizuje populaciju zraka i određuje njenu dimenzionalnost
func LightSpectrum(problem models.Problem) (result models.Result, err error) {

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

	population := lightSpectrumPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := lightSpectrumMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := lightSpectrumTolerance
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
			individual[d] = lightSpectrumRangeLow + rand.Float64()*(lightSpectrumRangeHigh-lightSpectrumRangeLow)
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
			wavelengthNorm := (values[i] - bestValue) / (worstValue - bestValue + 1e-10)
			dispersion := 0.1 + 0.9*wavelengthNorm

			for d := range individuals[i] {
				pull := rand.Float64() * (best[d] - individuals[i][d]) * (1 - wavelengthNorm*0.5)
				noise := (rand.Float64()*2 - 1) * dispersion * 0.3
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, lightSpectrumRangeLow, lightSpectrumRangeHigh)
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
		Method:     "light_spectrum",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
